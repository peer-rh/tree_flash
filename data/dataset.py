"""
Stage-2 dataset for tree-flash training.

HDF5 layout (written by data_pipeline/stage2_trees.py)
-------------------------------------------------------
    prompt_ids               : vlen int64   [N]
        Tokenised prompt per sequence.

    response_ids             : vlen int64   [N]
        Tokenised response per sequence.

    response_probs           : float32      [T]
        p(response[t] | context[:t]) for every response position t, all
        sequences concatenated.  Provides primary-path cumprod weights.

    continuation_trees       : int64        [T, subtree_size]
        Subtree token ids at each response position.  Only selected
        positions (num_trees_per_seq per sequence) are filled; the rest
        contain IGNORE_IDX = -1.  T = sum of all response lengths.

    continuation_trees_probs : float32      [T, subtree_size]
        Individual AR probability of each subtree token:
            p_target(token | context + ancestor path)
        0.0 for IGNORE_IDX slots.

    selected_positions       : vlen int64   [N]
        Response-relative indices t of the selected subtree positions.

    sequence_offsets         : int64        [N+1]
        Row-pointer index into the flat datasets.
        Sequence n occupies rows sequence_offsets[n] : sequence_offsets[n+1].

Dataset item
------------
Each item corresponds to one valid (sequence, anchor-t) pair, where
"valid" means t + n_subtrees <= S_R (primary path fits within response)
AND t is a selected position (subtree data exists at t).

The full training tree is assembled on the fly:

    tree_tokens [tree_size]:
        [:n_subtrees]          = response_ids[t : t + n_subtrees]    (primary path)
        [n_subtrees + i*ss :
         n_subtrees + (i+1)*ss] = continuation_trees[base + t + i]   (subtree at node i)
        where ss = subtree_size

    tree_probs [tree_size]:
        [:n_subtrees]          = response_probs[base + t : base + t + n_subtrees]
        [n_subtrees + i*ss :
         n_subtrees + (i+1)*ss] = continuation_trees_probs[base + t + i]

    context_ids [ctx_len]:
        last ctx_len tokens of prompt + response[:t]

A train/val split is implemented via the ``split`` argument:
    "train" → first  floor(N * train_frac) sequences
    "val"   → remaining sequences
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

IGNORE_IDX: int = -1


class Stage2Dataset(Dataset):
    """
    Memory-efficient HDF5-backed dataset.

    One item = one valid (sequence, anchor-t) pair.

    The HDF5 file is opened once per DataLoader worker (lazy, via _open()).
    Only the index arrays are held in memory; all token data is fetched on
    demand.

    Parameters
    ----------
    path        : path to the HDF5 file produced by stage2_trees.py
    ctx_len     : context window length
    n_subtrees  : primary path length used during training; determines which
                  anchors are valid (t + n_subtrees <= S_R)
    split       : "train" or "val"
    train_frac  : fraction of sequences used for training (default 0.99)
    """

    def __init__(
        self,
        path: str,
        ctx_len: int,
        n_subtrees: int,
        split: str = "train",
        train_frac: float = 0.99,
    ) -> None:
        assert split in ("train", "val"), f"split must be 'train'|'val', got {split!r}"
        self.path = path
        self.ctx_len = ctx_len
        self.n_subtrees = n_subtrees
        self.split = split

        import h5py
        with h5py.File(path, "r") as f:
            for key in (
                "prompt_ids", "response_ids",
                "response_probs",
                "continuation_trees", "continuation_trees_probs",
                "selected_positions", "sequence_offsets",
            ):
                assert key in f, f"HDF5 missing dataset '{key}'"

            N = f["prompt_ids"].shape[0]
            offsets = f["sequence_offsets"][:]          # [N+1] int64
            sel_raw = [f["selected_positions"][n] for n in range(N)]

        n_train = int(N * train_frac)
        seq_range = range(0, n_train) if split == "train" else range(n_train, N)

        # Build flat index: (seq_idx, h5_base_row, t)
        # h5_base_row = offsets[n]  (start row for this sequence in flat datasets)
        # Valid condition: t + n_subtrees <= S_R_n  where S_R_n = offsets[n+1] - offsets[n]
        index: list[tuple[int, int, int]] = []
        for n in seq_range:
            base_row = int(offsets[n])
            S_R_n = int(offsets[n + 1]) - base_row
            for t in sel_raw[n].tolist():
                t = int(t)
                if t + n_subtrees <= S_R_n:
                    index.append((n, base_row, t))

        if index:
            idx_arr = np.array(index, dtype=np.int64)  # [M, 3]
            self._seq_idx  = idx_arr[:, 0]
            self._h5_base  = idx_arr[:, 1]   # offsets[n]
            self._t        = idx_arr[:, 2]
        else:
            self._seq_idx  = np.empty(0, dtype=np.int64)
            self._h5_base  = np.empty(0, dtype=np.int64)
            self._t        = np.empty(0, dtype=np.int64)

        self._h5 = None  # opened lazily per worker

    def _open(self) -> None:
        """Open HDF5 handle lazily so each DataLoader worker gets its own fd."""
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.path, "r")

    def __len__(self) -> int:
        return len(self._seq_idx)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        context_ids : [ctx_len]    int64   — last ctx_len tokens of prompt+response[:t]
        tree_tokens : [tree_size]  int64   — primary path + subtrees; IGNORE_IDX=-1 for
                                             unfilled subtree nodes
        tree_probs  : [tree_size]  float32 — individual AR probs; 0.0 for unfilled nodes
        """
        self._open()
        n         = int(self._seq_idx[idx])
        base_row  = int(self._h5_base[idx])   # = offsets[n]
        t         = int(self._t[idx])

        n_sub = self.n_subtrees

        # ── Context ids ───────────────────────────────────────────────────
        prompt_ids   = self._h5["prompt_ids"][n]    # np int64 [S_P]
        response_ids = self._h5["response_ids"][n]  # np int64 [S_R]

        full_prefix = np.concatenate([prompt_ids, response_ids[:t]])
        if len(full_prefix) >= self.ctx_len:
            ctx = full_prefix[-self.ctx_len:]
        else:
            pad_len = self.ctx_len - len(full_prefix)
            pad_val = int(full_prefix[0]) if len(full_prefix) > 0 else 0
            ctx = np.concatenate([
                np.full(pad_len, pad_val, dtype=np.int64),
                full_prefix,
            ])

        # ── Primary path ─────────────────────────────────────────────────
        primary_tokens = response_ids[t : t + n_sub]          # [n_sub]
        # Probs for primary path: response_probs[base_row + t : base_row + t + n_sub]
        primary_probs = self._h5["response_probs"][
            base_row + t : base_row + t + n_sub
        ]  # [n_sub]

        # ── Subtrees for each primary node ────────────────────────────────
        # continuation_trees[base_row + t + i] = subtree at primary node i
        # Read n_sub consecutive rows in one slice
        subtrees = self._h5["continuation_trees"][
            base_row + t : base_row + t + n_sub
        ]  # [n_sub, subtree_size]
        subtree_probs = self._h5["continuation_trees_probs"][
            base_row + t : base_row + t + n_sub
        ]  # [n_sub, subtree_size]

        # ── Assemble full tree ────────────────────────────────────────────
        # Layout: [primary_0, ..., primary_{n-1},
        #          subtree_at_0[0..ss-1], subtree_at_1[0..ss-1], ...]
        tree_tokens = np.concatenate([
            primary_tokens,
            subtrees.flatten(),
        ])  # [tree_size]
        tree_probs_arr = np.concatenate([
            primary_probs,
            subtree_probs.flatten(),
        ])  # [tree_size]

        return (
            torch.from_numpy(ctx.astype(np.int64)).long(),
            torch.from_numpy(tree_tokens.astype(np.int64)).long(),
            torch.from_numpy(tree_probs_arr.astype(np.float32)).float(),
        )
