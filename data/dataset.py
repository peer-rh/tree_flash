"""
Stage-2 dataset for tree-flash training.

Expected HDF5 layout
--------------------
The file contains three top-level datasets:

    context_ids      : [N, ctx_len]    int64
        Tokenised context (padded to ctx_len with pad_token_id).

    tree_tokens      : [N, tree_size]  int64
        Ground-truth tokens at each tree node.
        Node ordering follows TreeSpec's BFS layout (primary path first,
        then sub_tree copies in primary-path order).

    cumprod_weights  : [N, tree_size]  float32
        Per-node CumProd loss weight = ∏ p_target(token_i | prefix) along
        the path from root to node i.  Pre-computed offline by the stage-2
        pipeline using the frozen target model.

A train/val split is implemented via an ``split`` argument:
    "train" → first  floor(N * train_frac) rows
    "val"   → remaining rows
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


class Stage2Dataset(Dataset):
    """
    Memory-efficient HDF5-backed dataset. The HDF5 file is opened once per
    worker and rows are fetched on demand — no full-file preload.

    Parameters
    ----------
    path        : path to the HDF5 file
    split       : "train" or "val"
    train_frac  : fraction of data used for training (default 0.99)
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        train_frac: float = 0.99,
    ) -> None:
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"
        self.path = path
        self.split = split
        self.train_frac = train_frac

        # Open once to read length and validate keys; worker copies re-open lazily.
        import h5py
        with h5py.File(path, "r") as f:
            n_total = f["context_ids"].shape[0]
            # Validate expected keys
            for key in ("context_ids", "tree_tokens", "cumprod_weights"):
                assert key in f, f"HDF5 file missing dataset '{key}'"

        n_train = int(n_total * train_frac)
        if split == "train":
            self._start, self._end = 0, n_train
        else:
            self._start, self._end = n_train, n_total

        self._h5 = None  # lazily opened per worker (see _open)

    def _open(self) -> None:
        """Open HDF5 handle lazily so each DataLoader worker gets its own fd."""
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.path, "r")

    def __len__(self) -> int:
        return self._end - self._start

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        context_ids     : [ctx_len]    int64
        tree_tokens     : [tree_size]  int64
        cumprod_weights : [tree_size]  float32
        """
        self._open()
        row = self._start + idx
        context_ids = torch.from_numpy(self._h5["context_ids"][row]).long()
        tree_tokens = torch.from_numpy(self._h5["tree_tokens"][row]).long()
        cumprod_weights = torch.from_numpy(self._h5["cumprod_weights"][row]).float()
        return context_ids, tree_tokens, cumprod_weights
