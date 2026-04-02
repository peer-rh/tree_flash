from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler, Subset, random_split

from stage2_v2 import (
    GeneratedAnchorTree,
    GeneratedSequenceTree,
    SequenceTreeNode,
    flush_stage2_v2_hdf5,
    initialize_stage2_v2_hdf5,
)


IGNORE_IDX = -1


def write_stage1_jsonl(records: Iterable[dict[str, str]], output_path: str | Path) -> None:
    """Write Stage 1 prompt/response pairs as JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_stage1_jsonl(path: str | Path) -> list[dict[str, str]]:
    """Read Stage 1 prompt/response JSONL records into memory."""
    records: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_sequence_tree_hdf5(
    *,
    output_path: str | Path,
    sequences: list[GeneratedSequenceTree],
    attrs: dict[str, Any],
    prob_dtype=np.float32,
) -> None:
    """Write Stage 2 v2-style sequence trees into one HDF5 file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as hf:
        initialize_stage2_v2_hdf5(hf, prob_dtype=prob_dtype, attrs=attrs)
        flush_stage2_v2_hdf5(
            hf,
            sequences,
            n_sequences_written=0,
            n_main_path_ids_written=0,
            n_anchors_written=0,
            n_nodes_written=0,
            prob_dtype=prob_dtype,
        )


@dataclass
class PackedBatch:
    """One packed training batch built from Stage 2 v2 sequence trees.

    Shapes:
    - `input_ids`: `(batch_size, pack_length)`
    - `position_ids`: `(batch_size, pack_length)`
    - `document_mask`: `(batch_size, pack_length)`
    - `context_valid_mask`: `(batch_size, pack_length)`
    - `anchor_positions`: `(batch_size, num_anchors)`
    - `anchor_document_ids`: `(batch_size, num_anchors)`
    - `anchor_valid_mask`: `(batch_size, num_anchors)`
    - `tree_labels`: `(batch_size, num_anchors, tree_size)`
    - `tree_noise_ids`: `(batch_size, num_anchors, tree_size)`
    - `tree_position_ids`: `(batch_size, num_anchors, tree_size)`
    - `tree_cum_probs`: `(batch_size, num_anchors, tree_size)`
    - `tree_valid_mask`: `(batch_size, num_anchors, tree_size)`
    - `tree_parent_indices`: `(batch_size, num_anchors, tree_size)`
    - `tree_depths`: `(batch_size, num_anchors, tree_size)`
    - `tree_node_ranks`: `(batch_size, num_anchors, tree_size)`
    - `tree_primary_path_mask`: `(batch_size, num_anchors, tree_size)`
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    document_mask: torch.Tensor
    context_valid_mask: torch.Tensor
    anchor_positions: torch.Tensor
    anchor_document_ids: torch.Tensor
    anchor_valid_mask: torch.Tensor
    tree_labels: torch.Tensor
    tree_noise_ids: torch.Tensor
    tree_position_ids: torch.Tensor
    tree_cum_probs: torch.Tensor
    tree_valid_mask: torch.Tensor
    tree_parent_indices: torch.Tensor
    tree_depths: torch.Tensor
    tree_node_ranks: torch.Tensor
    tree_primary_path_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def num_anchors(self) -> int:
        return int(self.anchor_positions.shape[1])

    @property
    def tree_size(self) -> int:
        return int(self.tree_labels.shape[-1]) if self.tree_labels.ndim == 3 else 0

    def to(self, device: torch.device | str) -> "PackedBatch":
        return PackedBatch(
            input_ids=self.input_ids.to(device),
            position_ids=self.position_ids.to(device),
            document_mask=self.document_mask.to(device),
            context_valid_mask=self.context_valid_mask.to(device),
            anchor_positions=self.anchor_positions.to(device),
            anchor_document_ids=self.anchor_document_ids.to(device),
            anchor_valid_mask=self.anchor_valid_mask.to(device),
            tree_labels=self.tree_labels.to(device),
            tree_noise_ids=self.tree_noise_ids.to(device),
            tree_position_ids=self.tree_position_ids.to(device),
            tree_cum_probs=self.tree_cum_probs.to(device),
            tree_valid_mask=self.tree_valid_mask.to(device),
            tree_parent_indices=self.tree_parent_indices.to(device),
            tree_depths=self.tree_depths.to(device),
            tree_node_ranks=self.tree_node_ranks.to(device),
            tree_primary_path_mask=self.tree_primary_path_mask.to(device),
        )


@dataclass
class DataConfig:
    path: str
    eval_path: str | None = None
    train_split: float = 0.95
    batch_size: int = 8
    pack_length: int = 3072
    num_anchors: int = 8
    num_workers: int = 0
    seed: int = 42
    shuffle: bool = True
    drop_last: bool = False
    max_train_sequences: int | None = None
    max_eval_sequences: int | None = None
    training_tree_size: int = 16


class SequenceTreeDataset(Dataset):
    """Read the Stage 2 v2 sequence-tree HDF5 format used by the cleaned path."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with h5py.File(self.path, "r") as hf:
            self.main_offsets = hf["main_path_offsets"][:]
            self.sequence_anchor_offsets = hf["sequence_anchor_offsets"][:]
            self.anchor_node_offsets = hf["anchor_node_offsets"][:]
            self.response_start_positions = hf["response_start_positions"][:]
            self.record_idx = hf["record_idx"][:]
            self.format_version = str(hf.attrs.get("format_version", ""))
        self.total_lengths = tuple(
            int(end - start)
            for start, end in zip(self.main_offsets[:-1], self.main_offsets[1:], strict=True)
        )
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.main_offsets) - 1

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _ensure_open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __getitem__(self, idx: int) -> dict[str, Any]:
        hf = self._ensure_open()
        main_start = int(self.main_offsets[idx])
        main_end = int(self.main_offsets[idx + 1])
        anchor_start = int(self.sequence_anchor_offsets[idx])
        anchor_end = int(self.sequence_anchor_offsets[idx + 1])

        anchors: list[dict[str, Any]] = []
        for anchor_idx in range(anchor_start, anchor_end):
            node_start = int(self.anchor_node_offsets[anchor_idx])
            node_end = int(self.anchor_node_offsets[anchor_idx + 1])
            anchors.append(
                {
                    "anchor_main_path_position": int(hf["anchor_main_path_positions"][anchor_idx]),
                    "anchor_next_token_prob": float(hf["anchor_next_token_probs"][anchor_idx]),
                    "node_token_ids": torch.tensor(hf["node_token_ids"][node_start:node_end], dtype=torch.long),
                    "node_parent_indices": torch.tensor(hf["node_parent_indices"][node_start:node_end], dtype=torch.long),
                    "node_depths": torch.tensor(hf["node_depths"][node_start:node_end], dtype=torch.long),
                    "node_local_probs": torch.tensor(hf["node_local_probs"][node_start:node_end], dtype=torch.float32),
                    "node_path_probs": torch.tensor(hf["node_path_probs"][node_start:node_end], dtype=torch.float32),
                    "node_ranks": torch.tensor(hf["node_ranks"][node_start:node_end], dtype=torch.long),
                    "node_main_path_positions": torch.tensor(
                        hf["node_main_path_positions"][node_start:node_end],
                        dtype=torch.long,
                    ),
                    "node_is_main_path": torch.tensor(hf["node_is_main_path"][node_start:node_end], dtype=torch.bool),
                    "node_first_child": torch.tensor(hf["node_first_child"][node_start:node_end], dtype=torch.long),
                    "node_child_count": torch.tensor(hf["node_child_count"][node_start:node_end], dtype=torch.long),
                }
            )
        main_path_ids = torch.tensor(hf["main_path_ids"][main_start:main_end], dtype=torch.long)
        return {
            "main_path_ids": main_path_ids,
            "response_start_position": int(self.response_start_positions[idx]),
            "anchors": anchors,
            "record_idx": int(self.record_idx[idx]),
            "total_len": self.total_lengths[idx],
        }


def _build_anchor_lookup(sample: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(anchor["anchor_main_path_position"]): anchor for anchor in sample["anchors"]}


def _anchor_child_indices(anchor: dict[str, Any], node_idx: int) -> list[int]:
    first_child = int(anchor["node_first_child"][node_idx].item())
    child_count = int(anchor["node_child_count"][node_idx].item())
    if first_child < 0 or child_count <= 0:
        return []
    return list(range(first_child, first_child + child_count))


def _weighted_choice(rng: random.Random, weights: list[float]) -> int:
    total = sum(max(float(weight), 0.0) for weight in weights)
    if total <= 0:
        return rng.randrange(len(weights))
    target = rng.random() * total
    running = 0.0
    for idx, weight in enumerate(weights):
        running += max(float(weight), 0.0)
        if target <= running:
            return idx
    return len(weights) - 1


def _children_for_node(
    *,
    sample: dict[str, Any],
    anchor_lookup: dict[int, dict[str, Any]],
    node_ref: tuple[str, int] | tuple[str, int, int],
) -> list[dict[str, Any]]:
    main_path_ids = sample["main_path_ids"]
    if node_ref[0] == "main":
        main_pos = int(node_ref[1])
        anchor = anchor_lookup.get(main_pos)
        if anchor is not None:
            child_indices = _anchor_child_indices(anchor, 0)
            children: list[dict[str, Any]] = []
            for child_idx in child_indices:
                is_main_path = bool(anchor["node_is_main_path"][child_idx].item())
                child_main_pos = int(anchor["node_main_path_positions"][child_idx].item())
                child_ref: tuple[str, int] | tuple[str, int, int]
                if is_main_path and child_main_pos >= 0:
                    child_ref = ("main", child_main_pos)
                else:
                    child_ref = ("anchor", main_pos, child_idx)
                children.append(
                    {
                        "ref": child_ref,
                        "token_id": int(anchor["node_token_ids"][child_idx].item()),
                        "local_prob": float(anchor["node_local_probs"][child_idx].item()),
                        "rank": int(anchor["node_ranks"][child_idx].item()),
                        "main_path_position": child_main_pos,
                        "is_main_path": is_main_path,
                    }
                )
            return children
        next_main_pos = main_pos + 1
        if next_main_pos >= int(main_path_ids.numel()):
            return []
        return [
            {
                "ref": ("main", next_main_pos),
                "token_id": int(main_path_ids[next_main_pos].item()),
                "local_prob": 1.0,
                "rank": 0,
                "main_path_position": next_main_pos,
                "is_main_path": True,
            }
        ]

    _, anchor_pos, node_idx = node_ref
    anchor = anchor_lookup.get(int(anchor_pos))
    if anchor is None:
        return []
    children = []
    for child_idx in _anchor_child_indices(anchor, int(node_idx)):
        is_main_path = bool(anchor["node_is_main_path"][child_idx].item())
        child_main_pos = int(anchor["node_main_path_positions"][child_idx].item())
        child_ref = ("main", child_main_pos) if is_main_path and child_main_pos >= 0 else ("anchor", int(anchor_pos), child_idx)
        children.append(
            {
                "ref": child_ref,
                "token_id": int(anchor["node_token_ids"][child_idx].item()),
                "local_prob": float(anchor["node_local_probs"][child_idx].item()),
                "rank": int(anchor["node_ranks"][child_idx].item()),
                "main_path_position": child_main_pos,
                "is_main_path": is_main_path,
            }
        )
    return children


def sample_training_subtree(
    *,
    sample: dict[str, Any],
    anchor_local_position: int,
    training_tree_size: int,
    mask_token_id: int,
    rng: random.Random,
    deterministic: bool,
) -> dict[str, torch.Tensor]:
    """Sample one dynamic training subtree from a stored sequence tree.

    Output shapes:
    - `tree_labels`: `(tree_size,)`
    - `tree_noise_ids`: `(tree_size,)`
    - `tree_position_ids`: `(tree_size,)`
    - `tree_cum_probs`: `(tree_size,)`
    - `tree_valid_mask`: `(tree_size,)`
    - `tree_parent_indices`: `(tree_size,)`
    - `tree_depths`: `(tree_size,)`
    - `tree_node_ranks`: `(tree_size,)`
    - `tree_primary_path_mask`: `(tree_size,)`
    """
    main_path_ids = sample["main_path_ids"]
    if training_tree_size <= 0:
        raise ValueError(f"training_tree_size must be positive, got {training_tree_size}.")
    if anchor_local_position < 0 or anchor_local_position >= int(main_path_ids.numel()):
        raise IndexError(f"anchor_local_position {anchor_local_position} is out of range.")

    anchor_lookup = _build_anchor_lookup(sample)
    kept_refs: dict[tuple[Any, ...], int] = {("main", int(anchor_local_position)): 0}
    tree_labels = [int(main_path_ids[anchor_local_position].item())]
    tree_parent_indices = [-1]
    tree_depths = [0]
    tree_node_ranks = [0]
    tree_position_ids = [int(anchor_local_position)]
    tree_cum_probs = [1.0]
    tree_valid_mask = [True]
    tree_primary_path_mask = [True]

    frontier: list[dict[str, Any]] = []
    frontier_refs: set[tuple[Any, ...]] = set()

    def push_children(parent_local_idx: int, parent_ref: tuple[Any, ...], parent_path_prob: float, parent_depth: int) -> None:
        for child in _children_for_node(sample=sample, anchor_lookup=anchor_lookup, node_ref=parent_ref):
            child_ref = tuple(child["ref"])
            if child_ref in kept_refs or child_ref in frontier_refs:
                continue
            local_prob = max(float(child["local_prob"]), 0.0)
            frontier.append(
                {
                    "ref": child_ref,
                    "token_id": int(child["token_id"]),
                    "parent_index": parent_local_idx,
                    "depth": parent_depth + 1,
                    "path_prob": parent_path_prob * local_prob,
                    "rank": int(child["rank"]),
                    "main_path_position": int(child["main_path_position"]),
                    "is_main_path": bool(child["is_main_path"]),
                }
            )
            frontier_refs.add(child_ref)

    root_ref = ("main", int(anchor_local_position))
    push_children(parent_local_idx=0, parent_ref=root_ref, parent_path_prob=1.0, parent_depth=0)

    while frontier and len(tree_labels) < training_tree_size:
        if deterministic:
            chosen_idx = max(range(len(frontier)), key=lambda idx: (float(frontier[idx]["path_prob"]), -idx))
        else:
            chosen_idx = _weighted_choice(rng, [entry["path_prob"] for entry in frontier])
        chosen = frontier.pop(chosen_idx)
        frontier_refs.discard(tuple(chosen["ref"]))
        local_idx = len(tree_labels)
        kept_refs[tuple(chosen["ref"])] = local_idx
        tree_labels.append(int(chosen["token_id"]))
        tree_parent_indices.append(int(chosen["parent_index"]))
        tree_depths.append(int(chosen["depth"]))
        tree_node_ranks.append(int(chosen["rank"]))
        tree_position_ids.append(int(anchor_local_position + chosen["depth"]))
        tree_cum_probs.append(float(chosen["path_prob"]))
        tree_valid_mask.append(True)
        tree_primary_path_mask.append(bool(chosen["is_main_path"]))
        push_children(
            parent_local_idx=local_idx,
            parent_ref=tuple(chosen["ref"]),
            parent_path_prob=float(chosen["path_prob"]),
            parent_depth=int(chosen["depth"]),
        )

    tree_size = len(tree_labels)
    tree_noise_ids = torch.full((tree_size,), mask_token_id, dtype=torch.long)
    tree_noise_ids[0] = tree_labels[0]
    return {
        "tree_labels": torch.tensor(tree_labels, dtype=torch.long),
        "tree_noise_ids": tree_noise_ids,
        "tree_position_ids": torch.tensor(tree_position_ids, dtype=torch.long),
        "tree_cum_probs": torch.tensor(tree_cum_probs, dtype=torch.float32),
        "tree_valid_mask": torch.tensor(tree_valid_mask, dtype=torch.bool),
        "tree_parent_indices": torch.tensor(tree_parent_indices, dtype=torch.long),
        "tree_depths": torch.tensor(tree_depths, dtype=torch.long),
        "tree_node_ranks": torch.tensor(tree_node_ranks, dtype=torch.long),
        "tree_primary_path_mask": torch.tensor(tree_primary_path_mask, dtype=torch.bool),
    }


def _pack_items_into_rows(
    items: list[tuple[Any, int]],
    *,
    pack_length: int,
) -> list[list[tuple[Any, int, int]]]:
    rows: list[dict[str, Any]] = []
    for item, total_len in sorted(items, key=lambda value: value[1], reverse=True):
        placed = False
        for row in rows:
            if row["used"] + total_len <= pack_length:
                doc_id = len(row["docs"]) + 1
                row["docs"].append((item, row["used"], doc_id))
                row["used"] += total_len
                placed = True
                break
        if not placed:
            rows.append({"used": total_len, "docs": [(item, 0, 1)]})
    return [row["docs"] for row in rows]


@dataclass(frozen=True)
class _PendingIndex:
    uid: int
    index: int
    current_epoch: bool


class PackedBatchSampler(BatchSampler):
    """Emit dataset indices that pack into a fixed number of rows per batch."""

    def __init__(
        self,
        *,
        sample_lengths: list[int],
        pack_length: int,
        packed_batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
        persistent_state: bool,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        super().__init__(
            sampler=SequentialSampler(range(len(sample_lengths))),
            batch_size=packed_batch_size,
            drop_last=drop_last,
        )
        if packed_batch_size <= 0:
            raise ValueError(f"packed_batch_size must be positive, got {packed_batch_size}.")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}.")
        self.pack_length = pack_length
        self.packed_batch_size = packed_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.persistent_state = persistent_state
        self.num_replicas = num_replicas
        self.rank = rank
        self._rng = random.Random(seed)
        self._next_uid = 0
        self._pending: list[tuple[int, int]] = []
        self._eligible = [(idx, total_len) for idx, total_len in enumerate(sample_lengths) if int(total_len) <= pack_length]
        if not self._eligible:
            raise ValueError(f"No dataset samples fit within pack_length={pack_length}.")
        self._eligible_indices = [idx for idx, _ in self._eligible]
        self._length_by_index = {idx: int(total_len) for idx, total_len in self._eligible}
        self._order: list[int] = []
        self._cursor = 0

    def _next_stream_index(self, order: list[int], cursor: int, rng: random.Random) -> tuple[int, list[int], int]:
        if not order or cursor >= len(order):
            order = self._eligible_indices.copy()
            if self.shuffle:
                rng.shuffle(order)
            cursor = 0
        return order[cursor], order, cursor + 1

    def _pack_pending(self, pending: list[_PendingIndex]) -> list[list[tuple[_PendingIndex, int, int]]]:
        items = [(entry, self._length_by_index[entry.index]) for entry in pending]
        return _pack_items_into_rows(items, pack_length=self.pack_length)

    def _run_epoch(
        self,
        *,
        order: list[int],
        cursor: int,
        pending_pairs: list[tuple[int, int]],
        next_uid: int,
        rng: random.Random,
        collect_batches: bool,
    ) -> tuple[list[list[int]], list[tuple[int, int]], list[int], int, int]:
        pending = [_PendingIndex(uid=uid, index=index, current_epoch=True) for uid, index in pending_pairs]
        remaining_primary = max(len(self._eligible_indices) - len(pending), 0)
        emitted_batches: list[list[int]] = []
        batch_count = 0

        while True:
            rows = self._pack_pending(pending)
            if len(rows) >= self.packed_batch_size:
                consumed_uids = {
                    entry.uid
                    for docs in rows[: self.packed_batch_size]
                    for entry, _, _ in docs
                }
                batch_indices = [entry.index for entry in pending if entry.uid in consumed_uids]
                pending = [entry for entry in pending if entry.uid not in consumed_uids]
                batch_count += 1
                if collect_batches:
                    emitted_batches.append(batch_indices)
                if remaining_primary == 0 and not any(entry.current_epoch for entry in pending):
                    break
                continue

            if remaining_primary > 0:
                index, order, cursor = self._next_stream_index(order, cursor, rng)
                pending.append(_PendingIndex(uid=next_uid, index=index, current_epoch=True))
                next_uid += 1
                remaining_primary -= 1
                continue

            if any(entry.current_epoch for entry in pending):
                if self.drop_last:
                    pending = [entry for entry in pending if not entry.current_epoch]
                    break
                index, order, cursor = self._next_stream_index(order, cursor, rng)
                pending.append(_PendingIndex(uid=next_uid, index=index, current_epoch=False))
                next_uid += 1
                continue
            break

        next_pending = [(entry.uid, entry.index) for entry in pending]
        if not collect_batches:
            emitted_batches = [[] for _ in range(batch_count)]
        return emitted_batches, next_pending, order, cursor, next_uid

    def _shard_batches(self, emitted_batches: list[list[int]]) -> list[list[int]]:
        if self.num_replicas == 1:
            return emitted_batches
        if not emitted_batches:
            return []
        remainder = len(emitted_batches) % self.num_replicas
        if remainder != 0:
            if self.drop_last:
                emitted_batches = emitted_batches[: len(emitted_batches) - remainder]
            else:
                emitted_batches = emitted_batches + emitted_batches[: self.num_replicas - remainder]
        return emitted_batches[self.rank :: self.num_replicas]

    def __iter__(self):
        if self.persistent_state:
            order = self._order.copy()
            cursor = self._cursor
            pending_pairs = self._pending.copy()
            next_uid = self._next_uid
            rng = self._rng
        else:
            order = []
            cursor = 0
            pending_pairs = []
            next_uid = 0
            rng = random.Random(self.seed)

        emitted_batches, next_pending, next_order, next_cursor, next_uid = self._run_epoch(
            order=order,
            cursor=cursor,
            pending_pairs=pending_pairs,
            next_uid=next_uid,
            rng=rng,
            collect_batches=True,
        )
        if self.persistent_state:
            self._pending = next_pending
            self._order = next_order
            self._cursor = next_cursor
            self._next_uid = next_uid
        for batch in self._shard_batches(emitted_batches):
            yield batch

    def __len__(self) -> int:
        sim_rng = random.Random()
        if self.persistent_state:
            sim_rng.setstate(self._rng.getstate())
            order = self._order.copy()
            cursor = self._cursor
            pending_pairs = self._pending.copy()
            next_uid = self._next_uid
        else:
            sim_rng.seed(self.seed)
            order = []
            cursor = 0
            pending_pairs = []
            next_uid = 0
        batches, _, _, _, _ = self._run_epoch(
            order=order,
            cursor=cursor,
            pending_pairs=pending_pairs,
            next_uid=next_uid,
            rng=sim_rng,
            collect_batches=False,
        )
        return len(self._shard_batches(batches))


def _dataset_total_lengths(dataset: Dataset) -> list[int]:
    if isinstance(dataset, SequenceTreeDataset):
        return list(dataset.total_lengths)
    if isinstance(dataset, Subset):
        base_lengths = _dataset_total_lengths(dataset.dataset)
        return [int(base_lengths[idx]) for idx in dataset.indices]
    raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")


class SequenceTreeCollator:
    """Pack one or more sequences into a fixed-length batch row.

    The collator samples dynamic training subtrees from the stored sequence-tree
    data and pads all anchor-local trees in the row to the same `tree_size`.
    """

    def __init__(
        self,
        *,
        pack_length: int,
        num_anchors: int,
        training_tree_size: int,
        mask_token_id: int,
        pad_token_id: int,
        seed: int,
        sample_anchors: bool,
    ) -> None:
        self.pack_length = pack_length
        self.num_anchors = num_anchors
        self.training_tree_size = training_tree_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.sample_anchors = sample_anchors
        self._rng = random.Random(seed)

    def _empty_batch(self) -> PackedBatch:
        empty_shape = (1, self.pack_length)
        empty_tree = (1, 0, 0)
        return PackedBatch(
            input_ids=torch.full(empty_shape, self.pad_token_id, dtype=torch.long),
            position_ids=torch.zeros(empty_shape, dtype=torch.long),
            document_mask=torch.zeros(empty_shape, dtype=torch.long),
            context_valid_mask=torch.zeros(empty_shape, dtype=torch.bool),
            anchor_positions=torch.zeros((1, 0), dtype=torch.long),
            anchor_document_ids=torch.zeros((1, 0), dtype=torch.long),
            anchor_valid_mask=torch.zeros((1, 0), dtype=torch.bool),
            tree_labels=torch.full(empty_tree, IGNORE_IDX, dtype=torch.long),
            tree_noise_ids=torch.full(empty_tree, self.mask_token_id, dtype=torch.long),
            tree_position_ids=torch.zeros(empty_tree, dtype=torch.long),
            tree_cum_probs=torch.zeros(empty_tree, dtype=torch.float32),
            tree_valid_mask=torch.zeros(empty_tree, dtype=torch.bool),
            tree_parent_indices=torch.full(empty_tree, -1, dtype=torch.long),
            tree_depths=torch.zeros(empty_tree, dtype=torch.long),
            tree_node_ranks=torch.zeros(empty_tree, dtype=torch.long),
            tree_primary_path_mask=torch.zeros(empty_tree, dtype=torch.bool),
        )

    def _pack_rows(self, samples: list[dict[str, Any]]) -> list[list[tuple[dict[str, Any], int, int]]]:
        items = [(sample, int(sample["total_len"])) for sample in samples]
        return _pack_items_into_rows(items, pack_length=self.pack_length)

    def _valid_anchor_positions(self, sample: dict[str, Any]) -> list[int]:
        response_start = int(sample["response_start_position"])
        total_len = int(sample["main_path_ids"].numel())
        if total_len <= response_start + 1:
            return []
        return list(range(response_start, total_len - 1))

    def _pad_1d(self, rows: list[torch.Tensor], fill_value: int, dtype: torch.dtype) -> torch.Tensor:
        max_len = max((row.numel() for row in rows), default=0)
        if max_len == 0:
            return torch.zeros((len(rows), 0), dtype=dtype)
        padded = []
        for row in rows:
            cur = torch.full((max_len,), fill_value, dtype=dtype)
            cur[: row.numel()] = row.to(dtype=dtype)
            padded.append(cur)
        return torch.stack(padded, dim=0)

    def _stack_row_blocks(
        self,
        blocks: list[dict[str, torch.Tensor]],
        *,
        key: str,
        fill_value: int | float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not blocks:
            return torch.zeros((0, 0), dtype=dtype)
        max_width = max(int(block[key].numel()) for block in blocks)
        stacked = torch.full((len(blocks), max_width), fill_value, dtype=dtype)
        for row_idx, block in enumerate(blocks):
            values = block[key].to(dtype=dtype)
            stacked[row_idx, : values.numel()] = values
        return stacked

    def _pad_tree_rows(self, rows: list[torch.Tensor], fill_value: int | float, dtype: torch.dtype) -> torch.Tensor:
        max_rows = max((row.shape[0] for row in rows), default=0)
        max_cols = max((row.shape[1] for row in rows), default=0)
        if max_rows == 0 or max_cols == 0:
            return torch.zeros((len(rows), max_rows, max_cols), dtype=dtype)
        padded = []
        for row in rows:
            cur = torch.full((max_rows, max_cols), fill_value, dtype=dtype)
            if row.numel() > 0:
                cur[: row.shape[0], : row.shape[1]] = row.to(dtype=dtype)
            padded.append(cur)
        return torch.stack(padded, dim=0)

    def __call__(self, samples: list[dict[str, Any]]) -> PackedBatch:
        samples = [sample for sample in samples if int(sample["total_len"]) <= self.pack_length]
        if not samples:
            return self._empty_batch()

        rows = self._pack_rows(samples)
        input_rows = []
        position_rows = []
        document_rows = []
        context_rows = []
        anchor_position_rows = []
        anchor_document_rows = []
        anchor_valid_rows = []
        tree_labels_rows = []
        tree_noise_rows = []
        tree_position_rows = []
        tree_cum_prob_rows = []
        tree_valid_rows = []
        tree_parent_rows = []
        tree_depth_rows = []
        tree_rank_rows = []
        tree_primary_rows = []

        for docs in rows:
            input_ids = torch.full((self.pack_length,), self.pad_token_id, dtype=torch.long)
            position_ids = torch.zeros((self.pack_length,), dtype=torch.long)
            document_mask = torch.zeros((self.pack_length,), dtype=torch.long)
            context_valid = torch.zeros((self.pack_length,), dtype=torch.bool)

            row_anchor_positions: list[int] = []
            row_anchor_document_ids: list[int] = []
            row_blocks: list[dict[str, torch.Tensor]] = []
            row_anchor_candidates: list[tuple[int, int, dict[str, Any], int]] = []

            for sample, row_start, doc_id in docs:
                full_ids = sample["main_path_ids"]
                total_len = int(full_ids.numel())
                row_end = row_start + total_len
                input_ids[row_start:row_end] = full_ids
                position_ids[row_start:row_end] = torch.arange(total_len, dtype=torch.long)
                document_mask[row_start:row_end] = doc_id
                context_valid[row_start:row_end] = True
                for anchor_local in self._valid_anchor_positions(sample):
                    row_anchor_candidates.append((row_start + anchor_local, doc_id, sample, anchor_local))

            row_anchor_candidates.sort(key=lambda item: item[0])
            if len(row_anchor_candidates) > self.num_anchors:
                if self.sample_anchors:
                    keep = sorted(self._rng.sample(range(len(row_anchor_candidates)), self.num_anchors))
                    selected_candidates = [row_anchor_candidates[idx] for idx in keep]
                else:
                    selected_candidates = row_anchor_candidates[: self.num_anchors]
            else:
                selected_candidates = row_anchor_candidates

            for anchor_position, doc_id, sample, anchor_local in selected_candidates:
                row_anchor_positions.append(anchor_position)
                row_anchor_document_ids.append(doc_id)
                row_blocks.append(
                    sample_training_subtree(
                        sample=sample,
                        anchor_local_position=anchor_local,
                        training_tree_size=self.training_tree_size,
                        mask_token_id=self.mask_token_id,
                        rng=self._rng,
                        deterministic=not self.sample_anchors,
                    )
                )

            input_rows.append(input_ids)
            position_rows.append(position_ids)
            document_rows.append(document_mask)
            context_rows.append(context_valid)
            anchor_position_rows.append(torch.tensor(row_anchor_positions, dtype=torch.long))
            anchor_document_rows.append(torch.tensor(row_anchor_document_ids, dtype=torch.long))
            anchor_valid_rows.append(torch.ones((len(row_anchor_positions),), dtype=torch.bool))
            tree_labels_rows.append(self._stack_row_blocks(row_blocks, key="tree_labels", fill_value=IGNORE_IDX, dtype=torch.long))
            tree_noise_rows.append(self._stack_row_blocks(row_blocks, key="tree_noise_ids", fill_value=self.mask_token_id, dtype=torch.long))
            tree_position_rows.append(self._stack_row_blocks(row_blocks, key="tree_position_ids", fill_value=0, dtype=torch.long))
            tree_cum_prob_rows.append(self._stack_row_blocks(row_blocks, key="tree_cum_probs", fill_value=0.0, dtype=torch.float32))
            tree_valid_rows.append(self._stack_row_blocks(row_blocks, key="tree_valid_mask", fill_value=0, dtype=torch.bool))
            tree_parent_rows.append(self._stack_row_blocks(row_blocks, key="tree_parent_indices", fill_value=-1, dtype=torch.long))
            tree_depth_rows.append(self._stack_row_blocks(row_blocks, key="tree_depths", fill_value=0, dtype=torch.long))
            tree_rank_rows.append(self._stack_row_blocks(row_blocks, key="tree_node_ranks", fill_value=0, dtype=torch.long))
            tree_primary_rows.append(self._stack_row_blocks(row_blocks, key="tree_primary_path_mask", fill_value=0, dtype=torch.bool))

        return PackedBatch(
            input_ids=torch.stack(input_rows, dim=0),
            position_ids=torch.stack(position_rows, dim=0),
            document_mask=torch.stack(document_rows, dim=0),
            context_valid_mask=torch.stack(context_rows, dim=0),
            anchor_positions=self._pad_1d(anchor_position_rows, 0, torch.long),
            anchor_document_ids=self._pad_1d(anchor_document_rows, 0, torch.long),
            anchor_valid_mask=self._pad_1d(anchor_valid_rows, 0, torch.bool),
            tree_labels=self._pad_tree_rows(tree_labels_rows, IGNORE_IDX, torch.long),
            tree_noise_ids=self._pad_tree_rows(tree_noise_rows, self.mask_token_id, torch.long),
            tree_position_ids=self._pad_tree_rows(tree_position_rows, 0, torch.long),
            tree_cum_probs=self._pad_tree_rows(tree_cum_prob_rows, 0.0, torch.float32),
            tree_valid_mask=self._pad_tree_rows(tree_valid_rows, 0, torch.bool),
            tree_parent_indices=self._pad_tree_rows(tree_parent_rows, -1, torch.long),
            tree_depths=self._pad_tree_rows(tree_depth_rows, 0, torch.long),
            tree_node_ranks=self._pad_tree_rows(tree_rank_rows, 0, torch.long),
            tree_primary_path_mask=self._pad_tree_rows(tree_primary_rows, 0, torch.bool),
        )


def _maybe_limit_dataset(dataset: Dataset, max_items: int | None) -> Dataset:
    if max_items is None or len(dataset) <= max_items:
        return dataset
    return Subset(dataset, list(range(max_items)))


def build_dataloaders(
    *,
    config: DataConfig,
    mask_token_id: int,
    pad_token_id: int,
    num_replicas: int = 1,
    rank: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build the only supported train/eval dataloaders for the cleaned-up path."""
    dataset = SequenceTreeDataset(config.path)
    if dataset.format_version != "stage2_v2":
        raise ValueError("The cleaned-up loader only supports Stage 2 v2 data.")

    if config.eval_path is not None:
        train_dataset: Dataset = dataset
        eval_dataset = SequenceTreeDataset(config.eval_path)
    else:
        train_size = int(len(dataset) * config.train_split)
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = random_split(
            dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(config.seed),
        )

    train_dataset = _maybe_limit_dataset(train_dataset, config.max_train_sequences)
    eval_dataset = _maybe_limit_dataset(eval_dataset, config.max_eval_sequences)

    train_collator = SequenceTreeCollator(
        pack_length=config.pack_length,
        num_anchors=config.num_anchors,
        training_tree_size=config.training_tree_size,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        seed=config.seed,
        sample_anchors=True,
    )
    eval_collator = SequenceTreeCollator(
        pack_length=config.pack_length,
        num_anchors=config.num_anchors,
        training_tree_size=config.training_tree_size,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        seed=config.seed,
        sample_anchors=False,
    )

    train_sampler = PackedBatchSampler(
        sample_lengths=_dataset_total_lengths(train_dataset),
        pack_length=config.pack_length,
        packed_batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
        seed=config.seed,
        persistent_state=True,
        num_replicas=num_replicas,
        rank=rank,
    )
    eval_sampler = PackedBatchSampler(
        sample_lengths=_dataset_total_lengths(eval_dataset),
        pack_length=config.pack_length,
        packed_batch_size=config.batch_size,
        shuffle=False,
        drop_last=config.drop_last,
        seed=config.seed,
        persistent_state=False,
        num_replicas=num_replicas,
        rank=rank,
    )

    loader_kwargs = {"num_workers": config.num_workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=train_collator, **loader_kwargs)
    eval_loader = DataLoader(eval_dataset, batch_sampler=eval_sampler, collate_fn=eval_collator, **loader_kwargs)
    return train_loader, eval_loader
