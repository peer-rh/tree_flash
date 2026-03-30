from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Protocol

import h5py
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, Subset, random_split
from datasets import load_dataset, Features, Sequence, Value

from data_pipeline.stage2 import IGNORE_IDX

from .trees import TreeInfo


class TreeProcessorProtocol(Protocol):
    tree_seq_depth: int
    block_size: int
    sub_tree_paths: tuple[str, ...]

    def build_anchor_tensors(
        self,
        *,
        response_subtrees: torch.Tensor,
        response_probs: torch.Tensor,
        anchor_local_positions: list[int],
        anchor_positions: list[int],
        mask_token_id: int,
    ) -> dict[str, torch.Tensor]:
        ...

    def build_tree_info(
        self,
        batch_size: int,
        num_blocks: int,
        device: torch.device,
    ) -> TreeInfo:
        ...


@dataclass
class PackedBatch:
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

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def num_anchors(self) -> int:
        return int(self.anchor_positions.shape[1])

    @property
    def block_size(self) -> int:
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
        )


@dataclass
class DataModuleConfig:
    path: str
    eval_path: str | None = None
    train_split: float = 0.95
    batch_size: int = 8
    pack_length: int = 3072
    num_anchors: int = 8
    tree_seq_depth: int = 4
    num_workers: int = 0
    seed: int = 42
    shuffle: bool = True
    drop_last: bool = False
    max_train_sequences: int | None = None
    max_eval_sequences: int | None = None


class Stage2Dataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with h5py.File(self.path, "r") as hf:
            self.offsets = hf["sequence_offsets"][:]
            prompt_lengths = []
            for prompt_ids in hf["prompt_ids"]:
                prompt_lengths.append(len(prompt_ids))
            raw_paths = hf.attrs.get("sub_tree_paths", [])
            self.sub_tree_paths = tuple(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in raw_paths
            )
        response_lengths = self.offsets[1:] - self.offsets[:-1]
        self.total_lengths = tuple(
            int(prompt_len + response_len)
            for prompt_len, response_len in zip(prompt_lengths, response_lengths, strict=True)
        )
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.offsets) - 1

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
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        prompt_ids = torch.tensor(hf["prompt_ids"][idx], dtype=torch.long)
        response_ids = torch.tensor(hf["response_ids"][idx], dtype=torch.long)
        sub_trees = torch.tensor(hf["sub_trees"][start:end], dtype=torch.long)
        sub_tree_ar_probs = torch.tensor(hf["sub_trees_ar_probs"][start:end], dtype=torch.float32)
        return {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "sub_trees": sub_trees,
            "sub_tree_ar_probs": sub_tree_ar_probs,
            "total_len": self.total_lengths[idx],
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
            rows.append(
                {
                    "used": total_len,
                    "docs": [(item, 0, 1)],
                }
            )
    return [row["docs"] for row in rows]


@dataclass(frozen=True)
class _PendingIndex:
    uid: int
    index: int
    current_epoch: bool


class FixedPackedBatchSampler(BatchSampler):
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
    ) -> None:
        if packed_batch_size <= 0:
            raise ValueError(f"packed_batch_size must be positive, got {packed_batch_size}.")
        self.pack_length = pack_length
        self.packed_batch_size = packed_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.persistent_state = persistent_state
        self._rng = random.Random(seed)
        self._next_uid = 0
        self._pending: list[tuple[int, int]] = []
        self._eligible = [
            (idx, total_len)
            for idx, total_len in enumerate(sample_lengths)
            if int(total_len) <= pack_length
        ]
        if not self._eligible:
            raise ValueError(
                f"No dataset samples fit within pack_length={pack_length}."
            )
        self._eligible_indices = [idx for idx, _ in self._eligible]
        self._length_by_index = {idx: int(total_len) for idx, total_len in self._eligible}
        self._order: list[int] = []
        self._cursor = 0

    def _reset_order(self) -> None:
        self._order = self._eligible_indices.copy()
        if self.shuffle:
            self._rng.shuffle(self._order)
        self._cursor = 0

    def _next_stream_index(
        self,
        order: list[int],
        cursor: int,
        rng: random.Random,
    ) -> tuple[int, list[int], int]:
        if not order or cursor >= len(order):
            order = self._eligible_indices.copy()
            if self.shuffle:
                rng.shuffle(order)
            cursor = 0
        index = order[cursor]
        return index, order, cursor + 1

    def _pack_pending(
        self,
        pending: list[_PendingIndex],
    ) -> list[list[tuple[_PendingIndex, int, int]]]:
        items = [
            (entry, self._length_by_index[entry.index])
            for entry in pending
        ]
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
        for batch in emitted_batches:
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
        return len(batches)


def _dataset_total_lengths(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Stage2Dataset):
        return list(dataset.total_lengths)
    if isinstance(dataset, Subset):
        base_lengths = _dataset_total_lengths(dataset.dataset)
        return [int(base_lengths[idx]) for idx in dataset.indices]
    raise TypeError(f"Unsupported dataset type for fixed packed batching: {type(dataset)!r}")


class PackedBatchCollator:
    def __init__(
        self,
        *,
        tree_processor: TreeProcessorProtocol,
        pack_length: int,
        num_anchors: int,
        mask_token_id: int,
        pad_token_id: int,
        seed: int,
        sample_anchors: bool,
    ) -> None:
        self.tree_processor = tree_processor
        self.pack_length = pack_length
        self.num_anchors = num_anchors
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.sample_anchors = sample_anchors
        self._rng = random.Random(seed)

    def _empty_batch(self) -> PackedBatch:
        empty_shape = (1, self.pack_length)
        empty_tree_shape = (1, 0, self.tree_processor.block_size)
        return PackedBatch(
            input_ids=torch.full(empty_shape, self.pad_token_id, dtype=torch.long),
            position_ids=torch.zeros(empty_shape, dtype=torch.long),
            document_mask=torch.zeros(empty_shape, dtype=torch.long),
            context_valid_mask=torch.zeros(empty_shape, dtype=torch.bool),
            anchor_positions=torch.zeros((1, 0), dtype=torch.long),
            anchor_document_ids=torch.zeros((1, 0), dtype=torch.long),
            anchor_valid_mask=torch.zeros((1, 0), dtype=torch.bool),
            tree_labels=torch.full(empty_tree_shape, IGNORE_IDX, dtype=torch.long),
            tree_noise_ids=torch.full(empty_tree_shape, self.mask_token_id, dtype=torch.long),
            tree_position_ids=torch.zeros(empty_tree_shape, dtype=torch.long),
            tree_cum_probs=torch.zeros(empty_tree_shape, dtype=torch.float32),
            tree_valid_mask=torch.zeros(empty_tree_shape, dtype=torch.bool),
        )

    def _pack_rows(self, samples: list[dict[str, Any]]) -> list[list[tuple[dict[str, Any], int, int]]]:
        items = [(sample, int(sample["total_len"])) for sample in samples]
        return _pack_items_into_rows(items, pack_length=self.pack_length)

    def _valid_anchor_locals(self, sample: dict[str, Any]) -> list[int]:
        response_ids = sample["response_ids"]
        response_len = int(response_ids.numel())
        if response_len < self.tree_processor.tree_seq_depth:
            return []
        valid = []
        max_start = response_len - self.tree_processor.tree_seq_depth + 1
        for anchor_local in range(max_start):
            root_slice = sample["sub_trees"][anchor_local : anchor_local + self.tree_processor.tree_seq_depth, 0]
            if torch.all(root_slice != IGNORE_IDX):
                valid.append(anchor_local)
        return valid

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

    def _pad_2d(self, rows: list[torch.Tensor], fill_value: int | float, dtype: torch.dtype) -> torch.Tensor:
        max_rows = max((row.shape[0] for row in rows), default=0)
        block_size = self.tree_processor.block_size
        if max_rows == 0:
            return torch.zeros((len(rows), 0, block_size), dtype=dtype)
        padded = []
        for row in rows:
            cur = torch.full((max_rows, block_size), fill_value, dtype=dtype)
            if row.shape[0] > 0:
                cur[: row.shape[0]] = row.to(dtype=dtype)
            padded.append(cur)
        return torch.stack(padded, dim=0)

    def __call__(self, samples: list[dict[str, Any]]) -> PackedBatch:
        samples = [sample for sample in samples if int(sample["total_len"]) <= self.pack_length]
        if not samples:
            return self._empty_batch()

        rows = self._pack_rows(samples)
        input_ids_rows = []
        position_ids_rows = []
        document_mask_rows = []
        context_valid_rows = []
        anchor_positions_rows = []
        anchor_document_rows = []
        anchor_valid_rows = []
        tree_labels_rows = []
        tree_noise_rows = []
        tree_position_rows = []
        tree_cum_prob_rows = []
        tree_valid_rows = []

        for docs in rows:
            input_ids = torch.full((self.pack_length,), self.pad_token_id, dtype=torch.long)
            position_ids = torch.zeros((self.pack_length,), dtype=torch.long)
            document_mask = torch.zeros((self.pack_length,), dtype=torch.long)
            context_valid = torch.zeros((self.pack_length,), dtype=torch.bool)

            row_anchor_positions: list[int] = []
            row_anchor_document_ids: list[int] = []
            row_tree_labels: list[torch.Tensor] = []
            row_tree_noise: list[torch.Tensor] = []
            row_tree_positions: list[torch.Tensor] = []
            row_tree_cum_probs: list[torch.Tensor] = []
            row_tree_valid: list[torch.Tensor] = []
            row_anchor_candidates: list[tuple[int, int, dict[str, Any], int, int]] = []

            for sample, row_start, doc_id in docs:
                prompt_ids = sample["prompt_ids"]
                response_ids = sample["response_ids"]
                full_ids = torch.cat([prompt_ids, response_ids], dim=0)
                total_len = int(full_ids.numel())
                prompt_len = int(prompt_ids.numel())

                row_end = row_start + total_len
                input_ids[row_start:row_end] = full_ids
                position_ids[row_start:row_end] = torch.arange(total_len, dtype=torch.long)
                document_mask[row_start:row_end] = doc_id
                context_valid[row_start:row_end] = True

                for anchor_local in self._valid_anchor_locals(sample):
                    row_anchor_candidates.append(
                        (
                            row_start + prompt_len + anchor_local,
                            doc_id,
                            sample,
                            row_start,
                            anchor_local,
                        )
                    )

            row_anchor_candidates.sort(key=lambda item: item[0])
            if len(row_anchor_candidates) > self.num_anchors:
                if self.sample_anchors:
                    selected_indices = sorted(self._rng.sample(range(len(row_anchor_candidates)), self.num_anchors))
                    selected_candidates = [row_anchor_candidates[idx] for idx in selected_indices]
                else:
                    selected_candidates = row_anchor_candidates[: self.num_anchors]
            else:
                selected_candidates = row_anchor_candidates

            for anchor_position, doc_id, sample, _, anchor_local in selected_candidates:
                tree_tensors = self.tree_processor.build_anchor_tensors(
                    response_subtrees=sample["sub_trees"],
                    response_probs=sample["sub_tree_ar_probs"],
                    anchor_local_positions=[anchor_local],
                    anchor_positions=[anchor_position],
                    mask_token_id=self.mask_token_id,
                )
                row_anchor_positions.append(anchor_position)
                row_anchor_document_ids.append(doc_id)
                row_tree_labels.append(tree_tensors["tree_labels"])
                row_tree_noise.append(tree_tensors["tree_noise_ids"])
                row_tree_positions.append(tree_tensors["tree_position_ids"])
                row_tree_cum_probs.append(tree_tensors["tree_cum_probs"])
                row_tree_valid.append(tree_tensors["tree_valid_mask"])

            if row_tree_labels:
                tree_labels = torch.cat(row_tree_labels, dim=0)
                tree_noise = torch.cat(row_tree_noise, dim=0)
                tree_positions = torch.cat(row_tree_positions, dim=0)
                tree_cum_probs = torch.cat(row_tree_cum_probs, dim=0)
                tree_valid = torch.cat(row_tree_valid, dim=0)
                anchor_valid = torch.ones((len(row_anchor_positions),), dtype=torch.bool)
            else:
                tree_labels = torch.empty((0, self.tree_processor.block_size), dtype=torch.long)
                tree_noise = torch.empty((0, self.tree_processor.block_size), dtype=torch.long)
                tree_positions = torch.empty((0, self.tree_processor.block_size), dtype=torch.long)
                tree_cum_probs = torch.empty((0, self.tree_processor.block_size), dtype=torch.float32)
                tree_valid = torch.empty((0, self.tree_processor.block_size), dtype=torch.bool)
                anchor_valid = torch.empty((0,), dtype=torch.bool)

            input_ids_rows.append(input_ids)
            position_ids_rows.append(position_ids)
            document_mask_rows.append(document_mask)
            context_valid_rows.append(context_valid)
            anchor_positions_rows.append(torch.tensor(row_anchor_positions, dtype=torch.long))
            anchor_document_rows.append(torch.tensor(row_anchor_document_ids, dtype=torch.long))
            anchor_valid_rows.append(anchor_valid)
            tree_labels_rows.append(tree_labels)
            tree_noise_rows.append(tree_noise)
            tree_position_rows.append(tree_positions)
            tree_cum_prob_rows.append(tree_cum_probs)
            tree_valid_rows.append(tree_valid)

        return PackedBatch(
            input_ids=torch.stack(input_ids_rows, dim=0),
            position_ids=torch.stack(position_ids_rows, dim=0),
            document_mask=torch.stack(document_mask_rows, dim=0),
            context_valid_mask=torch.stack(context_valid_rows, dim=0),
            anchor_positions=self._pad_1d(anchor_positions_rows, 0, torch.long),
            anchor_document_ids=self._pad_1d(anchor_document_rows, 0, torch.long),
            anchor_valid_mask=self._pad_1d(anchor_valid_rows, 0, torch.bool),
            tree_labels=self._pad_2d(tree_labels_rows, IGNORE_IDX, torch.long),
            tree_noise_ids=self._pad_2d(tree_noise_rows, self.mask_token_id, torch.long),
            tree_position_ids=self._pad_2d(tree_position_rows, 0, torch.long),
            tree_cum_probs=self._pad_2d(tree_cum_prob_rows, 0.0, torch.float32),
            tree_valid_mask=self._pad_2d(tree_valid_rows, 0, torch.bool),
        )


def _maybe_limit_dataset(dataset: Dataset, max_items: int | None) -> Dataset:
    if max_items is None or len(dataset) <= max_items:
        return dataset
    return Subset(dataset, list(range(max_items)))


def build_dataloaders(
    *,
    config: DataModuleConfig,
    tree_processor: TreeProcessorProtocol,
    mask_token_id: int,
    pad_token_id: int,
) -> tuple[DataLoader, DataLoader]:
    dataset = Stage2Dataset(config.path)
    if dataset.sub_tree_paths and tuple(dataset.sub_tree_paths) != tuple(tree_processor.sub_tree_paths):
        raise ValueError(
            "Stage 2 subtree paths do not match the configured tree processor: "
            f"{dataset.sub_tree_paths} != {tree_processor.sub_tree_paths}"
        )

    if config.eval_path is not None:
        train_dataset: Dataset = dataset
        eval_dataset = Stage2Dataset(config.eval_path)
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

    train_collator = PackedBatchCollator(
        tree_processor=tree_processor,
        pack_length=config.pack_length,
        num_anchors=config.num_anchors,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        seed=config.seed,
        sample_anchors=True,
    )
    eval_collator = PackedBatchCollator(
        tree_processor=tree_processor,
        pack_length=config.pack_length,
        num_anchors=config.num_anchors,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        seed=config.seed,
        sample_anchors=False,
    )

    train_lengths = _dataset_total_lengths(train_dataset)
    eval_lengths = _dataset_total_lengths(eval_dataset)
    train_batch_sampler = FixedPackedBatchSampler(
        sample_lengths=train_lengths,
        pack_length=config.pack_length,
        packed_batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
        seed=config.seed,
        persistent_state=True,
    )
    eval_batch_sampler = FixedPackedBatchSampler(
        sample_lengths=eval_lengths,
        pack_length=config.pack_length,
        packed_batch_size=config.batch_size,
        shuffle=False,
        drop_last=config.drop_last,
        seed=config.seed,
        persistent_state=False,
    )
    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=train_collator,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_sampler=eval_batch_sampler,
        collate_fn=eval_collator,
        **loader_kwargs,
    )
    return train_loader, eval_loader


def load_and_process_eval_dataset(data_name: str):
    # Math datasets
    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    # Chat datasets
    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(
            lambda x: {
                "formatted_input": (
                    f"{x['instruction']}\n\nInput:\n{x['input']}"
                    if x["input"]
                    else x["instruction"]
                )
            }
        )
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})

    elif data_name == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    # Coding datasets
    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "mbpp":
        dataset = load_dataset(
            "google-research-datasets/mbpp", "sanitized", split="test"
        )
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})

    elif data_name == "lbpp":
        LBPP_PY_TEST_URL = "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet"
        dataset = load_dataset("parquet", data_files={"test": LBPP_PY_TEST_URL})["test"]
        dataset = dataset.map(lambda x: {"turns": [x["instruction"]]})

    elif data_name == "swe-bench":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        prompt_fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "livecodebench":
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        allowed_files = [
            "test.jsonl",
            "test2.jsonl",
            "test3.jsonl",
            "test4.jsonl",
            "test5.jsonl",
            "test6.jsonl",
        ]
        urls = [base + fn for fn in allowed_files]
        dataset = load_dataset("json", data_files={"test": urls})["test"]

        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"

        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=target_features,
        )

    return dataset
