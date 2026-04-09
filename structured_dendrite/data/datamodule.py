from __future__ import annotations

import csv
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Any

import lightning as L
import torch
from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset as TorchDataset

from structured_dendrite.data.tokenization import build_tokenizer


def cfg_value(cfg, key: str, default):
    value = OmegaConf.select(cfg, key, default=default)
    return default if value is None else value


def _normalize_delimiter(delimiter: Any) -> str:
    if delimiter is None:
        return ","
    if not isinstance(delimiter, str):
        delimiter = str(delimiter)

    normalized = delimiter.strip().strip("\"'")
    aliases = {
        r"\t": "\t",
        "`t": "\t",
        "tab": "\t",
        "TAB": "\t",
        r"\n": "\n",
        "`n": "\n",
        r"\r": "\r",
        "`r": "\r",
    }
    if normalized in aliases:
        return aliases[normalized]
    if len(normalized) == 1:
        return normalized
    if "\t" in delimiter or "\\t" in normalized or "`t" in normalized:
        return "\t"
    raise ValueError(f"Unsupported delimiter specification: {delimiter!r}")


@dataclass
class DatasetInfo:
    task_name: str
    input_kind: str
    num_classes: int | None = None
    vocab_size: int | None = None
    pad_token_id: int | None = None
    sequence_length: int | None = None
    image_channels: int | None = None
    pair_inputs: bool = False


class ListDataset(TorchDataset):
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class FlexibleSequenceDataModule(L.LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = None
        self.info = DatasetInfo(
            task_name=cfg.task_name,
            input_kind=cfg.input_kind,
            sequence_length=max(int(cfg.max_length), int(cfg_value(cfg, "eval_max_length", cfg.max_length))),
            image_channels=cfg.image.channels if cfg.input_kind == "image_sequence" else None,
            pair_inputs=cfg.input_kind == "pair_text",
        )

        self.train_dataset: TorchDataset | None = None
        self.val_dataset: TorchDataset | None = None
        self.test_dataset: TorchDataset | None = None

    def prepare_data(self) -> None:
        self._load_raw_dataset()

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return

        raw_dataset = self._load_raw_dataset()
        input_kind = self.cfg.input_kind
        if input_kind in {"text", "pair_text", "language_model"}:
            self._setup_tokenized_datasets(raw_dataset)
        elif input_kind == "image_sequence":
            self._setup_image_datasets(raw_dataset)
        else:
            raise ValueError(f"Unsupported input kind: {input_kind}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.loader.batch_size,
            shuffle=True,
            num_workers=self.cfg.loader.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.loader.eval_batch_size or self.cfg.loader.batch_size,
            shuffle=False,
            num_workers=self.cfg.loader.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.loader.eval_batch_size or self.cfg.loader.batch_size,
            shuffle=False,
            num_workers=self.cfg.loader.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def _load_raw_dataset(self):
        source_cfg = OmegaConf.to_container(self.cfg.source, resolve=True)
        source_cfg = {key: value for key, value in source_cfg.items() if value is not None}
        path = source_cfg.pop("path")

        if path == "csv" and "data_files" in source_cfg:
            delimiter = _normalize_delimiter(source_cfg.get("sep", ","))
            return self._load_local_tabular_dataset(source_cfg["data_files"], delimiter)
        if path == "text" and "data_files" in source_cfg:
            return self._load_local_text_dataset(source_cfg["data_files"])
        return load_dataset(path, **source_cfg)

    def _load_local_tabular_dataset(self, data_files: dict[str, str], delimiter: str) -> DatasetDict:
        splits = {}
        for split_name, file_path in data_files.items():
            with Path(file_path).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                splits[split_name] = Dataset.from_list(list(reader))
        return DatasetDict(splits)

    def _load_local_text_dataset(self, data_files: dict[str, str]) -> DatasetDict:
        splits = {}
        for split_name, file_path in data_files.items():
            with Path(file_path).open("r", encoding="utf-8") as handle:
                rows = [line.rstrip("\n") for line in handle if line.strip()]
            splits[split_name] = Dataset.from_dict({"text": rows})
        return DatasetDict(splits)

    def _setup_tokenized_datasets(self, raw_dataset) -> None:
        train_split = self._maybe_limit_split(raw_dataset[self.cfg.splits.train], stage_name="train")
        val_split = self._maybe_limit_split(raw_dataset[self.cfg.splits.validation], stage_name="validation")
        test_split = self._maybe_limit_split(raw_dataset[self.cfg.splits.test], stage_name="test")

        if self.cfg.input_kind == "pair_text":
            texts = list(train_split[self.cfg.text.primary_field]) + list(train_split[self.cfg.text.secondary_field])
        else:
            texts = list(train_split[self.cfg.text.primary_field])

        self.tokenizer = build_tokenizer(self.cfg.tokenizer, texts)
        self.info.vocab_size = self.tokenizer.state.vocab_size
        self.info.pad_token_id = self.tokenizer.state.pad_token_id

        if self.cfg.input_kind == "language_model":
            self.train_dataset = self._build_lm_dataset(train_split, split_name="train")
            self.val_dataset = self._build_lm_dataset(val_split, split_name="validation")
            self.test_dataset = self._build_lm_dataset(test_split, split_name="test")
            return

        self.train_dataset = ListDataset(self._encode_classification_split(train_split, split_name="train"))
        self.val_dataset = ListDataset(self._encode_classification_split(val_split, split_name="validation"))
        self.test_dataset = ListDataset(self._encode_classification_split(test_split, split_name="test"))

        if self.info.num_classes is None:
            labels = [example["label"] for example in self.train_dataset.examples]
            self.info.num_classes = len(sorted(set(labels)))

    def _setup_image_datasets(self, raw_dataset) -> None:
        self.train_dataset = self._maybe_limit_split(raw_dataset[self.cfg.splits.train], stage_name="train")
        self.val_dataset = self._maybe_limit_split(raw_dataset[self.cfg.splits.validation], stage_name="validation")
        self.test_dataset = self._maybe_limit_split(raw_dataset[self.cfg.splits.test], stage_name="test")
        self.info.num_classes = self.cfg.num_classes

    def _maybe_limit_split(self, split, stage_name: str):
        fraction_key = "train_fraction" if stage_name == "train" else "eval_fraction"
        max_examples_key = "max_train_examples" if stage_name == "train" else "max_eval_examples"
        fraction = float(cfg_value(self.cfg, fraction_key, 1.0))
        max_examples = cfg_value(self.cfg, max_examples_key, None)
        seed = int(cfg_value(self.cfg, "split_seed", 1111))

        target_size = len(split)
        if fraction < 1.0:
            target_size = min(target_size, max(1, floor(len(split) * fraction)))
        if max_examples is not None:
            target_size = min(target_size, int(max_examples))
        if target_size >= len(split):
            return split
        return split.shuffle(seed=seed).select(range(target_size))

    def _split_max_length(self, split_name: str) -> int:
        if split_name == "train":
            return int(cfg_value(self.cfg, "train_max_length", self.cfg.max_length))
        return int(cfg_value(self.cfg, "eval_max_length", self.cfg.max_length))

    def _split_lm_stride(self, split_name: str, block_size: int) -> int:
        if split_name == "train":
            return int(cfg_value(self.cfg, "train_lm_stride", cfg_value(self.cfg, "lm_stride", block_size) or block_size))
        return int(cfg_value(self.cfg, "eval_lm_stride", cfg_value(self.cfg, "lm_stride", block_size) or block_size))

    def _encode_classification_split(self, split, split_name: str) -> list[dict[str, Any]]:
        max_length = self._split_max_length(split_name)
        examples: list[dict[str, Any]] = []
        for row in split:
            label = int(row[self.cfg.label_field])
            if self.cfg.input_kind == "text":
                examples.append(
                    {
                        "input_ids": self.tokenizer.encode(row[self.cfg.text.primary_field], max_length=max_length),
                        "label": label,
                    }
                )
            else:
                examples.append(
                    {
                        "input_ids_a": self.tokenizer.encode(row[self.cfg.text.primary_field], max_length=max_length),
                        "input_ids_b": self.tokenizer.encode(row[self.cfg.text.secondary_field], max_length=max_length),
                        "label": label,
                    }
                )
        return examples

    def _build_lm_dataset(self, split, split_name: str) -> ListDataset:
        tokens: list[int] = []
        for row in split:
            tokens.extend(self.tokenizer.encode(row[self.cfg.text.primary_field], max_length=None))

        block_size = self._split_max_length(split_name)
        stride = self._split_lm_stride(split_name, block_size)
        examples: list[dict[str, Any]] = []
        upper = max(len(tokens) - block_size - 1, 0)
        for start in range(0, upper + 1, stride):
            chunk = tokens[start : start + block_size + 1]
            if len(chunk) < block_size + 1:
                continue
            examples.append({"input_ids": chunk[:-1], "labels": chunk[1:]})

        return ListDataset(examples)

    def _collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if self.cfg.input_kind == "text":
            return self._collate_text(batch)
        if self.cfg.input_kind == "pair_text":
            return self._collate_pair_text(batch)
        if self.cfg.input_kind == "language_model":
            return self._collate_lm(batch)
        if self.cfg.input_kind == "image_sequence":
            return self._collate_images(batch)
        raise ValueError(f"Unsupported input kind: {self.cfg.input_kind}")

    def _collate_text(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids, attention_mask = self._pad_sequences([item["input_ids"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return {"inputs": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _collate_pair_text(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids_a, mask_a = self._pad_sequences([item["input_ids_a"] for item in batch])
        input_ids_b, mask_b = self._pad_sequences([item["input_ids_b"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return {
            "inputs_a": input_ids_a,
            "mask_a": mask_a,
            "inputs_b": input_ids_b,
            "mask_b": mask_b,
            "labels": labels,
        }

    def _collate_lm(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        inputs = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {"inputs": inputs, "labels": labels}

    def _collate_images(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        image_field = self.cfg.image.field
        label_field = self.cfg.label_field
        images = [self._image_to_sequence(example[image_field]) for example in batch]
        labels = torch.tensor([int(example[label_field]) for example in batch], dtype=torch.long)
        stacked = torch.stack(images, dim=0)
        mask = torch.ones(stacked.shape[:2], dtype=torch.bool)
        return {"inputs": stacked, "attention_mask": mask, "labels": labels}

    def _image_to_sequence(self, image) -> torch.Tensor:
        if hasattr(image, "convert"):
            image = image.convert("RGB" if self.cfg.image.channels == 3 else "L")
        tensor = image if torch.is_tensor(image) else torch.tensor(image)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[0] not in {1, 3}:
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor.float()
        if tensor.max() > 1:
            tensor = tensor / 255.0
        return tensor.permute(1, 2, 0).reshape(-1, tensor.shape[0])

    def _pad_sequences(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_length = max(len(sequence) for sequence in sequences)
        pad_id = int(self.info.pad_token_id or 0)
        values = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
        for row_index, sequence in enumerate(sequences):
            length = len(sequence)
            values[row_index, :length] = torch.tensor(sequence, dtype=torch.long)
            mask[row_index, :length] = True
        return values, mask


