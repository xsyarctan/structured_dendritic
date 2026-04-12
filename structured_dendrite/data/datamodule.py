from __future__ import annotations

import csv
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Any

import lightning as L
import torch
from datasets import Dataset, DatasetDict, Image as HFImage, load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset as TorchDataset, Subset
from torchvision.transforms.functional import pil_to_tensor

from structured_dendrite.data.l5pc import L5PCDataset, TEST_SPLIT, TRAIN_SPLIT, prepare_l5pc_cache
from structured_dendrite.data.tokenization import build_tokenizer


PATHFINDER_BLACKLIST = {"pathfinder32/curv_baseline/imgs/0/sample_172.png"}


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


def _numeric_sort_key(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def _resolve_pathfinder_root(data_dir: str | Path, resolution: int | None) -> Path:
    root = Path(data_dir).expanduser()
    if resolution is not None:
        candidate = root / f"pathfinder{int(resolution)}"
        if candidate.is_dir():
            return candidate
        if root.name == f"pathfinder{int(resolution)}" and root.is_dir():
            return root
    if root.is_dir() and any((root / level).is_dir() for level in ["curv_baseline", "curv_contour_length_9", "curv_contour_length_14"]):
        return root
    raise FileNotFoundError(f"Could not find a Pathfinder directory under {root}")


@dataclass
class DatasetInfo:
    task_name: str
    input_kind: str
    num_classes: int | None = None
    vocab_size: int | None = None
    pad_token_id: int | None = None
    sequence_length: int | None = None
    image_channels: int | None = None
    input_channels: int | None = None
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
        input_kind = cfg.input_kind
        if input_kind == "l5pc":
            sequence_length = int(cfg.total_length)
            image_channels = None
            input_channels = int(cfg.input_channels)
        else:
            sequence_length = max(int(cfg.max_length), int(cfg_value(cfg, "eval_max_length", cfg.max_length)))
            image_channels = cfg.image.channels if input_kind == "image_sequence" else None
            input_channels = None
        self.info = DatasetInfo(
            task_name=cfg.task_name,
            input_kind=input_kind,
            sequence_length=sequence_length,
            image_channels=image_channels,
            input_channels=input_channels,
            pair_inputs=input_kind == "pair_text",
        )

        self.train_dataset: TorchDataset | None = None
        self.val_dataset: TorchDataset | None = None
        self.test_dataset: TorchDataset | None = None

    def prepare_data(self) -> None:
        if self.cfg.input_kind == "l5pc":
            prepare_l5pc_cache(self.cfg.source.root)
            return
        self._load_raw_dataset()

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return

        input_kind = self.cfg.input_kind
        if input_kind == "l5pc":
            self._setup_l5pc_datasets()
            return

        raw_dataset = self._load_raw_dataset()
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
            delimiter = _normalize_delimiter(source_cfg.pop("sep", ","))
            column_names = source_cfg.pop("column_names", None)
            skip_header = bool(source_cfg.pop("skip_header", False))
            return self._load_local_tabular_dataset(
                source_cfg["data_files"],
                delimiter,
                column_names=column_names,
                skip_header=skip_header,
            )
        if path == "text" and "data_files" in source_cfg:
            return self._load_local_text_dataset(source_cfg["data_files"])
        if path == "imdb":
            local_data_dir = source_cfg.pop("local_data_dir", None)
            if local_data_dir is not None:
                local_root = Path(local_data_dir).expanduser()
                acl_root = local_root / "aclImdb" if (local_root / "aclImdb").is_dir() else local_root
                if (acl_root / "train" / "pos").is_dir():
                    return self._load_local_acl_imdb_dataset(local_data_dir, RuntimeError("Using local IMDB fallback"))
            try:
                return load_dataset(path, **source_cfg)
            except Exception as error:
                if local_data_dir is None:
                    raise
                return self._load_local_acl_imdb_dataset(local_data_dir, error)
        if path == "pathfinder_metadata":
            return self._load_pathfinder_metadata_dataset(source_cfg)
        return load_dataset(path, **source_cfg)

    def _load_local_tabular_dataset(
        self,
        data_files: dict[str, str],
        delimiter: str,
        column_names: list[str] | tuple[str, ...] | None = None,
        skip_header: bool = False,
    ) -> DatasetDict:
        csv.field_size_limit(1_000_000_000)
        splits = {}
        normalized_column_names = list(column_names) if column_names is not None else None
        for split_name, file_path in data_files.items():
            with Path(file_path).open("r", encoding="utf-8", newline="") as handle:
                if normalized_column_names is None:
                    reader = csv.DictReader(handle, delimiter=delimiter)
                    rows = list(reader)
                else:
                    reader = csv.reader(handle, delimiter=delimiter)
                    if skip_header:
                        next(reader, None)
                    rows = []
                    for row in reader:
                        if not row:
                            continue
                        if len(row) != len(normalized_column_names):
                            raise ValueError(
                                f"Expected {len(normalized_column_names)} columns in {file_path}, found {len(row)}"
                            )
                        rows.append(dict(zip(normalized_column_names, row, strict=True)))
                splits[split_name] = Dataset.from_list(rows)
        return DatasetDict(splits)

    def _load_local_text_dataset(self, data_files: dict[str, str]) -> DatasetDict:
        splits = {}
        for split_name, file_path in data_files.items():
            with Path(file_path).open("r", encoding="utf-8") as handle:
                rows = [line.rstrip("\n") for line in handle if line.strip()]
            splits[split_name] = Dataset.from_dict({"text": rows})
        return DatasetDict(splits)

    def _load_local_acl_imdb_dataset(self, data_dir: str | Path, original_error: Exception) -> DatasetDict:
        root = Path(data_dir).expanduser()
        acl_root = root / "aclImdb" if (root / "aclImdb").is_dir() else root
        expected = acl_root / "train" / "pos"
        if not expected.is_dir():
            raise original_error

        splits = {}
        for split_name in ["train", "test"]:
            rows = []
            for label_name, label in [("neg", 0), ("pos", 1)]:
                for file_path in sorted((acl_root / split_name / label_name).glob("*.txt")):
                    text = file_path.read_text(encoding="utf-8").replace("<br />", " ").strip()
                    rows.append({"text": text, "label": label})
            splits[split_name] = Dataset.from_list(rows)
        return DatasetDict(splits)

    def _load_pathfinder_metadata_dataset(self, source_cfg: dict[str, Any]) -> DatasetDict:
        root = _resolve_pathfinder_root(source_cfg["data_dir"], source_cfg.get("resolution"))
        difficulty_levels = source_cfg.get("difficulty_levels") or ["curv_contour_length_14"]
        val_split = float(source_cfg.get("val_split", 0.1))
        test_split = float(source_cfg.get("test_split", 0.1))
        seed = int(source_cfg.get("seed", cfg_value(self.cfg, "split_seed", 1111)))

        if val_split < 0 or test_split < 0 or val_split + test_split >= 1.0:
            raise ValueError("Pathfinder val/test splits must be non-negative and sum to less than 1")

        samples: list[dict[str, Any]] = []
        for difficulty in difficulty_levels:
            metadata_dir = root / difficulty / "metadata"
            if not metadata_dir.is_dir():
                raise FileNotFoundError(f"Pathfinder metadata directory not found: {metadata_dir}")
            metadata_files = sorted(metadata_dir.glob("*.npy"), key=_numeric_sort_key)
            if not metadata_files:
                raise FileNotFoundError(f"No Pathfinder metadata files found in {metadata_dir}")
            for metadata_file in metadata_files:
                with metadata_file.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        parts = line.split()
                        if len(parts) < 4:
                            continue
                        image_rel = Path(difficulty) / parts[0] / parts[1]
                        blacklist_key = f"{root.name}/{image_rel.as_posix()}"
                        if blacklist_key in PATHFINDER_BLACKLIST:
                            continue
                        image_path = root / image_rel
                        samples.append({"image": str(image_path), "label": int(parts[3])})

        if not samples:
            raise FileNotFoundError(f"No Pathfinder samples found under {root}")

        dataset = Dataset.from_list(samples).cast_column("image", HFImage())
        total = len(dataset)
        val_size = int(total * val_split)
        test_size = int(total * test_split)
        train_size = total - val_size - test_size
        permutation = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()

        train_indices = permutation[:train_size]
        val_indices = permutation[train_size : train_size + val_size]
        test_indices = permutation[train_size + val_size :]
        return DatasetDict(
            {
                "train": dataset.select(train_indices),
                "validation": dataset.select(val_indices),
                "test": dataset.select(test_indices),
            }
        )

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

    def _setup_l5pc_datasets(self) -> None:
        root = Path(self.cfg.source.root)
        clip_voltage_above = float(self.cfg.voltage.clip_above)
        voltage_offset = float(self.cfg.voltage.offset)
        train_groups = cfg_value(self.cfg, "source.train_groups", None)
        validation_source = str(cfg_value(self.cfg, "source.validation_source", "test"))
        validation_groups = cfg_value(self.cfg, "source.validation_groups", None)
        test_groups = cfg_value(self.cfg, "source.test_groups", None)

        self.train_dataset = L5PCDataset(
            root=root,
            split_source=TRAIN_SPLIT,
            mode="train",
            crop_length=int(self.cfg.train_crop_length),
            train_repeats=int(self.cfg.train_repeats),
            select_groups=train_groups,
            clip_voltage_above=clip_voltage_above,
            voltage_offset=voltage_offset,
        )

        if validation_source == "train":
            self.val_dataset = L5PCDataset(
                root=root,
                split_source=TRAIN_SPLIT,
                mode="eval",
                crop_length=int(self.cfg.total_length),
                train_repeats=int(self.cfg.train_repeats),
                select_groups=validation_groups,
                clip_voltage_above=clip_voltage_above,
                voltage_offset=voltage_offset,
            )
        elif validation_source == "test":
            self.val_dataset = L5PCDataset(
                root=root,
                split_source=TEST_SPLIT,
                mode="eval",
                crop_length=int(self.cfg.total_length),
                train_repeats=int(self.cfg.train_repeats),
                select_groups=validation_groups,
                clip_voltage_above=clip_voltage_above,
                voltage_offset=voltage_offset,
            )
        else:
            raise ValueError(f"Unknown L5PC validation source: {validation_source}")

        self.test_dataset = L5PCDataset(
            root=root,
            split_source=TEST_SPLIT,
            mode="eval",
            crop_length=int(self.cfg.total_length),
            train_repeats=int(self.cfg.train_repeats),
            select_groups=test_groups,
            clip_voltage_above=clip_voltage_above,
            voltage_offset=voltage_offset,
        )

        self.train_dataset = self._maybe_limit_torch_dataset(self.train_dataset, stage_name="train")
        self.val_dataset = self._maybe_limit_torch_dataset(self.val_dataset, stage_name="validation")
        self.test_dataset = self._maybe_limit_torch_dataset(self.test_dataset, stage_name="test")
        self.info.input_channels = int(self.cfg.input_channels)

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

    def _maybe_limit_torch_dataset(self, dataset: TorchDataset, stage_name: str) -> TorchDataset:
        fraction_key = "train_fraction" if stage_name == "train" else "eval_fraction"
        max_examples_key = "max_train_examples" if stage_name == "train" else "max_eval_examples"
        fraction = float(cfg_value(self.cfg, fraction_key, 1.0))
        max_examples = cfg_value(self.cfg, max_examples_key, None)
        seed = int(cfg_value(self.cfg, "split_seed", 1111))

        target_size = len(dataset)
        if fraction < 1.0:
            target_size = min(target_size, max(1, floor(len(dataset) * fraction)))
        if max_examples is not None:
            target_size = min(target_size, int(max_examples))
        if target_size >= len(dataset):
            return dataset
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:target_size].tolist()
        return Subset(dataset, indices)

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
            label = int(float(row[self.cfg.label_field]))
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
        if self.cfg.input_kind == "l5pc":
            return self._collate_l5pc(batch)
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

    def _collate_l5pc(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return {
            "inputs": torch.stack([item["inputs"] for item in batch], dim=0),
            "spike_targets": torch.stack([item["spike_targets"] for item in batch], dim=0),
            "voltage_targets": torch.stack([item["voltage_targets"] for item in batch], dim=0),
            "raw_voltage_targets": torch.stack([item["raw_voltage_targets"] for item in batch], dim=0),
            "example_index": torch.stack([item["example_index"] for item in batch], dim=0),
            "crop_start": torch.stack([item["crop_start"] for item in batch], dim=0),
        }

    def _image_to_sequence(self, image) -> torch.Tensor:
        if hasattr(image, "convert"):
            image = image.convert("RGB" if self.cfg.image.channels == 3 else "L")
            tensor = pil_to_tensor(image)
        else:
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
