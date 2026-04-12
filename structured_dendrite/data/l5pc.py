from __future__ import annotations

import json
import math
import pickle
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy import sparse as sps
from torch.utils.data import Dataset


TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

TEST_GROUP_MAP = {
    "std": "Data_test",
    "sub": "Data_test_subthreshold",
    "ood": "Data_test_OOD",
    "erg": "Data_test_combined_ergodic",
}


def dict_to_sparse(indices, values, num_synapses: int, sim_duration_ms: int):
    columns: list[int] = []
    [columns.extend((index,) * len(event_times)) for index, event_times in zip(indices, values, strict=False)]

    rows: list[int] = []
    [rows.extend(event_times) for event_times in values]

    data = [1] * len(rows)
    return sps.coo_matrix(
        (data, (rows, columns)),
        shape=(sim_duration_ms, num_synapses),
        dtype=bool,
    ).tocsc()


def _load_single_experiment_file(sim_experiment_file: Path, output_root: Path, global_index: int) -> int:
    with sim_experiment_file.open("rb") as handle:
        experiment_dict = pickle.load(handle, encoding="latin1")

    num_synapses = len(experiment_dict["Params"]["allSegmentsType"])
    sim_duration_ms = int(experiment_dict["Params"]["totalSimDurationInSec"]) * 1000

    for simulation_dict in experiment_dict["Results"]["listOfSingleSimulationDicts"]:
        excitatory = dict_to_sparse(
            simulation_dict["exInputSpikeTimes"].keys(),
            simulation_dict["exInputSpikeTimes"].values(),
            num_synapses,
            sim_duration_ms,
        )
        inhibitory = dict_to_sparse(
            simulation_dict["inhInputSpikeTimes"].keys(),
            simulation_dict["inhInputSpikeTimes"].values(),
            num_synapses,
            sim_duration_ms,
        )
        spike_times = [min(sim_duration_ms - 1, max(0, int(x) - 1)) for x in simulation_dict["outputSpikeTimes"]]
        spikes = dict_to_sparse(
            (0,),
            (spike_times,),
            1,
            sim_duration_ms,
        )
        features = sps.hstack(
            (excitatory[:, 262:], inhibitory[:, 262:], excitatory[:, :262], inhibitory[:, :262]),
            format="coo",
        )
        voltage = np.asarray(simulation_dict["somaVoltageLowRes"], dtype=np.float32)
        np.savez_compressed(output_root / f"{global_index:08d}", X=features, y_spike=spikes, v_soma=voltage)
        global_index += 1
    return global_index


def preprocess_split(raw_root: Path, processed_root: Path, split_name: str) -> dict[str, Any]:
    processed_root.mkdir(parents=True, exist_ok=True)
    for stale_file in processed_root.glob("*.npz"):
        stale_file.unlink()

    files = sorted(raw_root.rglob("*.p"))
    if not files:
        raise FileNotFoundError(f"No raw L5PC experiment files found under {raw_root}")

    started = time.time()
    global_index = 0
    segments: dict[str, list[int]] = {}
    current_parent = files[0].parent.name
    segments[current_parent] = [0]

    for file_path in files:
        parent_name = file_path.parent.name
        if parent_name != current_parent:
            segments[current_parent].append(global_index)
            current_parent = parent_name
            segments[current_parent] = [global_index]
        global_index = _load_single_experiment_file(file_path, processed_root, global_index)

    segments[current_parent].append(global_index)
    metadata = {
        "split_name": split_name,
        "total": global_index,
        "segments": segments,
        "elapsed_sec": time.time() - started,
    }
    with (processed_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
    return metadata


def _resolve_group_name(metadata: dict[str, Any], group: int) -> str:
    candidates = [
        f"full_ergodic_train_batch_{group}",
        f"full_ergodic_train_batch_{group:02d}",
    ]
    for candidate in candidates:
        if candidate in metadata["segments"]:
            return candidate
    available = ", ".join(sorted(metadata["segments"].keys()))
    raise KeyError(f"Could not find training group {group}. Available metadata groups: {available}")


def _slice_indices(metadata: dict[str, Any], split_source: str, select_groups: Sequence[int | str] | None) -> tuple[int, ...] | None:
    if select_groups is None:
        return None

    indices: list[int] = []
    if split_source == TRAIN_SPLIT:
        if not all(isinstance(group, int) for group in select_groups):
            raise ValueError("L5PC training-group selection must use integer batch ids in [1, 10].")
        for group in select_groups:
            if group < 1 or group > 10:
                raise IndexError(f"Training batch id must be between 1 and 10, got {group}.")
            resolved = _resolve_group_name(metadata, group)
            start, stop = metadata["segments"][resolved]
            indices.extend(range(start, stop))
    else:
        if not all(isinstance(group, str) for group in select_groups):
            raise ValueError("L5PC test-group selection must use keyword strings such as 'erg'.")
        for group in select_groups:
            if group not in TEST_GROUP_MAP:
                raise KeyError(f"Unknown L5PC test keyword {group!r}. Expected one of {sorted(TEST_GROUP_MAP)}.")
            resolved = TEST_GROUP_MAP[group]
            if resolved not in metadata["segments"]:
                raise KeyError(f"Metadata does not contain test segment {resolved!r}.")
            start, stop = metadata["segments"][resolved]
            indices.extend(range(start, stop))
    return tuple(indices)


class L5PCDataset(Dataset):
    in_features = 1278
    ex_apical_features = 377
    inh_apical_features = 377
    ex_basal_features = 262
    inh_basal_features = 262
    max_t = 6000
    v_rest = -76.0

    def __init__(
        self,
        root: str | Path,
        split_source: str,
        mode: str,
        crop_length: int,
        train_repeats: int = 20,
        select_groups: Sequence[int | str] | None = None,
        clip_voltage_above: float = -55.0,
        voltage_offset: float = -67.6,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split_source = split_source
        self.mode = mode
        self.crop_length = int(crop_length)
        self.train_repeats = int(train_repeats)
        self.clip_voltage_above = float(clip_voltage_above)
        self.voltage_offset = float(voltage_offset)

        if self.split_source == TRAIN_SPLIT:
            self.raw_root = self.root / "TrainingData"
            self.processed_root = self.root / "train_data_processed"
        elif self.split_source == TEST_SPLIT:
            self.raw_root = self.root / "Data_test_combined_ergodic"
            self.processed_root = self.root / "test_erg_processed"
        else:
            raise ValueError(f"Unknown L5PC split source: {split_source}")

        self.metadata_path = self.processed_root / "metadata.json"
        self.metadata = self._load_or_create_metadata()
        self.base_indices = _slice_indices(self.metadata, self.split_source, select_groups)
        self.base_total = len(self.base_indices) if self.base_indices is not None else int(self.metadata["total"])

        if self.mode not in {"train", "eval"}:
            raise ValueError(f"Unknown L5PC dataset mode: {mode}")
        if self.mode == "train" and self.split_source != TRAIN_SPLIT:
            raise ValueError("Training mode must use the train split source.")
        if self.mode == "train" and self.crop_length <= 0:
            raise ValueError("Training crop length must be positive.")

    def _load_or_create_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return preprocess_split(self.raw_root, self.processed_root, split_name=self.split_source)

        with self.metadata_path.open("r", encoding="utf-8") as handle:
            metadata: dict[str, Any] = json.load(handle)
        expected = int(metadata.get("total", -1))
        actual = len(list(self.processed_root.glob("*.npz")))
        if expected <= 0 or actual != expected:
            raise FileNotFoundError(
                f"L5PC cache under {self.processed_root} is incomplete or stale. "
                f"Delete the directory and rerun preprocessing."
            )
        return metadata

    def __len__(self) -> int:
        if self.mode == "train":
            return self.base_total * self.train_repeats
        return self.base_total

    def _base_index(self, item_index: int) -> int:
        base_index = item_index // self.train_repeats if self.mode == "train" else item_index
        if base_index < 0 or base_index >= self.base_total:
            raise IndexError(f"L5PC example index {base_index} is out of range [0, {self.base_total}).")
        if self.base_indices is None:
            return base_index
        return self.base_indices[base_index]

    def _training_crop_bounds(self, repeat_index: int) -> tuple[int, int]:
        max_start = max(0, self.max_t - self.crop_length)
        bucket_span = max(1, math.ceil((max_start + 1) / self.train_repeats))
        bucket_start = min(repeat_index * bucket_span, max_start)
        bucket_end = min(max_start, bucket_start + bucket_span - 1)
        return bucket_start, bucket_end

    def _load_dense_example(self, base_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path = self.processed_root / f"{base_index:08d}.npz"
        loaded = np.load(path, allow_pickle=True)
        features = np.asarray(loaded["X"].item().todense(), dtype=np.float32)
        spikes = np.asarray(loaded["y_spike"].item().todense(), dtype=np.float32).reshape(-1)
        voltage = np.asarray(loaded["v_soma"], dtype=np.float32)
        return features, spikes, voltage

    def __getitem__(self, item_index: int) -> dict[str, torch.Tensor]:
        base_index = self._base_index(item_index)
        features, spikes, voltage = self._load_dense_example(base_index)

        crop_start = 0
        if self.mode == "train":
            repeat_index = item_index % self.train_repeats
            lower, upper = self._training_crop_bounds(repeat_index)
            crop_start = random.randint(lower, upper)
            crop_stop = crop_start + self.crop_length
            features = features[crop_start:crop_stop]
            spikes = spikes[crop_start:crop_stop]
            voltage = voltage[crop_start:crop_stop]

        clipped_voltage = np.minimum(voltage, self.clip_voltage_above).astype(np.float32)
        normalized_voltage = clipped_voltage - self.voltage_offset

        return {
            "inputs": torch.from_numpy(features),
            "spike_targets": torch.from_numpy(spikes),
            "voltage_targets": torch.from_numpy(normalized_voltage),
            "raw_voltage_targets": torch.from_numpy(clipped_voltage),
            "example_index": torch.tensor(base_index, dtype=torch.long),
            "crop_start": torch.tensor(crop_start, dtype=torch.long),
        }


def prepare_l5pc_cache(root: str | Path) -> None:
    root_path = Path(root)
    for split_source in [TRAIN_SPLIT, TEST_SPLIT]:
        if split_source == TRAIN_SPLIT:
            raw_root = root_path / "TrainingData"
            processed_root = root_path / "train_data_processed"
        else:
            raw_root = root_path / "Data_test_combined_ergodic"
            processed_root = root_path / "test_erg_processed"
        metadata_path = processed_root / "metadata.json"
        if metadata_path.exists():
            continue
        preprocess_split(raw_root, processed_root, split_name=split_source)


