"""
Lightweight testing and validation suite for the change detection project.

Run all tests from the project root with:
    pytest -q

You can also run this file directly:
    python tests/test_pipeline.py

Expected successful output should look similar to:
    13 passed in X.XXs

How to interpret the tests:
- Passing tests mean the local data folders are internally aligned, the
  preprocessing utilities return tensors in the expected format, model
  configuration files are consistent, and saved evaluation outputs have valid
  schemas and value ranges.
- These tests do not retrain the deep learning models or prove that a model is
  optimal. They validate that the reported experiments are based on coherent
  inputs, configs, and output artifacts.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data"
CONFIG_ROOT = PROJECT_ROOT / "configs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

SPLITS = ("train", "val", "test")
MODELS = ("unet", "deeplabv3plus", "segformer")
MODEL_DISPLAY_NAMES = {
    "unet": "U-Net",
    "deeplabv3plus": "DeepLabV3+",
    "segformer": "SegFormer",
}
MODEL_CONFIG_REQUIRED_FIELDS = {
    "model_name",
    "input_channels",
    "num_classes",
    "data_root",
    "train_split",
    "val_split",
    "test_split",
    "image_size",
    "batch_size",
    "num_epochs",
    "learning_rate",
    "threshold",
    "seed",
    "output_dir",
}
METRIC_COLUMNS = {
    "model",
    "iou",
    "f1",
    "precision",
    "recall",
    "params_m",
    "inference_time_ms",
}


def _png_ids(folder: Path) -> set[str]:
    return {path.stem for path in folder.glob("*.png")}


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _parse_simple_yaml(path: Path) -> dict[str, object]:
    """Parse the simple top-level key/value YAML files used in this project."""
    values: dict[str, object] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or not raw_value:
            continue

        lowered = raw_value.lower()
        if lowered in {"true", "false"}:
            value: object = lowered == "true"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value.strip("'\"")
        values[key] = value
    return values


@pytest.mark.parametrize("split", SPLITS)
def test_data_folders_are_aligned(split: str) -> None:
    split_dir = DATA_ROOT / split
    time1_ids = _png_ids(split_dir / "time1")
    time2_ids = _png_ids(split_dir / "time2")
    label_ids = _png_ids(split_dir / "label")

    assert time1_ids, f"No images found for split: {split}"
    assert time1_ids == time2_ids == label_ids


@pytest.mark.parametrize("split", SPLITS)
def test_dataset_sample_has_expected_tensor_shapes(split: str) -> None:
    data_utils = pytest.importorskip("src.data_utils")

    dataset = data_utils.SYSUCDDataset(root_dir=str(DATA_ROOT), split=split)
    sample = dataset[0]

    assert sample["image"].shape[0] == 6
    assert sample["mask"].shape[0] == 1
    assert sample["image"].shape[1:] == sample["mask"].shape[1:]
    assert sample["image"].dtype.is_floating_point
    assert sample["mask"].dtype.is_floating_point
    assert isinstance(sample["id"], str)
    assert sample["id"]


@pytest.mark.parametrize("split", SPLITS)
def test_masks_are_loaded_as_binary_values(split: str) -> None:
    data_utils = pytest.importorskip("src.data_utils")

    label_dir = DATA_ROOT / split / "label"
    sample_paths = sorted(label_dir.glob("*.png"))[:10]

    assert sample_paths, f"No masks found for split: {split}"
    for path in sample_paths:
        mask = data_utils.read_mask(path)
        assert set(mask.flatten()).issubset({0.0, 1.0})


def test_model_configs_are_valid_and_consistent() -> None:
    configs = {
        model: _parse_simple_yaml(CONFIG_ROOT / f"{model}.yaml")
        for model in MODELS
    }

    for model, config in configs.items():
        missing_fields = MODEL_CONFIG_REQUIRED_FIELDS - config.keys()
        assert not missing_fields, f"{model} config is missing: {missing_fields}"
        assert config["model_name"] == model
        assert config["input_channels"] == 6
        assert config["num_classes"] == 1
        assert 0 <= config["threshold"] <= 1
        assert config["batch_size"] > 0
        assert config["num_epochs"] > 0
        assert config["learning_rate"] > 0

    shared_fields = ("train_split", "val_split", "test_split", "image_size", "seed")
    baseline = configs["unet"]
    for model, config in configs.items():
        for field in shared_fields:
            assert config[field] == baseline[field], (
                f"{model} uses a different {field}: {config[field]}"
            )


def test_metric_files_have_valid_schema_and_ranges() -> None:
    for model in MODELS:
        path = OUTPUT_ROOT / model / "metrics.csv"
        rows = _read_csv(path)

        assert rows, f"No metric rows found in {path}"
        assert METRIC_COLUMNS.issubset(rows[0].keys())

        row = rows[0]
        assert row["model"] == model
        for metric in ("iou", "f1", "precision", "recall"):
            value = float(row[metric])
            assert 0.0 <= value <= 1.0, f"{model} {metric} is out of range"

        assert float(row["params_m"]) > 0
        assert float(row["inference_time_ms"]) >= 0


def test_shared_test_ids_are_consistent_across_models() -> None:
    shared_ids = {
        model: _read_lines(OUTPUT_ROOT / model / "shared_test_ids.txt")
        for model in MODELS
    }
    subset_test_ids = set(_read_lines(DATA_ROOT / "subsets" / "subset_test_500.txt"))

    baseline = shared_ids["unet"]
    assert baseline, "Shared test ID file is empty"
    assert set(baseline).issubset(subset_test_ids)

    for model, ids in shared_ids.items():
        assert ids == baseline, f"{model} was evaluated on different shared IDs"


def test_prediction_coverage_is_complete() -> None:
    coverage_rows = _read_csv(OUTPUT_ROOT / "error_analysis" / "prediction_coverage.csv")
    rows_by_model = {row["model"]: row for row in coverage_rows}

    assert set(MODELS).issubset(rows_by_model.keys())
    for model in MODELS:
        row = rows_by_model[model]
        predicted_samples = int(row["predicted_samples"])
        total_test_patches = int(row["total_test_patches"])
        prediction_files = list(
            (OUTPUT_ROOT / model / "test_predictions_full").glob("*_pred.png")
        )

        assert row["display_name"] == MODEL_DISPLAY_NAMES[model]
        assert row["status"] == "checkpoint_loaded"
        assert predicted_samples == total_test_patches
        assert predicted_samples > 0
        assert len(prediction_files) == total_test_patches


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
