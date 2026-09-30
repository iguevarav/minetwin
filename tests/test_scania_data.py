import csv
from pathlib import Path

import pytest

from minetwin.data.scania import (
    CategoricalNodePartitioner,
    DatasetSplit,
    FeatureReadout,
    ScaniaDataset,
    ScaniaManifest,
    ScaniaReplaySession,
    ScaniaReplayStore,
    WindowConfig,
    build_windows,
    decode_readout,
    encode_readout,
    prepare_replay_store,
    prepare_scania_dataset,
)
from minetwin.domain import WireFormat


def _write(path: Path, fieldnames: tuple[str, ...], rows: list[tuple]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(fieldnames)
        writer.writerows(rows)


@pytest.fixture
def scania_data(tmp_path):
    root = tmp_path / "scania"
    root.mkdir()
    readout_fields = ("vehicle_id", "time_step", "feature_0", "feature_1")
    specification_fields = ("vehicle_id", "specification_0", "specification_1")
    _write(
        root / "train_operational_readouts.csv",
        readout_fields,
        [
            ("T-1", 0, "", 1),
            ("T-1", 4, 2, ""),
            ("T-1", 8, "", 3),
            ("T-2", 0, 4, 5),
            ("T-2", 10, 6, 7),
        ],
    )
    _write(
        root / "validation_operational_readouts.csv",
        readout_fields,
        [("V-1", 0, 8, 9), ("V-1", 3, 10, 11)],
    )
    _write(
        root / "test_operational_readouts.csv",
        readout_fields,
        [("E-1", 0, 12, 13), ("E-1", 2, 14, 15)],
    )
    _write(
        root / "train_specifications.csv",
        specification_fields,
        [("T-1", "Cat0", "Cat1"), ("T-2", "Cat1", "Cat1")],
    )
    _write(
        root / "validation_specifications.csv",
        specification_fields,
        [("V-1", "Cat0", "Cat1")],
    )
    _write(
        root / "test_specifications.csv",
        specification_fields,
        [("E-1", "Cat2", "Cat0")],
    )
    _write(
        root / "train_tte.csv",
        ("vehicle_id", "length_of_study_time_step", "in_study_repair"),
        [("T-1", 50, 1), ("T-2", 60, 0)],
    )
    _write(
        root / "validation_labels.csv",
        ("vehicle_id", "class_label"),
        [("V-1", 3)],
    )
    return root


def test_dataset_preserves_official_splits_and_validates_dynamic_schema(scania_data):
    dataset = ScaniaDataset(scania_data)
    summaries = dataset.validate()
    assert dataset.schema.feature_names == ("feature_0", "feature_1")
    assert dataset.schema.specification_names == (
        "specification_0",
        "specification_1",
    )
    assert [(row.split, row.vehicles, row.readouts) for row in summaries] == [
        (DatasetSplit.TRAIN, 2, 5),
        (DatasetSplit.VALIDATION, 1, 2),
        (DatasetSplit.TEST, 1, 2),
    ]
    assert summaries[0].missing_values == 3


def test_training_labels_are_derived_from_time_to_event_without_future_rows(
    scania_data,
):
    outcomes = ScaniaDataset(scania_data).load_outcomes(DatasetSplit.TRAIN)
    repaired = outcomes["T-1"]
    assert [repaired.class_at(time) for time in (0, 2, 26, 38, 44, 50)] == [
        0,
        1,
        2,
        3,
        4,
        4,
    ]
    assert outcomes["T-2"].class_at(59) == 0


def test_causal_windows_use_only_prior_values_and_never_cross_vehicles(scania_data):
    dataset = ScaniaDataset(scania_data)
    specifications = dataset.load_specifications(DatasetSplit.TRAIN)
    windows = list(
        build_windows(
            dataset.iter_readouts(DatasetSplit.TRAIN),
            specifications,
            CategoricalNodePartitioner(seed=7),
            WindowConfig(size=3),
        )
    )
    assert len(windows) == 1
    assert windows[0].vehicle_id == "T-1"
    assert windows[0].time_steps == (0, 4, 8)
    assert windows[0].values == ((0.0, 1.0), (2.0, 1.0), (2.0, 3.0))
    assert windows[0].missing == (
        (True, False),
        (False, True),
        (True, False),
    )


@pytest.mark.parametrize("wire_format", list(WireFormat))
def test_generic_readout_is_equivalent_across_wire_formats(wire_format):
    readout = FeatureReadout(
        DatasetSplit.VALIDATION,
        "V-20",
        12,
        ("variable_0", "variable_1"),
        (3.5, None),
    )
    encoded = encode_readout(readout, wire_format)
    assert decode_readout(encoded, wire_format, readout.feature_names) == readout


def test_partitioner_is_reproducible_without_labels(scania_data):
    specifications = ScaniaDataset(scania_data).load_specifications(DatasetSplit.TRAIN)
    first = CategoricalNodePartitioner(seed=19)
    second = CategoricalNodePartitioner(seed=19)
    assert [first.assign(item) for item in specifications.values()] == [
        second.assign(item) for item in specifications.values()
    ]


def test_manifest_detects_source_changes(scania_data):
    manifest = ScaniaManifest.build(scania_data)
    manifest.verify(scania_data)
    with (scania_data / "train_tte.csv").open("a", encoding="utf-8") as target:
        target.write("\n")
    with pytest.raises(ValueError, match="manifiesto"):
        manifest.verify(scania_data)


def test_preparation_exports_reproducible_metadata_and_partitions(
    scania_data, tmp_path
):
    output = tmp_path / "prepared"
    result = prepare_scania_dataset(scania_data, output)
    assert result["features"] == 2
    assert sum(result["nodes"].values()) == 4
    assert {path.name for path in output.iterdir()} == {
        "dataset_manifest.json",
        "experiment.json",
        "partitions.csv",
        "summary.json",
    }


def test_replay_reads_only_the_selected_vehicle_and_builds_a_causal_window(
    scania_data, tmp_path
):
    phase_one = tmp_path / "phase1"
    prepare_scania_dataset(scania_data, phase_one)
    replay = tmp_path / "replay"
    summary = prepare_replay_store(scania_data, phase_one, replay)
    assert summary["vehicles"] == 4
    assert summary["readouts"] == 9
    store = ScaniaReplayStore(scania_data, replay / "index.json")
    session = ScaniaReplaySession(
        store,
        DatasetSplit.TRAIN,
        vehicle_id="T-1",
        window_size=3,
    )
    session.advance(2)
    view = session.view
    assert view.vehicle_id == "T-1"
    assert view.readout_count == 3
    assert tuple(item.time_step for item in view.history) == (0, 4, 8)
    assert view.observed_class == 1
    assert view.window.time_steps == (0, 4, 8)
    assert view.window.values == ((0.0, 1.0), (2.0, 1.0), (2.0, 3.0))
    assert view.window.missing == (
        (True, False),
        (False, True),
        (True, False),
    )
    validation = ScaniaReplaySession(
        store,
        DatasetSplit.VALIDATION,
        vehicle_id="V-1",
    )
    assert validation.view.observed_class is None
    validation.advance()
    assert validation.view.observed_class == 3
