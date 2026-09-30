import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_headless_degradation_outputs_valid_json_on_windows():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "minetwin",
            "--scenario",
            "engine_degradation",
            "--steps",
            "180",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(result.stdout)
    assert output["steps"] == 180
    assert output["engine_status"] == "alert"
    assert output["alerts"] == 1


def test_headless_invalid_steps_returns_usage_error():
    result = subprocess.run(
        [sys.executable, "-m", "minetwin", "--steps", "0"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "steps" in result.stderr
    assert not result.stdout


def test_comparison_command_exports_all_three_policies(tmp_path):
    output = tmp_path / "comparison"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "minetwin",
            "--scenario",
            "engine_degradation",
            "--steps",
            "30",
            "--compare",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    metrics = json.loads(result.stdout)
    assert [row["policy"] for row in metrics] == ["none", "threshold", "predictive"]
    assert (output / "comparison.csv").exists()
    assert all((output / row["policy"] / "manifest.json").exists() for row in metrics)


def test_fleet_evaluation_command_accepts_explicit_dataset_split(tmp_path):
    output = tmp_path / "fleet-evaluation"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "minetwin",
            "--evaluate-fleet",
            "--steps",
            "1",
            "--seeds",
            "201",
            "--scenarios",
            "normal",
            "--dataset-role",
            "calibration",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)
    assert len(summary) == 9
    assert {row["dataset_role"] for row in summary} == {"calibration"}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["evaluation"]["seeds"] == [201]


def test_evaluation_commands_require_a_new_output_directory():
    for mode in ("--evaluate-fleet", "--evaluate-langflow"):
        result = subprocess.run(
            [sys.executable, "-m", "minetwin", mode],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
        assert "--output" in result.stderr
        assert not result.stdout
