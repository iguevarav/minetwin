from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATASET_ROOT = PACKAGE_ROOT / "data" / "scania"
RESULTS_ROOT = PROJECT_ROOT / "results"


def result_path(*parts: str) -> Path:
    return RESULTS_ROOT.joinpath(*parts)
