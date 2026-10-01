import hashlib
import json
from pathlib import Path

CACHE_FILES = ("train.npz", "validation.npz", "metadata.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value: dict) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def training_provenance(
    cache: Path, models: Path, config: dict, torch_version: str
) -> dict:
    model_files = sorted(Path(models).glob("*.pt"))
    if not model_files:
        raise ValueError("No se encontraron modelos para vincular al entrenamiento.")
    identity = {
        "cache": {name: file_sha256(Path(cache) / name) for name in CACHE_FILES},
        "scaler": file_sha256(Path(models) / "scaler.npz"),
        "models": {path.name: file_sha256(path) for path in model_files},
        "config": config,
        "torch": torch_version,
    }
    return {"schema_version": 1, "run_id": value_sha256(identity), **identity}


def verify_training_provenance(
    cache: Path, models: Path, report: dict
) -> dict:
    expected = report.get("provenance")
    if expected is None:
        raise ValueError(
            "Los modelos no incluyen procedencia verificable. Regenera el estudio "
            "de aprendizaje antes de publicar estados."
        )
    actual = training_provenance(cache, models, report["config"], report["torch"])
    if actual != expected:
        raise ValueError(
            "La caché, el escalador, los modelos o la configuración no coinciden "
            "con el entrenamiento registrado."
        )
    return actual
