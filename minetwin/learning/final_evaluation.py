import json
import shutil
from pathlib import Path

from minetwin.learning.contracts import TrainingConfig
from minetwin.learning.data import CachedSplit
from minetwin.learning.diagnostics import run_model_diagnostics
from minetwin.learning.federation import export_federated_inference
from minetwin.learning.interpretability import (
    PermutationConfig,
    run_permutation_importance,
)
from minetwin.learning.provenance import file_sha256, verify_training_provenance
from minetwin.learning.training import train_learning_regimes


def run_final_evaluation(
    cache: Path,
    comparison: Path,
    policy: Path,
    output: Path,
) -> dict:
    selection = _read_json(Path(comparison) / "selection.json")
    priority = _read_json(Path(policy) / "policy.json")
    config = _training_config(selection, priority)
    cache = Path(cache)
    _validate_cache(cache, selection)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        models = output / "models"
        training = train_learning_regimes(
            cache,
            models,
            config,
            weighted_loss=False,
            decision="cost",
            prior_correction=False,
        )
        provenance = verify_training_provenance(cache, models, training)
        federation = export_federated_inference(
            cache,
            models,
            output / "federation",
            split="validation",
            regime="fedprox",
        )
        diagnostics = run_model_diagnostics(
            cache,
            models,
            output / "diagnostic",
        )
        interpretability = run_permutation_importance(
            cache,
            models,
            output / "interpretability",
            split="validation",
            regime="fedprox",
            config=PermutationConfig(seed=config.seed),
        )
        report = {
            "selection": {
                "representation": selection["selected_representation"],
                "model": selection["selected_model"],
                "priority_capacity": priority["selected_capacity"],
                "priority_score": priority["priority_score"],
            },
            "training_config": training["config"],
            "validation": {
                "role": "final descriptive evaluation; not used for selection",
                "examples": training["validation_examples"],
                "metrics": training["results"],
            },
            "artifacts": {
                "models": "models",
                "federation": "federation/summary.json",
                "diagnostic": "diagnostic/diagnostics.json",
                "interpretability": "interpretability/summary.json",
            },
            "provenance": {
                "training": provenance,
                "comparison_sha256": file_sha256(Path(comparison) / "selection.json"),
                "policy_sha256": file_sha256(Path(policy) / "policy.json"),
            },
            "generated": {
                "federation": federation,
                "diagnostic": diagnostics,
                "interpretability": interpretability,
            },
        }
        _write_json(output / "report.json", report)
        return {
            "output": str(output),
            "priority_capacity": priority["selected_capacity"],
            "validation_examples": training["validation_examples"],
            "regimes": sorted(training["results"]),
        }
    except (Exception, KeyboardInterrupt):
        shutil.rmtree(output)
        raise


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"No existe el artefacto requerido: {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def _training_config(selection: dict, priority: dict) -> TrainingConfig:
    if selection.get("selected_model") != "mlp":
        raise ValueError(
            "La evaluación federada requiere que el modelo seleccionado sea MLP."
        )
    if priority.get("selected_model") != selection["selected_model"]:
        raise ValueError("La política y la selección no usan el mismo modelo.")
    if priority.get("selected_representation") != selection["selected_representation"]:
        raise ValueError("La política y la selección no usan la misma representación.")
    config = dict(selection["mlp_config"])
    config["hidden_sizes"] = tuple(config["hidden_sizes"])
    return TrainingConfig(**config)


def _validate_cache(cache: Path, selection: dict) -> None:
    representation = selection["representations"][selection["selected_representation"]]
    cached = CachedSplit.load(cache / "train.npz")
    if cached.features.shape[1] != representation["features"]:
        raise ValueError("La caché no coincide con la representación seleccionada.")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
