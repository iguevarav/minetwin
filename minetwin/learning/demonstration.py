import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from minetwin.learning.provenance import file_sha256


def prepare_demonstration(policy: Path, evaluation: Path, output: Path) -> dict:
    policy = Path(policy)
    evaluation = Path(evaluation)
    policy_config = _read_json(policy / "policy.json")
    evaluation_report = _read_json(evaluation / "report.json")
    rows = _read_csv(policy / "oof_priority_queue.csv")
    if not rows:
        raise ValueError("La política no contiene predicciones OOF.")
    _validate_queue(rows)
    cases = _stratified_cases(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        _write_csv(output / "demo_cases.csv", cases)
        manifest = {
            "scope": "cross_validated_train_demonstration",
            "source": "real SCANIA Component X causal windows",
            "selection": "up to two cases per experimental node and observed class",
            "priority": {
                "capacity": policy_config["selected_capacity"],
                "score": policy_config["priority_score"],
                "model": policy_config["selected_model"],
                "representation": policy_config["selected_representation"],
                "seed": policy_config["model_config"]["seed"],
            },
            "evaluation": {
                "source": str(evaluation.resolve()),
                "sha256": file_sha256(evaluation / "report.json"),
                "validation_role": evaluation_report["validation"]["role"],
            },
            "policy": {
                "source": str(policy.resolve()),
                "sha256": file_sha256(policy / "policy.json"),
            },
            "cases": len(cases),
        }
        _write_json(output / "manifest.json", manifest)
        return {"output": str(output), "cases": len(cases)}
    except (Exception, KeyboardInterrupt):
        shutil.rmtree(output)
        raise


def _stratified_cases(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["node_id"], int(row["label"]))].append(row)
    cases = []
    for (node, label), members in sorted(groups.items()):
        ordered = sorted(
            members,
            key=lambda row: (
                row["selected"] != "True",
                -float(row["calibrated_probability_4"]),
                row["vehicle_id"],
            ),
        )
        cases.extend(
            {
                **row,
                "stratum": f"{node}: clase {label}",
                "priority": (
                    "Revisión humana prioritaria"
                    if row["selected"] == "True"
                    else "No priorizado en la capacidad seleccionada"
                ),
            }
            for row in ordered[:2]
        )
    return cases


def _validate_queue(rows: list[dict]) -> None:
    required = {
        "fold",
        "vehicle_id",
        "node_id",
        "time_step",
        "window_observations",
        "label",
        "most_likely_class",
        "maximum_probability",
        "published_decision",
        "published_expected_cost",
        "calibrated_probability_4",
        "selected",
        *(f"probability_{label}" for label in range(5)),
        *(f"expected_cost_{label}" for label in range(5)),
    }
    if missing := required.difference(rows[0]):
        raise ValueError(
            "La política no contiene las variables de demostración: "
            f"{', '.join(sorted(missing))}. Ejecute prioritize nuevamente."
        )


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"No existe el artefacto requerido: {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"No existe el artefacto requerido: {path}.")
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
