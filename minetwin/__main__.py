import argparse
import json
from pathlib import Path

from minetwin.data.scania import prepare_replay_store, prepare_scania_dataset
from minetwin.domain import MaintenancePolicy, Scenario
from minetwin.evaluation import EvaluationConfig, run_fleet_evaluation
from minetwin.experiments import run_comparison, run_experiment
from minetwin.federation import FederatedFleet
from minetwin.langflow_evaluation import (
    run_langflow_evaluation,
    summarize_langflow_review,
)
from minetwin.services import MineTwin
from minetwin.simulation import SimulationConfig


def _seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Las semillas deben ser enteros separados por comas."
        ) from error
    if not seeds or any(seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError(
            "Las semillas deben ser únicas y no negativas."
        )
    return seeds


def _scenarios(value: str) -> tuple[Scenario, ...]:
    try:
        scenarios = tuple(Scenario(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "La lista contiene un escenario desconocido."
        ) from error
    if not scenarios or len(scenarios) != len(set(scenarios)):
        raise argparse.ArgumentTypeError("Los escenarios deben ser únicos.")
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Herramientas de MineTwin sin interfaz."
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=list(Scenario), default=Scenario.NORMAL)
    parser.add_argument(
        "--policy", choices=list(MaintenancePolicy), default=MaintenancePolicy.NONE
    )
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--compare", action="store_true")
    modes.add_argument("--fleet", action="store_true")
    modes.add_argument("--evaluate-fleet", action="store_true")
    modes.add_argument("--evaluate-langflow", action="store_true")
    modes.add_argument("--summarize-langflow-review", action="store_true")
    modes.add_argument(
        "--prepare-scania",
        type=Path,
        metavar="DATASET",
        help="Valida y registra el dataset oficial SCANIA Component X.",
    )
    modes.add_argument(
        "--prepare-scania-replay",
        type=Path,
        metavar="DATASET",
        help="Prepara el acceso por vehículo a los readouts SCANIA.",
    )
    parser.add_argument(
        "--phase1",
        type=Path,
        help="Directorio generado previamente por --prepare-scania.",
    )
    parser.add_argument(
        "--seeds",
        type=_seeds,
        help=(
            "Semillas de repetición separadas por comas; cada conjunto usa "
            "valores predeterminados distintos."
        ),
    )
    parser.add_argument(
        "--scenarios",
        type=_scenarios,
        default=(
            Scenario.NORMAL,
            Scenario.ENGINE_DEGRADATION,
            Scenario.ENGINE_VARIABLE_DEGRADATION,
        ),
        help="Escenarios de evaluación separados por comas.",
    )
    parser.add_argument(
        "--dataset-role",
        choices=("calibration", "evaluation"),
        default="evaluation",
    )
    args = parser.parse_args()
    steps = (
        args.steps if args.steps is not None else (450 if args.evaluate_fleet else 180)
    )
    if args.compare and args.output is None:
        parser.error("--compare requiere --output con un directorio nuevo.")
    if (
        args.evaluate_fleet
        or args.evaluate_langflow
        or args.prepare_scania is not None
        or args.prepare_scania_replay is not None
    ) and args.output is None:
        parser.error("El modo seleccionado requiere --output con un directorio nuevo.")
    if args.prepare_scania_replay is not None and args.phase1 is None:
        parser.error("--prepare-scania-replay requiere --phase1.")
    if args.phase1 is not None and args.prepare_scania_replay is None:
        parser.error("--phase1 solo se utiliza con --prepare-scania-replay.")
    if args.summarize_langflow_review and args.output is None:
        parser.error(
            "--summarize-langflow-review requiere el directorio de evaluación."
        )
    if args.fleet and (
        args.output or args.compare or args.policy != MaintenancePolicy.NONE
    ):
        parser.error(
            "--fleet demuestra la federación; la evaluación por políticas y "
            "exportación de flota corresponde a la fase 4."
        )
    try:
        if args.prepare_scania is not None:
            print(
                json.dumps(
                    prepare_scania_dataset(args.prepare_scania, args.output),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        if args.prepare_scania_replay is not None:
            print(
                json.dumps(
                    prepare_replay_store(
                        args.prepare_scania_replay,
                        args.phase1,
                        args.output,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        config = SimulationConfig(seed=args.seed, scenario=Scenario(args.scenario))
        if args.fleet:
            fleet = FederatedFleet(config)
            fleet.advance(steps)
            print(
                json.dumps(
                    [
                        {
                            "truck_id": truck_id,
                            "node_id": fleet.view(truck_id).node_id,
                            "profile": fleet.view(truck_id).profile.id,
                            "simulated_time": fleet.time.isoformat(),
                            "engine_status": fleet.view(truck_id).twin.diagnosis.status,
                            "component_statuses": {
                                item.component: item.status
                                for item in fleet.view(
                                    truck_id
                                ).twin.component_diagnoses
                            },
                            "alerts": len(fleet.view(truck_id).alerts),
                        }
                        for truck_id in fleet.truck_ids
                    ],
                    indent=2,
                )
            )
            return
        if args.evaluate_fleet:
            result = run_fleet_evaluation(
                args.output,
                EvaluationConfig(
                    steps=steps,
                    seeds=args.seeds
                    or (
                        tuple(range(10))
                        if args.dataset_role == "calibration"
                        else tuple(range(100, 110))
                    ),
                    scenarios=args.scenarios,
                    dataset_role=args.dataset_role,
                ),
            )
            print(json.dumps(result, indent=2))
            return
        if args.evaluate_langflow:
            records = run_langflow_evaluation(args.output)
            print(
                json.dumps(
                    {
                        "cases": len(records),
                        "responses": sum(
                            record["response"] is not None for record in records
                        ),
                        "errors": sum(
                            record["error"] is not None for record in records
                        ),
                        "output": str(args.output),
                    },
                    indent=2,
                )
            )
            return
        if args.summarize_langflow_review:
            print(json.dumps(summarize_langflow_review(args.output), indent=2))
            return
        if args.output:
            result = (
                run_comparison(args.output, config, steps)
                if args.compare
                else run_experiment(
                    args.output, config, steps, MaintenancePolicy(args.policy)
                )
            )
            print(json.dumps(result, indent=2))
            return
        simulation = MineTwin(
            config,
            policy=MaintenancePolicy(args.policy),
        )
        twin = simulation.advance(steps)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "truck_id": twin.observation.truck_id,
                "scenario": simulation.config.scenario.value,
                "seed": simulation.config.seed,
                "steps": steps,
                "simulated_time": simulation.time.isoformat(),
                "cycles": twin.observation.cycles,
                "operating_hours": twin.observation.reading("operating_hours").value,
                "engine_status": twin.diagnosis.status.value,
                "explanation": twin.diagnosis.explanation,
                "alerts": simulation.alert_count,
                "asset_status": simulation.asset_status,
                "forecast_status": simulation.prediction.status,
                "hours_to_threshold": simulation.prediction.hours_to_threshold,
                "metrics": simulation.metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
