import argparse
import json
from pathlib import Path

from minetwin.learning.comparison import run_representation_comparison
from minetwin.learning.contracts import FeatureConfig, TrainingConfig
from minetwin.learning.data import prepare_learning_cache
from minetwin.learning.demonstration import prepare_demonstration
from minetwin.learning.diagnostics import run_model_diagnostics
from minetwin.learning.federation import export_federated_inference
from minetwin.learning.final_evaluation import run_final_evaluation
from minetwin.learning.interpretability import (
    PermutationConfig,
    run_permutation_importance,
)
from minetwin.learning.priority_policy import run_priority_policy
from minetwin.learning.selection import SelectionConfig, run_model_selection
from minetwin.learning.training import train_learning_regimes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preparacion y aprendizaje federado con SCANIA Component X."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--phase1", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--window-size", type=int, default=12)
    prepare.add_argument("--stride", type=int, default=1)
    prepare.add_argument("--usage-features", action="store_true")
    train = commands.add_parser("train")
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--rounds", type=int, default=10)
    train.add_argument("--local-epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--seed", type=int, default=42)
    select = commands.add_parser("select")
    select.add_argument("--cache", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--folds", type=int, default=5)
    select.add_argument("--seed", type=int, default=42)
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline-cache", type=Path, required=True)
    compare.add_argument("--usage-cache", type=Path, required=True)
    compare.add_argument("--baseline-selection", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--folds", type=int, default=5)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--review-fraction", type=float, default=0.01)
    prioritize = commands.add_parser("prioritize")
    prioritize.add_argument("--cache", type=Path, required=True)
    prioritize.add_argument("--comparison", type=Path, required=True)
    prioritize.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--cache", type=Path, required=True)
    evaluate.add_argument("--comparison", type=Path, required=True)
    evaluate.add_argument("--policy", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    demo = commands.add_parser("demo")
    demo.add_argument("--policy", type=Path, required=True)
    demo.add_argument("--evaluation", type=Path, required=True)
    demo.add_argument("--output", type=Path, required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--cache", type=Path, required=True)
    publish.add_argument("--models", type=Path, required=True)
    publish.add_argument("--output", type=Path, required=True)
    publish.add_argument("--split", choices=("validation", "test"), default="validation")
    publish.add_argument(
        "--regime",
        choices=("local", "centralized", "fedavg", "fedprox"),
        default="fedprox",
    )
    interpret = commands.add_parser("interpret")
    interpret.add_argument("--cache", type=Path, required=True)
    interpret.add_argument("--models", type=Path, required=True)
    interpret.add_argument("--output", type=Path, required=True)
    interpret.add_argument(
        "--split", choices=("train", "validation"), default="validation"
    )
    interpret.add_argument(
        "--regime",
        choices=("centralized", "fedavg", "fedprox"),
        default="fedprox",
    )
    interpret.add_argument("--repeats", type=int, default=3)
    interpret.add_argument("--seed", type=int, default=42)
    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("--cache", type=Path, required=True)
    diagnose.add_argument("--models", type=Path, required=True)
    diagnose.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_learning_cache(
                args.dataset,
                args.phase1,
                args.output,
                FeatureConfig(
                    window_size=args.window_size,
                    stride=args.stride,
                    usage_features=args.usage_features,
                ),
            )
        elif args.command == "train":
            result = train_learning_regimes(
                args.cache,
                args.output,
                TrainingConfig(
                    epochs=args.epochs,
                    federated_rounds=args.rounds,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    seed=args.seed,
                ),
            )
        elif args.command == "select":
            result = run_model_selection(
                args.cache,
                args.output,
                SelectionConfig(folds=args.folds, seed=args.seed),
            )
        elif args.command == "compare":
            result = run_representation_comparison(
                args.baseline_cache,
                args.usage_cache,
                args.baseline_selection,
                args.output,
                args.folds,
                args.seed,
                args.review_fraction,
            )
        elif args.command == "prioritize":
            result = run_priority_policy(args.cache, args.comparison, args.output)
        elif args.command == "evaluate":
            result = run_final_evaluation(
                args.cache,
                args.comparison,
                args.policy,
                args.output,
            )
        elif args.command == "demo":
            result = prepare_demonstration(
                args.policy,
                args.evaluation,
                args.output,
            )
        elif args.command == "publish":
            result = export_federated_inference(
                args.cache,
                args.models,
                args.output,
                args.split,
                args.regime,
            )
        elif args.command == "interpret":
            result = run_permutation_importance(
                args.cache,
                args.models,
                args.output,
                args.split,
                args.regime,
                PermutationConfig(repeats=args.repeats, seed=args.seed),
            )
        else:
            result = run_model_diagnostics(args.cache, args.models, args.output)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
