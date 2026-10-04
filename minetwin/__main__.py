import argparse
import json
from pathlib import Path

from minetwin.data.scania import prepare_replay_store, prepare_scania_dataset
from minetwin.langflow_evaluation import (
    run_langflow_evaluation,
    summarize_langflow_review,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Herramientas SCANIA de MineTwin.")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--evaluate-langflow", action="store_true")
    modes.add_argument("--summarize-langflow-review", action="store_true")
    modes.add_argument("--prepare-scania", type=Path, metavar="DATASET")
    modes.add_argument("--prepare-scania-replay", type=Path, metavar="DATASET")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase1", type=Path)
    parser.add_argument("--flow-definition", type=Path)
    args = parser.parse_args()
    if args.output is None:
        parser.error("El modo seleccionado requiere --output.")
    if args.prepare_scania_replay is not None and args.phase1 is None:
        parser.error("--prepare-scania-replay requiere --phase1.")
    if args.phase1 is not None and args.prepare_scania_replay is None:
        parser.error("--phase1 solo se utiliza con --prepare-scania-replay.")
    if args.flow_definition is not None and not args.evaluate_langflow:
        parser.error("--flow-definition solo se utiliza con --evaluate-langflow.")
    try:
        if args.prepare_scania is not None:
            result = prepare_scania_dataset(args.prepare_scania, args.output)
        elif args.prepare_scania_replay is not None:
            result = prepare_replay_store(
                args.prepare_scania_replay, args.phase1, args.output
            )
        elif args.evaluate_langflow:
            records = run_langflow_evaluation(
                args.output, flow_definition=args.flow_definition
            )
            result = {
                "cases": len(records),
                "responses": sum(record["response"] is not None for record in records),
                "errors": sum(record["error"] is not None for record in records),
                "output": str(args.output),
            }
        else:
            result = summarize_langflow_review(args.output)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
