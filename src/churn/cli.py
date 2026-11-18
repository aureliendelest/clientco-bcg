"""Command line entry point: `python -m churn <command>` (or `churn <command>`)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def _cmd_ingest(args: argparse.Namespace) -> None:
    from churn.config import load_config
    from churn.data.ingest import ingest

    ingest(load_config(args.config), overwrite=args.overwrite)


def _cmd_run(args: argparse.Namespace) -> None:
    from churn.config import load_config
    from churn.pipeline.runner import run

    run(load_config(args.config))


def _cmd_sweep(args: argparse.Namespace) -> None:
    from churn.pipeline.sweep import run_sweep

    run_sweep(args.sweep)


def _cmd_collect(args: argparse.Namespace) -> None:
    from churn.pipeline.collect import collect

    collect()


def _cmd_score(args: argparse.Namespace) -> None:
    from datetime import date

    from churn.scoring.batch import score_batch

    score_batch(args.run_id, date.fromisoformat(args.date))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="churn", description="ClientCo churn prevention experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Build data/sales_shifted.parquet from the raw CSV (+2 years).")
    p.add_argument("config", nargs="?", default="configs/base.yaml", help="Config providing the dataset paths.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite an existing shifted Parquet.")
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("run", help="Run one experiment from a YAML config.")
    p.add_argument("config", help="Path to an experiment YAML (e.g. configs/experiments/hgb.yaml).")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("sweep", help="Run a series of experiments from a sweep YAML.")
    p.add_argument("sweep", help="Path to a sweep YAML (e.g. configs/sweeps/models.yaml).")
    p.set_defaults(func=_cmd_sweep)

    p = sub.add_parser("collect", help="Merge all runs/ into results/*.parquet.")
    p.set_defaults(func=_cmd_collect)

    p = sub.add_parser("score", help="Write the weekly action list (CSV) for a run and a snapshot date.")
    p.add_argument("run_id", help="Run folder name under runs/.")
    p.add_argument("--date", required=True, help="Snapshot date, ISO format (YYYY-MM-DD).")
    p.set_defaults(func=_cmd_score)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except NotImplementedError as exc:
        print(f"churn {args.command}: not implemented yet ({exc})", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
