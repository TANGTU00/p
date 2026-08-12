from __future__ import annotations

import argparse
from pathlib import Path

from doctoral_models.config import load_config
from doctoral_models.interior_pr import train_interior_pr
from doctoral_models.single_target import train_single_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train doctoral building element prediction models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a single-target classifier.")
    train.add_argument("--config", required=True, help="Path to a YAML config.")
    train.add_argument("--data", required=True, help="Path to a CSV dataset.")
    train.add_argument("--output-dir", default=None, help="Optional output directory.")

    interior = subparsers.add_parser("train-interior-pr", help="Train the interior PR two-stage pipeline.")
    interior.add_argument("--config", required=True, help="Path to a YAML config.")
    interior.add_argument("--data", required=True, help="Path to a CSV dataset.")
    interior.add_argument("--output-dir", default=None, help="Optional output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    if args.command == "train":
        train_single_target(config, args.data, args.output_dir)
    elif args.command == "train-interior-pr":
        train_interior_pr(config, args.data, args.output_dir)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
