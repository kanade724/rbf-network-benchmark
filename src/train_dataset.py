from __future__ import annotations

import argparse
from pathlib import Path

from rbf_benchmark.config import DATASETS, project_root, read_config
from rbf_benchmark.training import run_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one preprocessed dataset with the shared GPU/CPU RBF ridge classifier.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--config", type=Path, default=project_root() / "config.yaml")
    args = parser.parse_args()
    run_dataset(args.dataset, read_config(args.config))


if __name__ == "__main__":
    main()
