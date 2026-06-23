from __future__ import annotations

import argparse
from pathlib import Path

from rbf_benchmark.config import DATASETS, project_root, read_config, workspace_path
from rbf_benchmark.data import write_preprocessed_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and fully prepare datasets for RBF training.")
    parser.add_argument("--config", type=Path, default=project_root() / "config.yaml")
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()
    config = read_config(args.config)
    data_dir = workspace_path(config, "data_dir")
    names = [args.dataset] if args.dataset else config.get("download", {}).get("datasets", [])
    for name in names:
        train, test = write_preprocessed_dataset(name, data_dir, config)
        print(f"{name}: {train} | {test}")


if __name__ == "__main__":
    main()
