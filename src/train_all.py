from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from rbf_benchmark.config import project_root, read_config, workspace_path
from rbf_benchmark.training import run_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train every dataset selected in config.yaml.")
    parser.add_argument("--config", type=Path, default=project_root() / "config.yaml")
    args = parser.parse_args()
    config = read_config(args.config)
    run_dir = workspace_path(config, "output_dir") / datetime.now().strftime("rbf_ridge_%Y%m%d_%H%M%S_%f")
    results = [run_dataset(name, config, run_dir) for name in config["experiment"]["datasets"]]
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
