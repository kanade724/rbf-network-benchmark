"""Download and cache datasets required by the project."""

import argparse
from pathlib import Path

from datasets import SKLEARN_LOADERS, load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download project datasets into the local data directory.")
    parser.add_argument("--dataset", choices=[*SKLEARN_LOADERS, "pendigits", "fashion_mnist", "all"], default="all")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    names = [*SKLEARN_LOADERS, "pendigits", "fashion_mnist"] if args.dataset == "all" else [args.dataset]
    for name in names:
        print(f"Preparing {name}...")
        load_dataset(name, args.data_dir, random_state=42, train_size=10_000 if name == "fashion_mnist" else None)
    print("Datasets are ready.")


if __name__ == "__main__":
    main()
