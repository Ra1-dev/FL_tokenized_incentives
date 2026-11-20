"""Download datasets once and record cryptographic hashes of every raw file."""

import argparse

from fltoken.datasets import DATASETS, prepare_dataset
from fltoken.io import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="mnist")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    names = DATASETS if args.dataset == "all" else [args.dataset]
    for name in names:
        record = prepare_dataset(name, args.data_dir)
        write_json(f"results/datasets/{name}.json", record)
        print(f"{name}: verified {len(record['sha256'])} raw files")


if __name__ == "__main__":
    main()
