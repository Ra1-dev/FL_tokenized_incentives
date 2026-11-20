"""Run the predeclared pilot corpus through Flower with resumable validation."""

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

from fltoken.cache import load_update_rounds
from fltoken.config import load_config
from fltoken.io import sha256, write_json


def cells(plan):
    for seed, attack, fraction in itertools.product(plan["seeds"], plan["attacks"], plan["fractions"]):
        name=f"mnist-{attack}-f{round(fraction*100):02d}-s{seed}-{plan['device']}-t{plan['rounds']}"
        yield dict(seed=seed,attack=attack,fraction=fraction,path=f"results/pilots/{name}")


def training_sources(plan):
    paths = sorted(Path("src/fltoken").glob("*.py")) + [Path("scripts/run_experiment.py"), Path(plan["config"])]
    return {path.as_posix(): sha256(path) for path in paths}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,default=Path("configs/pilot_plan.json"))
    parser.add_argument("--no-run",action="store_true")
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text(encoding="utf-8"))
    committed=subprocess.check_output(["git","show",f"HEAD:{args.plan.as_posix()}"])
    if committed != args.plan.read_bytes():
        raise ValueError("pilot plan must be committed before execution")
    paths=["src","configs","scripts/run_experiment.py","scripts/run_pilots.py"]
    if subprocess.check_output(["git","status","--porcelain","--",*paths],text=True).strip():
        raise ValueError("commit the training and scoring sources before starting the pilot corpus")
    base=load_config(plan["config"])
    if base["rounds"] != plan["rounds"] or base["train_limit"] or base["test_limit"]:
        raise ValueError("pilot plan requires the full recorded training protocol")
    sources = training_sources(plan)
    completed=[]
    for cell in cells(plan):
        if training_sources(plan) != sources:
            raise ValueError("training sources changed during the pilot corpus")
        folder=Path(cell["path"])
        expected={**base,**{key:cell[key] for key in ["seed","attack","fraction"]},
                  **{key:plan[key] for key in ["device","workers","gpu_fraction"]}}
        metadata_path=folder/"metadata.json"
        existing=json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if existing.get("status") != "complete":
            if args.no_run:
                raise RuntimeError(f"missing complete pilot cell: {folder}")
            subprocess.run([sys.executable,"scripts/run_experiment.py","--config",plan["config"],
                            "--seed",str(cell["seed"]),"--attack",cell["attack"],"--fraction",str(cell["fraction"]),
                            "--device",plan["device"],"--workers",str(plan["workers"]),
                            "--gpu-fraction",str(plan["gpu_fraction"]),"--output",str(folder)],check=True)
            existing=json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing["config"] != expected:
            raise ValueError(f"existing pilot configuration differs: {folder}")
        if existing["source_git_dirty"] or existing["source_sha256"] != sources or training_sources(plan) != sources:
            raise ValueError(f"pilot training source differs from the committed corpus: {folder}")
        checked=sum(1 for _ in load_update_rounds(folder))
        if checked != plan["rounds"]:
            raise ValueError("pilot cache is incomplete")
        completed.append(dict(**cell,rounds=checked,per_round_sha256=sha256(folder/"per_round.csv"),
                              wall_seconds=existing["wall_seconds"],source_commit=existing["git_commit"]))
        write_json("results/pilots/manifest.json",dict(plan_sha256=sha256(args.plan),
                   training_source_sha256=sources, expected_cells=sum(1 for _ in cells(plan)),completed_cells=len(completed),cells=completed))
        print(f"Validated pilot cells: {len(completed)}",flush=True)


if __name__ == "__main__":
    main()
