import importlib
import json
import csv
import hashlib
from pathlib import Path
import shutil
import sys

import numpy as np

import pytest


def selection(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/"scripts"))
    return importlib.import_module("select_method"),json.loads((root/"configs/pilot_plan.json").read_text())


def test_candidate_space_has_no_inactive_parameter_duplicates(monkeypatch):
    module,plan=selection(monkeypatch)
    candidates=list(module.candidates(plan))
    assert len(candidates)==18
    assert len({json.dumps(c,sort_keys=True) for c in candidates})==18
    assert all(c["reference"] for c in candidates)
    assert all(c["ema"] and c["penalties"] for c in candidates)


def test_objective_keeps_empty_allocations_and_honest_only_cells(monkeypatch):
    module,plan=selection(monkeypatch)
    base=dict(empty_rounds=0,honest_ever_slashed=0)
    metrics=[dict(**base,attack="noise",attacker_share=.2,honest_false_penalty_rate=.1),
             dict(**base,attack="zero",attacker_share=None,honest_false_penalty_rate=1.),
             dict(**base,attack="none",attacker_share=0.,honest_false_penalty_rate=.4)]
    result=module.objective(metrics,plan)
    assert result["mean_attacker_share_cost"]==pytest.approx(.6)
    assert result["mean_honest_false_penalty_rate"]==pytest.approx(.5)
    assert result["objective"]==pytest.approx(1.1)
    assert result["undefined_attacker_cells"]==1


def test_ties_prefer_the_declared_simpler_choice(monkeypatch):
    module,plan=selection(monkeypatch)
    candidates=list(module.candidates(plan))
    chosen=min(candidates,key=module.tie_key)
    assert chosen["ema"] and chosen["penalties"]
    assert chosen["threshold"]==0 and chosen["patience"]==3 and chosen["beta"]==0


def test_selection_and_recorded_replay_match_on_synthetic_corpus(tmp_path,monkeypatch):
    module,plan=selection(monkeypatch)
    root=Path(__file__).resolve().parents[1]
    plan.update(seeds=[137,139],attacks=["none","noise"],fractions=[.3],rounds=4)
    for folder in ["configs","src/fltoken","results/pilots","scripts"]:
        (tmp_path/folder).mkdir(parents=True)
    for name in ["ours.py","cache.py","metrics.py"]:
        shutil.copyfile(root/"src/fltoken"/name,tmp_path/"src/fltoken"/name)
    for name in ["run_experiment.py","evaluation.py","select_method.py"]:
        shutil.copyfile(root/"scripts"/name,tmp_path/"scripts"/name)
    prior=json.loads((root/"configs/method_pilot.json").read_text())
    prior.update(pilot_seeds=plan["seeds"],attacks=plan["attacks"],fractions=plan["fractions"])
    for name,value in [("pilot_plan.json",plan),("method_pilot.json",prior)]:
        (tmp_path/"configs"/name).write_text(json.dumps(value),encoding="utf-8")
    (tmp_path/plan["config"]).write_text("num_clients: 10\nclients_per_round: 10\nrounds: 4\n",encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    sources = importlib.import_module("run_pilots").training_sources(plan)
    def git_output(command,**kwargs):
        if command[1]=="show":
            return Path(command[2].split(":",1)[1]).read_bytes()
        if command[1]=="status":
            return ""
        raise AssertionError(command)
    monkeypatch.setattr(module.subprocess,"check_output",git_output)
    completed=[]
    for cell in module.cells(plan):
        folder=Path(cell["path"])
        folder.mkdir(parents=True)
        cache=Path("results/cache")/folder.name
        cache.mkdir(parents=True)
        config={**module.load_config(plan["config"]),**{k:cell[k] for k in ["seed","attack","fraction"]},
                **{k:plan[k] for k in ["device","workers","gpu_fraction"]}}
        attacks=module.expected_attacks(config)
        rng=np.random.default_rng(cell["seed"])
        rows=[]
        for round in range(1,5):
            ids=np.arange(10)
            x=rng.normal(.5,1.,size=(10,23))
            for client in attacks:
                x[client]=rng.normal(0.,1.,23)
            digest=hashlib.sha256(x.astype("<f8").tobytes()).hexdigest()
            np.savez(cache/f"round-{round:03d}.npz",client_ids=ids,deltas=x)
            rows.extend(dict(round=round,client_id=int(client),attack=attacks.get(int(client),"none"),
                             update_set_sha256=digest,accuracy=.8) for client in ids)
        with (folder/"per_round.csv").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
            writer.writeheader();writer.writerows(rows)
        (folder/"round_metrics.csv").write_text("round,accuracy\n",encoding="utf-8")
        module.write_json(folder/"ledger.json",{})
        module.write_json(folder/"hashes.json",{name:module.sha256(folder/name) for name in ["per_round.csv","round_metrics.csv","ledger.json"]})
        module.write_json(folder/"metadata.json",dict(status="complete",config=config,synthetic_fixture=True,
                                                    source_git_dirty=False,source_sha256=sources,git_commit="e"*40))
        module.write_json(folder/"protocol.json",dict(adversaries=attacks))
        completed.append(dict(**cell,per_round_sha256=module.sha256(folder/"per_round.csv"),source_commit="e"*40))
    module.write_json("results/pilots/manifest.json",dict(plan_sha256=module.sha256("configs/pilot_plan.json"),
                                                       training_source_sha256=sources,completed_cells=4,cells=completed))
    monkeypatch.setattr(sys,"argv",["select_method.py","--freeze"])
    module.main()
    first=Path("results/pilot_selection/summary.json").read_bytes()
    assert module.read("configs/method_frozen.json")["corpus_sha256"]==module.read("results/pilot_selection/summary.json")["corpus_sha256"]
    monkeypatch.setattr(sys,"argv",["select_method.py","--recorded-only"])
    module.main()
    assert Path("results/pilot_selection/summary.json").read_bytes()==first
    documentation = importlib.import_module("document_method")
    monkeypatch.setattr(documentation, "recorded_method", lambda: module.read("configs/method_frozen.json"))
    record = documentation.frozen_selection()
    assert record["selection"]["cells"] == 4
    note = documentation.selection_note(record)
    assert "pilot selection measurements" in note
    assert record["configuration"]["corpus_sha256"] in note
    Path("results/pilot_selection/summary.json").write_bytes(first + b"\n")
    with pytest.raises(ValueError, match="frozen method evidence differs"):
        documentation.frozen_selection()
