"""Compare all predeclared reward candidates on a complete pilot corpus."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from fltoken.cache import load_update_rounds
from fltoken.config import load_config
from fltoken.io import sha256, write_json
from fltoken.metrics import ROUND_POOL, payout, summarize
from fltoken.scoring import OursScorer
from run_pilots import cells


def candidates(plan):
    selection = plan["selection"]
    for ema, penalties in itertools.product(selection["ema"], selection["penalties"]):
        betas = selection["beta"] if ema else [0.]
        thresholds = selection["threshold"] if penalties else [0.]
        patience_values = selection["patience"] if penalties else [max(selection["patience"])]
        for beta, threshold, patience in itertools.product(betas, thresholds, patience_values):
            yield dict(reference=True, ema=ema, penalties=penalties, beta=beta,
                       threshold=threshold, patience=patience,
                       initial_reputation=selection["initial_reputation"])


def tie_key(params):
    return (params["threshold"], -params["patience"], params["beta"])


def objective(metrics, plan):
    attacked = [m for m in metrics if m["attack"] != "none"]
    if not attacked or not metrics:
        raise ValueError("selection requires attack-present and honest-cost observations")
    cost = plan["selection"]["undefined_attacker_share_cost"]
    shares = [m["attacker_share"] if m["attacker_share"] is not None else cost for m in attacked]
    if any(m["honest_false_penalty_rate"] is None for m in metrics):
        raise ValueError("pilot cell has no honest participations")
    attacker = math.fsum(shares)/len(shares)
    honest = math.fsum(m["honest_false_penalty_rate"] for m in metrics)/len(metrics)
    return dict(objective=attacker+honest, mean_attacker_share_cost=attacker,
                mean_honest_false_penalty_rate=honest,
                undefined_attacker_cells=sum(m["attacker_share"] is None for m in attacked),
                empty_rounds=sum(m["empty_rounds"] for m in metrics),
                honest_clients_slashed_sum=sum(m["honest_ever_slashed"] for m in metrics))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expected_attacks(config):
    """Independently check the seeded identity assignment in pilot records."""
    order=np.random.default_rng(config["seed"]).permutation(config["num_clients"])
    if config["attack"]=="none":
        return {}
    if config["attack"]=="sendros_mix":
        return dict(zip(map(int,order[:5]),["zero","zero","noise","noise","stale"]))
    count=round(config["fraction"]*config["num_clients"])
    return {int(i):config["attack"] for i in order[:count]}


def geometry(folder, output, recorded_only):
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(participation_sha256=sha256(folder/"per_round.csv"),
                      geometry_source_sha256=sha256("src/fltoken/ours.py"),
                      cache_source_sha256=sha256("src/fltoken/cache.py"))
    if not recorded_only:
        scorer=OursScorer({"ema":False,"penalties":False})
        rows=[]
        for round, ids, deltas, original in load_update_rounds(folder):
            quality=scorer.score(ids,deltas,round)["scores"]
            rows.extend(dict(round=round,client_id=int(client),quality=float(q),
                             update_set_sha256=row["update_set_sha256"]) for client,q,row in zip(ids,quality,original))
        with (output/"quality.csv").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=["round","client_id","quality","update_set_sha256"],lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        write_json(output/"provenance.json",dict(**provenance,quality_sha256=sha256(output/"quality.csv"),numpy_version=np.__version__))
    saved=read(output/"provenance.json")
    if any(saved.get(k)!=v for k,v in provenance.items()) or saved["quality_sha256"]!=sha256(output/"quality.csv"):
        raise ValueError("pilot geometry provenance differs")
    with (output/"quality.csv").open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream)), saved


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,default=Path("configs/pilot_plan.json"))
    parser.add_argument("--recorded-only",action="store_true",help="recompute candidate allocations from committed pilot qualities")
    parser.add_argument("--freeze",action="store_true")
    args=parser.parse_args()
    plan=read(args.plan)
    if plan["selection"]["ema"] != [True] or plan["selection"]["penalties"] != [True]:
        raise ValueError("pilot selection compares the fully enabled method; ablations share its parameters")
    prior=read("configs/method_pilot.json")
    for left,right in [("seeds","pilot_seeds"),("attacks","attacks"),("fractions","fractions")]:
        if plan[left]!=prior[right]:
            raise ValueError("pilot coverage differs from the recorded design prior")
    for left,right in [("beta","ema_decay_candidates"),("threshold","low_quality_candidates"),("patience","patience_candidates")]:
        if plan["selection"][left]!=prior[right]:
            raise ValueError("pilot candidates differ from the recorded design prior")
    if plan["selection"]["initial_reputation"]!=prior["ema_initial"]:
        raise ValueError("initial reputation differs from the recorded design prior")
    frozen_path=Path("configs/method_frozen.json")
    binding=dict(pilot_plan_sha256=sha256(args.plan),design_prior_sha256=sha256("configs/method_pilot.json"),
                 scorer_sha256=sha256("src/fltoken/ours.py"),selection_program_sha256=sha256(__file__),
                 metrics_source_sha256=sha256("src/fltoken/metrics.py"))
    if frozen_path.exists() and any(read(frozen_path).get(key)!=value for key,value in binding.items()):
        raise ValueError("scientific inputs differ from the frozen method; preserve its evidence")
    if subprocess.check_output(["git","show",f"HEAD:{args.plan.as_posix()}"])!=args.plan.read_bytes():
        raise ValueError("pilot selection plan must be committed")
    if subprocess.check_output(["git","status","--porcelain","--","src","configs","scripts/select_method.py"],text=True).strip():
        raise ValueError("commit scoring sources and the selection program before pilot selection")
    manifest=read("results/pilots/manifest.json")
    sources = manifest["training_source_sha256"]
    verified_commits = set()
    planned=list(cells(plan))
    if manifest["plan_sha256"]!=sha256(args.plan) or manifest["completed_cells"]!=len(planned):
        raise ValueError("complete predeclared pilot corpus required")
    if [{k:c[k] for k in planned[0]} for c in manifest["cells"]] != planned:
        raise ValueError("pilot coverage differs from the declared cells")
    params=list(candidates(plan))
    measurements=[[] for _ in params]
    evidence=[]
    for cell, recorded in zip(planned,manifest["cells"]):
        folder=Path(cell["path"])
        metadata=read(folder/"metadata.json")
        if metadata["status"]!="complete" or sha256(folder/"per_round.csv")!=recorded["per_round_sha256"]:
            raise ValueError("pilot training provenance differs")
        if metadata["source_git_dirty"] or metadata["source_sha256"] != sources:
            raise ValueError("pilot run used a different or uncommitted scientific implementation")
        commit = metadata["git_commit"]
        if commit != recorded["source_commit"]:
            raise ValueError("pilot source commit differs from its manifest")
        if commit not in verified_commits:
            for name, expected_hash in sources.items():
                original = subprocess.check_output(["git", "show", f"{commit}:{name}"])
                if hashlib.sha256(original).hexdigest() != expected_hash:
                    raise ValueError(f"pilot source differs from its recorded commit: {name}")
            verified_commits.add(commit)
        expected={**load_config(plan["config"]),**{key:cell[key] for key in ["seed","attack","fraction"]},
                  **{key:plan[key] for key in ["device","workers","gpu_fraction"]}}
        if metadata["config"]!=expected:
            raise ValueError("pilot scientific configuration differs")
        quality, provenance=geometry(folder,Path("results/pilot_selection/geometry")/folder.name,args.recorded_only)
        with (folder/"per_round.csv").open(encoding="utf-8",newline="") as stream:
            source=list(csv.DictReader(stream))
        if len(source)!=len(quality) or any((a["round"],a["client_id"],a["update_set_sha256"])!=(b["round"],b["client_id"],b["update_set_sha256"]) for a,b in zip(source,quality)):
            raise ValueError("pilot qualities do not follow recorded participants")
        attacks={int(k):v for k,v in read(folder/"protocol.json")["adversaries"].items()}
        if attacks!=expected_attacks(metadata["config"]) or any(row["attack"]!=attacks.get(int(row["client_id"]),"none") for row in source):
            raise ValueError("pilot attack identities differ from the seeded assignment")
        for index, parameter in enumerate(params):
            scorer=OursScorer(parameter)
            rows, ledger=[],[]
            for round in range(1,metadata["config"]["rounds"]+1):
                values=[(a,b) for a,b in zip(source,quality) if int(a["round"])==round]
                ids=[int(a["client_id"]) for a,b in values]
                scores=scorer.allocate(ids,[float(b["quality"]) for a,b in values],round)
                units, amounts=payout(scores["shares"])
                ledger.append(dict(round=round,ids=ids,shares1e18=units,payouts=amounts,
                                   slash=scores["slash"].tolist(),remainder=ROUND_POOL-sum(amounts)))
                rows.extend(dict(round=round,client_id=client,adversary=client in attacks,
                                 accuracy=float(a["accuracy"]),Mstar_share=float(share),Mstar_slash=bool(slash))
                            for client,(a,b),share,slash in zip(ids,values,scores["shares"],scores["slash"]))
            metrics=summarize(rows,{"Mstar":ledger},metadata["config"],attacks)["Mstar"]
            measurements[index].append(dict(**cell,**metrics))
        evidence.append(dict(**cell,**provenance, metadata_sha256=sha256(folder/"metadata.json"),
                             protocol_sha256=sha256(folder/"protocol.json")))
        print(f"Scored {folder.name} across {len(params)} candidates",flush=True)
    candidates_summary=[]
    for index,(parameter,metrics) in enumerate(zip(params,measurements)):
        result=dict(candidate=index,params=parameter,**objective(metrics,plan),cells=metrics)
        write_json(f"results/pilot_selection/candidate-{index:02d}.json",result)
        candidates_summary.append({k:v for k,v in result.items() if k!="cells"})
    values=[c["objective"] for c in candidates_summary]
    tolerance=plan["selection"]["tie_tolerance"]
    if max(values)-min(values)<=tolerance:
        raise RuntimeError("pilot measurement does not distinguish candidates; harden it before freezing")
    tied=[c for c in candidates_summary if c["objective"]<=min(values)+tolerance]
    winner=min(tied,key=lambda c:tie_key(c["params"]))
    corpus_digest=hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    result=dict(candidates=candidates_summary,selected=winner,corpus_sha256=corpus_digest,
                plan_sha256=sha256(args.plan),cells=len(planned),source_sha256=sha256("src/fltoken/ours.py"),
                selection_source_sha256=sha256(__file__),design_prior_sha256=binding["design_prior_sha256"])
    write_json("results/pilot_selection/summary.json",result)
    if args.freeze:
        target=frozen_path
        frozen=dict(params=winner["params"],**binding,corpus_sha256=corpus_digest,
                    selection_summary_sha256=sha256("results/pilot_selection/summary.json"),
                    settlement=dict(vesting_rounds=prior["vesting_rounds"] if winner["params"]["penalties"] else 0,
                                    slash_numerator=prior["slash_numerator"],slash_denominator=prior["slash_denominator"],
                                    stake_tokens=prior["stake_tokens"]))
        if target.exists() and read(target)!=frozen:
            raise ValueError("a different frozen method already exists; preserve its provenance")
        write_json(target,frozen)
    print(json.dumps(winner,indent=2))


if __name__ == "__main__":
    main()
