"""One result summary feeds every numerical paper macro."""

import json
from pathlib import Path

from fltoken.config import load_config
from fltoken.io import write_json
from fltoken.models import LeNet5


def main():
    path = Path("results/summary.json")
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    source = json.loads(Path("results/literature/sendros.json").read_text(encoding="utf-8"))
    config = load_config("configs/sendros_mix.yaml")
    summary["protocol"] = {**config, "parameters": sum(p.numel() for p in LeNet5().parameters())}
    summary["published"] = source
    write_json(path, summary)
    p = summary["protocol"]
    values = {"NumClients": p["num_clients"], "SampledClients": p["clients_per_round"], "NumRounds": p["rounds"],
              "LocalEpochs": p["local_epochs"], "BatchSize": p["batch_size"], "LearningRate": p["lr"],
              "Momentum": p["momentum"], "DirichletAlpha": p["alpha"], "NumParameters": f"{p['parameters']:,}",
              "NoiseSigma": p["noise_sigma"], "StaleLag": p["stale_lag"],
              "PublishedShare": f"{100*source['standard']['attacker_share_mean']:.2f}",
              "PublishedShareSD": f"{100*source['standard']['attacker_share_sd']:.2f}"}
    r = summary.get("reproduction")
    if r:
        values.update(MeasuredShare=f"{100*r['attacker_share_mean']:.2f}",
                      MeasuredShareSD=f"{100*r['attacker_share_sample_sd']:.2f}",
                      BiasMedianError=f"{100*r['relative_error_quantiles']['0.5']:.2f}",
                      BiasMaxError=f"{100*r['relative_error_quantiles']['1']:.2f}",
                      NoiseParticipations=r["noise_participations"])
    lines = ["% Generated from results/summary.json. Do not edit numerical values here."]
    lines += [f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in values.items()]
    folder = Path("paper/generated")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Generated {len(values)} numerical macros")


if __name__ == "__main__":
    main()
