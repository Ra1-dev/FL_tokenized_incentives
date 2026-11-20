"""Run pinned offline Solidity tests and retain classified static findings."""

import argparse
from collections import Counter
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

from replay_onchain import build_contracts, digest, read, write
from contract_evidence import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/contracts/checks.json"))
    args = parser.parse_args()
    measured_sources = provenance("scripts/check_contracts.py")
    measured_sources.update({p.as_posix():digest(p) for p in sorted(Path("contracts/test").glob("*.sol"))})
    build_contracts()
    binaries = Path("out/tools/foundry").resolve()
    suffix = ".exe" if os.name == "nt" else ""
    env = {**os.environ, "PATH": str(binaries)+os.pathsep+os.environ.get("PATH", "")}
    forge = binaries/("forge"+suffix)
    seed = "0x8b9cf4e3"
    result = subprocess.run([str(forge), "test", "--root", "contracts", "--offline", "--json", "--fuzz-seed", seed],
                            env=env, capture_output=True, text=True, encoding="utf-8")
    Path("out/contract-tests.json").write_text(result.stdout, encoding="utf-8", newline="\n")
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    suites = read("out/contract-tests.json")
    tests = {suite+"/"+name: value for suite, body in suites.items() for name, value in body["test_results"].items()}
    if not tests or any(test["status"] != "Success" for test in tests.values()):
        raise AssertionError("contract tests failed")
    raw = Path("out/contract-static.json")
    raw.unlink(missing_ok=True)
    # Compile directly: this avoids dependence on a compiler-artifact adapter.
    command = [sys.executable, "-m", "slither", "contracts/src/RewardPool.sol", "--compile-force-framework", "solc",
               "--solc", str(binaries/("solc"+suffix)), "--solc-remaps",
               "@openzeppelin/contracts/=third_party/openzeppelin-contracts/contracts/",
               "--solc-args", "--optimize --optimize-runs 200 --evm-version cancun", "--json", str(raw)]
    static = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8")
    Path("out/contract-static.log").write_text(static.stdout+static.stderr, encoding="utf-8", newline="\n")
    findings = read(raw)
    if not findings["success"]:
        raise RuntimeError("static analysis did not complete; see out/contract-static.log")
    detectors = findings["results"].get("detectors", [])
    counts = Counter(d["impact"] for d in detectors)
    if counts["High"] or counts["Medium"]:
        raise AssertionError(f"unresolved high/medium findings: {dict(counts)}")
    rationale = {
        "timestamp": "Round deadlines intentionally use block timestamps. Boundary tests cover equality; small proposer clock variation remains possible.",
        "pragma": "First-party sources pin the compiler. Vendored sources keep their original supported ranges; the build pins the executable and settings.",
        "solc-version": "Vendored sources keep their original supported compiler ranges. Compilation uses the checksummed pinned compiler.",
        "dead-code": "Unused context helpers are retained unchanged with the vendored token library.",
        "cyclomatic-complexity": "Settlement has explicit maturity, slash and mint phases. Stateful independent-ledger comparisons exercise their ordering.",
    }
    classified = []
    for detector in detectors:
        if detector["check"] not in rationale:
            raise AssertionError(f"unclassified static finding: {detector['check']}")
        sources = sorted({e["source_mapping"]["filename_relative"].replace("\\", "/") for e in detector["elements"]})
        classified.append(dict(check=detector["check"], impact=detector["impact"], confidence=detector["confidence"],
                               sources=sources, rationale=rationale[detector["check"]]))
    current_sources = provenance("scripts/check_contracts.py")
    current_sources.update({p.as_posix():digest(p) for p in sorted(Path("contracts/test").glob("*.sol"))})
    if measured_sources != current_sources:
        raise ValueError("contract sources changed during checks")
    summary = dict(tests_passed=len(tests), tests_failed=0, fuzz_seed=seed, provenance=measured_sources,
                   fuzz_runs=sum(t["kind"].get("Fuzz", {}).get("runs", 0) for t in tests.values()),
                   tests=sorted(tests), static_counts={level:counts[level] for level in ["High", "Medium", "Low", "Informational", "Optimization"]},
                   findings=classified, forge_version=subprocess.check_output([str(forge), "--version"], text=True).strip(),
                   compiler_version=subprocess.check_output([str(binaries/("solc"+suffix)), "--version"], text=True).strip(),
                   static_version=importlib.metadata.version("slither-analyzer"),
                   source_sha256={p.as_posix():digest(p) for p in sorted(Path("contracts/src").glob("*.sol"))})
    write(args.output, summary)
    print(f"tests={len(tests)}; fuzz_runs={summary['fuzz_runs']}; static={summary['static_counts']}")


if __name__ == "__main__":
    main()
