"""
A/B analysis — did variant B actually earn a promotion?

Reads the API's telemetry log (impressions + purchases per variant), builds
the conversion contingency table, and runs a chi-square test of independence.
The decision rule is pre-registered here, not improvised after peeking:

    promote B iff  p < 0.05  AND  lift > 0
    (otherwise: keep A; a non-significant "win" is noise wearing a medal)

The offline context that motivates this experiment: the two-stage ranker
LOST the offline bake-off (80.0% vs CF's 84.9% hit-rate@10) but optimizes
margin-aware ordering that hit-rate can't see. Offline metrics pick the
champion; online experiments are how a challenger with a different objective
gets a fair hearing. That's the discipline this script encodes.

Usage:
    python experiments/ab_analysis.py                # analyze real telemetry
    python experiments/ab_analysis.py --simulate 5000  # SIMULATED traffic demo
"""

import argparse
import json
import random
from pathlib import Path

from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parent.parent
TELEMETRY = ROOT / "output" / "telemetry_events.jsonl"

ALPHA = 0.05


def load_counts(path: Path = TELEMETRY) -> dict:
    counts = {"A": {"impressions": 0, "purchases": 0},
              "B": {"impressions": 0, "purchases": 0}}
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        variant = event.get("variant")
        if variant not in counts:
            continue
        if event["event_type"] == "impression":
            counts[variant]["impressions"] += 1
        elif event["event_type"] == "purchase":
            counts[variant]["purchases"] += 1
    return counts


def analyze(counts: dict) -> dict:
    """Chi-square on the purchases-vs-no-purchase split across variants."""
    a, b = counts["A"], counts["B"]
    if min(a["impressions"], b["impressions"]) < 30:
        return {"decision": "insufficient_data",
                "reason": f"need >=30 impressions per variant, have "
                          f"A={a['impressions']}, B={b['impressions']}"}

    table = [[a["purchases"], a["impressions"] - a["purchases"]],
             [b["purchases"], b["impressions"] - b["purchases"]]]
    chi2, p_value, _, _ = chi2_contingency(table)
    cvr_a = a["purchases"] / a["impressions"]
    cvr_b = b["purchases"] / b["impressions"]
    lift = (cvr_b - cvr_a) / cvr_a if cvr_a else float("inf")

    if p_value < ALPHA and lift > 0:
        decision = "promote_B"
    elif p_value < ALPHA and lift < 0:
        decision = "retire_B"
    else:
        decision = "keep_A_keep_testing"
    return {"cvr_a": cvr_a, "cvr_b": cvr_b, "lift": lift,
            "chi2": chi2, "p_value": p_value, "decision": decision}


def simulate(n_per_variant: int, cvr_a: float = 0.08, cvr_b: float = 0.11,
             seed: int = 42) -> dict:
    """SIMULATION (clearly labeled as such): synthetic traffic with known true
    conversion rates, to demonstrate the framework detects a real difference.
    This demonstrates the statistics, not the challenger's actual merit —
    only real traffic can do that."""
    rng = random.Random(seed)
    counts = {"A": {"impressions": n_per_variant,
                    "purchases": sum(rng.random() < cvr_a for _ in range(n_per_variant))},
              "B": {"impressions": n_per_variant,
                    "purchases": sum(rng.random() < cvr_b for _ in range(n_per_variant))}}
    return counts


def report(counts: dict, label: str) -> dict:
    result = analyze(counts)
    print(f"== A/B analysis ({label}) ==")
    for v in ("A", "B"):
        c = counts[v]
        print(f"  {v}: {c['purchases']:,}/{c['impressions']:,} purchases")
    if result.get("decision") == "insufficient_data":
        print(f"  {result['reason']}")
        return result
    print(f"  CVR A {result['cvr_a']:.2%} | CVR B {result['cvr_b']:.2%} "
          f"| lift {result['lift']:+.1%}")
    print(f"  chi2 {result['chi2']:.2f}, p = {result['p_value']:.4f} "
          f"(alpha {ALPHA})")
    print(f"  decision: {result['decision']}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulate", type=int, metavar="N",
                    help="run on SIMULATED traffic (N impressions/variant) "
                         "instead of the real telemetry log")
    args = ap.parse_args()
    if args.simulate:
        report(simulate(args.simulate), f"SIMULATED, n={args.simulate}/variant")
    else:
        report(load_counts(), "telemetry_events.jsonl")


if __name__ == "__main__":
    main()
