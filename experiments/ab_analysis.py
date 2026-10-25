"""
A/B analysis — did variant B actually earn a promotion?

Reads the API's telemetry at its customer randomization grain. Repeated requests
and purchases cannot create extra independent observations. Only a purchase of
an item on a preceding personalized slate counts; degraded fallbacks are excluded.
The binary customer-conversion diagnostic uses chi-square, or Fisher's exact
test for sparse/constant outcomes, after an assignment-ratio check.

This is a diagnostic, not the execution of REC-PILOT-2026-001. Its per-customer
outcome is separate from the plan's per-impression metric. Maturity, power,
commercial/trust/latency guardrails and slice review still require a final read.
No result authorizes model promotion or establishes commercial lift.

The offline context that motivates this experiment: the two-stage ranker
LOST the offline bake-off (81.2% vs CF's 84.9% recall@10) but optimizes
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

from scipy.stats import chi2_contingency, chisquare, fisher_exact

ROOT = Path(__file__).resolve().parent.parent
TELEMETRY = ROOT / "output" / "telemetry_events.jsonl"

ALPHA = 0.05


def load_counts(path: Path = TELEMETRY) -> dict:
    """One observation per exposed customer, preserving event order."""
    counts = {"A": {"impressions": 0, "purchases": 0},
              "B": {"impressions": 0, "purchases": 0}}
    if not path.exists():
        return counts
    assignments: dict[str, str] = {}
    exposed: dict[str, set[str]] = {"A": set(), "B": set()}
    converted: dict[str, set[str]] = {"A": set(), "B": set()}
    active_skus: dict[str, set[str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError(f"Telemetry line {line_number} must be an object")
        variant = event.get("variant")
        event_type = event.get("event_type")
        if variant not in counts or event_type not in {"impression", "purchase"}:
            continue
        customer = event.get("customer_id")
        if not isinstance(customer, str) or not customer.strip():
            raise ValueError(f"Telemetry line {line_number} needs a customer_id")
        if event_type == "impression":
            skus = event.get("skus")
            if not isinstance(skus, list) or any(not isinstance(s, str) for s in skus):
                raise ValueError(f"Telemetry line {line_number} needs a list of SKUs")
            active_skus[customer] = set()
            if event.get("serving_mode", "personalized_batch") != "personalized_batch" or not skus:
                continue
            if assignments.setdefault(customer, variant) != variant:
                raise ValueError("A customer changed experiment variants; investigate assignment")
            exposed[variant].add(customer)
            active_skus[customer] = set(skus)
        elif (assignments.get(customer) == variant
              and event.get("sku") in active_skus.get(customer, set())):
            converted[variant].add(customer)
    for variant in counts:
        counts[variant] = {"impressions": len(exposed[variant]), "purchases": len(converted[variant])}
    return counts


def analyze(counts: dict) -> dict:
    """Binary conversion diagnostic; callers must supply independent units."""
    try:
        a, b = counts["A"], counts["B"]
        values = [arm[key] for arm in (a, b) for key in ("impressions", "purchases")]
    except (KeyError, TypeError) as error:
        raise ValueError("Both variants need impression and purchase counts") from error
    if any(type(value) is not int or value < 0 or value > 2**53 - 1 for value in values):
        raise ValueError("Counts must be bounded nonnegative integers")
    if any(arm["purchases"] > arm["impressions"] for arm in (a, b)):
        raise ValueError("Converted independent units cannot exceed exposed units")
    boundary = {"promotion_authorized": False, "unit": "unique exposed customer",
                "evidence_boundary": "Binary customer diagnostic only; the registered pilot and all guardrails still require review."}
    if min(a["impressions"], b["impressions"]) < 30:
        return {**boundary, "decision": "insufficient_data",
                "reason": f"need >=30 independent exposed customers per variant, have "
                          f"A={a['impressions']}, B={b['impressions']}"}

    srm_p = float(chisquare([a["impressions"], b["impressions"]]).pvalue)
    if srm_p < .01:
        return {**boundary, "decision": "assignment_quality_review", "srm_p_value": srm_p,
                "reason": "Assignment ratio differs from the planned 50/50 split; investigate before reading outcomes."}

    table = [[a["purchases"], a["impressions"] - a["purchases"]],
             [b["purchases"], b["impressions"] - b["purchases"]]]
    total = a["impressions"] + b["impressions"]
    successes = a["purchases"] + b["purchases"]
    expected = [n * outcome / total for n in (a["impressions"], b["impressions"])
                for outcome in (successes, total - successes)]
    if min(expected) < 5:
        _, p_value = fisher_exact(table)
        chi2, method = None, "Fisher exact (two-sided)"
    else:
        chi2, p_value, _, _ = chi2_contingency(table)
        method = "Chi-square with continuity correction"
    cvr_a = a["purchases"] / a["impressions"]
    cvr_b = b["purchases"] / b["impressions"]
    lift = (cvr_b - cvr_a) / cvr_a if cvr_a else (0.0 if cvr_b == 0 else None)

    if p_value < ALPHA and cvr_b > cvr_a:
        decision = "review_B"
    elif p_value < ALPHA and cvr_b < cvr_a:
        decision = "review_harm_in_B"
    else:
        decision = "keep_A_keep_testing"
    return {**boundary, "cvr_a": cvr_a, "cvr_b": cvr_b, "lift": lift,
            "chi2": chi2, "p_value": float(p_value), "method": method,
            "srm_p_value": srm_p, "decision": decision}


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
    if "reason" in result:
        print(f"  {result['reason']}")
        return result
    lift = "undefined (zero control conversion)" if result["lift"] is None else f"{result['lift']:+.1%}"
    print(f"  CVR A {result['cvr_a']:.2%} | CVR B {result['cvr_b']:.2%} | lift {lift}")
    print(f"  {result['method']}, p = {result['p_value']:.4f} "
          f"(alpha {ALPHA})")
    print(f"  decision: {result['decision']}")
    print(f"  {result['evidence_boundary']}")
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
