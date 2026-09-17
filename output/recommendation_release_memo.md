# Recommendation release memo — REC-REL-2026-001

## Decision

**CONDITIONALLY APPROVE PILOT** for a synthetic-data rep-assist pilot. The collaborative-filtering champion remains simpler and stronger on the defined warm-customer holdout. This authorizes controlled evaluation, not automated outreach and not a revenue-lift claim.

## Objective hierarchy and release evidence

- **Recall@10:** 0.849 — PASS (warm-customer basket holdout)
- **Lift over popularity:** 1.126 — PASS (same holdout and denominator)
- **Catalog coverage:** 1.000 — PASS (share of catalog served at least once)
- **Mean novelty (bits):** 5.260 — PASS (self-information under purchase-volume popularity)
- **Intra-list diversity:** 0.716 — PASS (share of slate pairs from different protein families)
- **Protein calibration error:** 0.670 — REVIEW (lower is closer to the customer's historical mix)
- **Eligibility pass rate:** 1.000 — PASS (catalog, ownership, reason and opportunity controls)

## Weakest reportable slices

- **history_density / sparse warm:** recall@10 67.6% across 182 hidden items
- **protein / Seafood:** recall@10 70.2% across 151 hidden items
- **rfm_segment / Hibernating:** recall@10 70.6% across 102 hidden items

These slices are diagnostic, not proof of fairness or causal impact. New customers are excluded from the warm-customer holdout and use the separately labelled regional-popularity fallback.

## Operating controls

- Every served item must exist in the catalog, be unowned, carry a recommendation reason and retain a positive indicative opportunity.
- The service target is p95 <= 150 ms, with p50/p95/p99 recorded in `serving_slo.json`.
- Degraded mode serves an explicitly labelled regional-popularity default and still excludes owned products.
- Every recommendation is human-reviewed before a sales representative acts.

## Experiment before promotion

Run `REC-PILOT-2026-001` with sticky customer-level assignment, a sample-ratio-mismatch check, one pre-registered final read and commercial, trust and latency guardrails. Promotion requires positive purchase-conversion lift at p < 0.05, every guardrail passing and no material slice harm.

## Evidence boundary

All data and traffic in this repository are synthetic. Offline metrics measure warm-customer basket recovery, not incremental sales.

Packet SHA-256: `ad00965d8337600dcc0dbc9eedb5da7ecdec6ba666a25d2773d24cba8103f93b`
