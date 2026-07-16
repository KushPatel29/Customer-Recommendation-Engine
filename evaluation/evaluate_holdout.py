"""
Holdout evaluation: does the collaborative filter actually work?

Protocol (per eligible customer, one with >= 8 distinct SKUs):
  1. Hide 25% of their SKUs (all purchase lines for those SKUs removed).
  2. Rebuild the customer x SKU matrix on the remaining data only.
  3. Ask the engine for top-10 cross-sell recommendations.
  4. Score hit-rate@10: how many hidden SKUs the engine re-discovered.

Compared against a popularity baseline (recommend the globally best-selling
SKUs the customer doesn't own). If CF can't beat popularity, ship
popularity — same discipline as the demand-forecast backtest in the
supply-chain repo.

Usage:
    python evaluation/evaluate_holdout.py
"""

import random
import sys
from pathlib import Path

import pandas as pd
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from recommend import build_customer_sku_matrix, cross_sell_recommendations, load_sales

random.seed(42)

MIN_SKUS = 8
HOLDOUT_FRAC = 0.25
K = 10


def popularity_baseline(train: pd.DataFrame, owned: set, k=K) -> list:
    pop = (train.groupby("sku")["quantity_lb"].sum()
           .sort_values(ascending=False).index)
    return [s for s in pop if s not in owned][:k]


def svd_recommendations(matrix: pd.DataFrame, cust: str, owned: set,
                        k=K, n_factors=12) -> list:
    """Latent-factor model: TruncatedSVD reconstructs the customer x SKU
    matrix from n_factors dimensions; the reconstruction scores unpurchased
    SKUs (classic matrix-factorization recommender)."""
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    latent = svd.fit_transform(matrix.values)
    recon = latent @ svd.components_
    scores = pd.Series(recon[matrix.index.get_loc(cust)], index=matrix.columns)
    return [s for s in scores.sort_values(ascending=False).index
            if s not in owned][:k]


N_CANDIDATES = 30   # stage-1 pool handed to the stage-2 ranker


def evaluate(sales: pd.DataFrame) -> pd.DataFrame:
    sku_counts = sales.groupby("customer_id")["sku"].nunique()
    eligible = sku_counts[sku_counts >= MIN_SKUS].index.tolist()

    rows = []
    candidate_pools = {}   # per customer: stage-1 pool with labels, for the ranker
    for cust in eligible:
        cust_skus = sorted(sales.loc[sales["customer_id"] == cust, "sku"].unique())
        n_hide = max(2, int(len(cust_skus) * HOLDOUT_FRAC))
        hidden = set(random.sample(cust_skus, n_hide))

        train = sales[~((sales["customer_id"] == cust) & (sales["sku"].isin(hidden)))]
        owned = set(train.loc[train["customer_id"] == cust, "sku"])
        matrix = build_customer_sku_matrix(train)

        # one retrieval pass at candidate depth; CF@10 is simply its head
        pool = cross_sell_recommendations(train, matrix=matrix, n_recs=N_CANDIDATES)
        pool = pool[pool["customer_id"] == cust].sort_values("rank")
        cf_recs = set(pool.head(K)["sku"])
        svd_recs = set(svd_recommendations(matrix, cust, owned))
        pop_recs = set(popularity_baseline(train, owned))

        labeled = pool[["customer_id", "sku", "score"]].copy()
        labeled["label"] = labeled["sku"].isin(hidden).astype(int)
        candidate_pools[cust] = labeled

        rows.append({
            "customer_id": cust,
            "hidden": len(hidden),
            "cf_hits": len(cf_recs & hidden),
            "svd_hits": len(svd_recs & hidden),
            "pop_hits": len(pop_recs & hidden),
            "cf_recs": ";".join(sorted(cf_recs)),
        })
    results = pd.DataFrame(rows)
    results["ts_hits"] = two_stage_hits(candidate_pools, results)
    return results


def two_stage_hits(candidate_pools: dict, results: pd.DataFrame) -> pd.Series:
    """Score the two-stage system (retrieval -> learned ranker) with
    customer-disjoint training: rank half A's candidates with a model trained
    only on half B's labels, and vice versa — zero label leakage."""
    from engine.ranker import rerank, train_ranker

    products = pd.read_csv(ROOT / "output" / "product_analytics.csv")
    customers = pd.read_csv(ROOT / "output" / "customer_analytics.csv")

    custs = sorted(candidate_pools)
    halves = (set(custs[0::2]), set(custs[1::2]))
    hits = {}
    for train_half, score_half in ((halves[0], halves[1]), (halves[1], halves[0])):
        train_pairs = pd.concat([candidate_pools[c] for c in sorted(train_half)],
                                ignore_index=True)
        model, cols = train_ranker(train_pairs, products, customers)
        for cust in score_half:
            pool = candidate_pools[cust]
            ranked = rerank(model, cols, pool, products, customers, n_final=K)
            hidden_skus = set(pool.loc[pool["label"] == 1, "sku"])
            hits[cust] = len(set(ranked["sku"]) & hidden_skus)
    return results["customer_id"].map(hits)


def log_to_mlflow(metrics_by_model: dict, n_customers: int) -> None:
    """Track the bake-off in MLflow (local SQLite store, mlflow.db).

    One run per model within an experiment, so `mlflow ui` shows the
    CF / SVD / popularity comparison side by side and every re-run of the
    pipeline appends a new evaluation generation. Kept import-optional: the
    core pipeline must not require the serving/MLOps extras.

    Inspect with:  mlflow ui --backend-store-uri sqlite:///mlflow.db
    """
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed - skipping experiment tracking "
              "(pip install -r requirements-api.txt)")
        return

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("cross-sell-bakeoff")
    shared = {"k": K, "holdout_frac": HOLDOUT_FRAC, "min_skus": MIN_SKUS,
              "n_customers": n_customers}
    for model_name, m in metrics_by_model.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params({**shared, **m.get("params", {})})
            mlflow.log_metrics({k: v for k, v in m.items() if k != "params"})
    print(f"mlflow: logged {len(metrics_by_model)} runs -> {ROOT / 'mlflow.db'} "
          "(inspect with `mlflow ui --backend-store-uri sqlite:///mlflow.db`)")


def main():
    sales = load_sales()
    results = evaluate(sales)
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    results.to_csv(out / "holdout_evaluation.csv", index=False)

    cf = results["cf_hits"].sum() / results["hidden"].sum()
    svd = results["svd_hits"].sum() / results["hidden"].sum()
    pop = results["pop_hits"].sum() / results["hidden"].sum()
    ts = results["ts_hits"].sum() / results["hidden"].sum()
    # beyond-accuracy: what share of the catalog does each method ever surface?
    n_skus = sales["sku"].nunique()
    cf_coverage = len(set(";".join(results["cf_recs"]).split(";"))) / n_skus
    print(f"customers evaluated: {len(results)}")
    print(f"hit-rate@{K}  two-stage (CF -> ranker): {ts:.1%}")
    print(f"hit-rate@{K}  collaborative filtering: {cf:.1%}")
    print(f"hit-rate@{K}  SVD latent factors:      {svd:.1%}")
    print(f"hit-rate@{K}  popularity baseline:     {pop:.1%}")
    print(f"CF lift over popularity: {cf / pop:.2f}x" if pop else "n/a")
    print(f"CF catalog coverage: {cf_coverage:.0%} of SKUs surfaced "
          f"(popularity by construction surfaces ~{K / n_skus:.0%})")

    log_to_mlflow({
        "item-based-cf": {"hit_rate_at_k": cf, "catalog_coverage": cf_coverage,
                          "lift_over_popularity": cf / pop if pop else 0.0,
                          "params": {"n_similar": 8, "damping": "log1p"}},
        "svd-latent-factors": {"hit_rate_at_k": svd,
                               "params": {"n_factors": 12}},
        "two-stage-cf-ranker": {"hit_rate_at_k": ts,
                                "params": {"n_candidates": 30,
                                           "ranker": "hist-gradient-boosting",
                                           "training": "customer-disjoint 2-fold"}},
        "popularity-baseline": {"hit_rate_at_k": pop,
                                "catalog_coverage": K / n_skus,
                                "params": {"strategy": "global-top-sellers"}},
    }, n_customers=len(results))


if __name__ == "__main__":
    main()
