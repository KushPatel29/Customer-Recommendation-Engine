"""
README visuals — generated from the actual model, not mockups.

1. persona_map.png        2-D SVD projection of the customer x SKU matrix,
                          colored by ground-truth persona. The model never
                          sees personas; clusters emerging is the visual
                          proof the similarity space is real.
2. similarity_heatmap.png customer-customer cosine similarity, rows sorted
                          by persona — block structure = segments recovered.
3. evaluation_chart.png   holdout hit-rate@10: CF vs SVD vs popularity.

Usage:
    python analytics/make_visuals.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "evaluation"))

from evaluate_holdout import evaluate
from recommend import build_customer_sku_matrix, customer_similarity, load_sales

DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

NAVY, TEAL, ORANGE, PLUM = "#12436D", "#28A197", "#F46A25", "#801650"
PERSONA_COLORS = {"steakhouse": NAVY, "grocery": TEAL,
                  "sushi": ORANGE, "charcuterie": PLUM}


def main():
    sales = load_sales()
    personas = pd.read_csv(ROOT / "data" / "customers.csv").set_index("customer_id")["persona"]
    matrix = build_customer_sku_matrix(sales)

    # ---- 1. persona map
    # L2-normalize rows first: basket *shape* (not order volume) drives
    # position, matching the cosine geometry the recommender actually uses.
    normed = matrix.values / np.linalg.norm(matrix.values, axis=1, keepdims=True)
    svd = TruncatedSVD(n_components=3, random_state=42)
    xyz = svd.fit_transform(normed)
    xy = xyz[:, 1:3]  # drop factor 1 (shared "buys everything a bit" direction)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for persona, color in PERSONA_COLORS.items():
        mask = personas.reindex(matrix.index).values == persona
        ax.scatter(xy[mask, 0], xy[mask, 1], c=color, s=42, alpha=0.85,
                   edgecolors="white", linewidths=0.6, label=persona)
    ax.set_title("Customers in similarity space — colored by ground-truth persona\n"
                 "(the model never sees personas; the clusters are learned from purchases alone)",
                 fontsize=10.5, fontweight="bold", color=NAVY, loc="left")
    ax.set_xlabel("latent factor 1")
    ax.set_ylabel("latent factor 2")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "persona_map.png", dpi=130)

    # ---- 2. similarity heatmap, persona-sorted
    order = personas.reindex(matrix.index).sort_values().index
    sim = customer_similarity(matrix).loc[order, order]
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    im = ax.imshow(sim.values, cmap="Blues", vmin=0, vmax=1)
    # persona boundary lines
    sorted_personas = personas.reindex(order).values
    boundaries = np.flatnonzero(sorted_personas[:-1] != sorted_personas[1:]) + 0.5
    for b in boundaries:
        ax.axhline(b, color=ORANGE, lw=1.1)
        ax.axvline(b, color=ORANGE, lw=1.1)
    ax.set_title("Customer-customer cosine similarity, sorted by persona\n"
                 "(block structure = buyer segments recovered from purchase history)",
                 fontsize=10.5, fontweight="bold", color=NAVY, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(DOCS / "similarity_heatmap.png", dpi=130)

    # ---- 3. evaluation chart
    results = evaluate(sales)
    hidden = results["hidden"].sum()
    rates = {
        "Collaborative\nfiltering": results["cf_hits"].sum() / hidden,
        "SVD latent\nfactors": results["svd_hits"].sum() / hidden,
        "Popularity\nbaseline": results["pop_hits"].sum() / hidden,
    }
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(rates.keys(), rates.values(), color=[NAVY, TEAL, "#9AA5B1"], width=0.55)
    for bar, v in zip(bars, rates.values(), strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.1%}",
                ha="center", fontsize=11, fontweight="bold", color=NAVY)
    ax.set_ylim(0, 1)
    ax.set_ylabel("hit-rate@10 on held-out SKUs")
    ax.set_title("Holdout evaluation: 25% of each customer's SKUs hidden,\n"
                 "how many does each recommender re-discover in its top 10?",
                 fontsize=10.5, fontweight="bold", color=NAVY, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "evaluation_chart.png", dpi=130)

    print(f"wrote persona_map.png, similarity_heatmap.png, evaluation_chart.png -> {DOCS}")


if __name__ == "__main__":
    main()
