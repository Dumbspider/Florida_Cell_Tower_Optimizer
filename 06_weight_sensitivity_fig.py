"""
06_weight_sensitivity_fig.py  —  visualize how county priority ranks shift
as the gap/equity weighting changes. Produces a bump chart + a stability table.

Run next to acs_features.csv and coverage_by_county.csv.
Outputs: weight_sensitivity_bump.png, weight_stability.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def minmax(s):
    return (s - s.min()) / (s.max() - s.min())


def priority(df, w_gap):
    d = df.copy()
    d["unserved_people"] = d["coverage_gap"] * d["Population"]
    gap = minmax(d["unserved_people"])
    need = minmax(minmax(d["Poverty Rate"]) + minmax(-d["Median Household Income"]) + minmax(d["Median Age"]))
    return w_gap * gap + (1 - w_gap) * need


def main():
    acs = pd.read_csv(HERE / "acs_features.csv")
    cov = pd.read_csv(HERE / "coverage_by_county.csv")
    df = acs.merge(cov, on="GEO_ID", how="inner")
    df["name"] = df["NAME"].str.replace(" County, FL", "", regex=False).str.replace(r",.*$", "", regex=True)

    # sweep the gap weight from 0.3 to 0.9
    weights = [0.3, 0.5, 0.7, 0.9]
    ranks = pd.DataFrame({"name": df["name"]})
    for w in weights:
        r = priority(df, w).rank(ascending=False, method="first").astype(int)
        ranks[w] = r.values

    # who ever appears in the top 10 under any weighting?
    top_names = set()
    for w in weights:
        top_names |= set(ranks.sort_values(w).head(10)["name"])

    # --- bump chart ---
    fig, ax = plt.subplots(figsize=(9, 8))
    for _, row in ranks.iterrows():
        highlight = row["name"] in top_names
        ax.plot(range(len(weights)), [row[w] for w in weights],
                marker="o", linewidth=2.2 if highlight else 0.6,
                alpha=1.0 if highlight else 0.15,
                color=None if highlight else "gray", zorder=3 if highlight else 1)
        if highlight:
            ax.annotate(row["name"], (0, row[0.3]), xytext=(-8, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=8)
            ax.annotate(f"{row[0.9]}. {row['name']}", (len(weights) - 1, row[0.9]),
                        xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=8)

    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels([f"{w:g} / {round(1-w,1):g}\n(gap / need)" for w in weights])
    ax.set_ylabel("Priority rank  (1 = highest need)")
    ax.set_title("Rank stability across gap/equity weightings")
    ax.invert_yaxis()                      # rank 1 at the top
    ax.set_ylim(20.5, 0.5)                 # show top ~20; drop this line to show all 67
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(HERE / "weight_sensitivity_bump.png", dpi=200)

    # --- stability table: rank range for the counties that top the list ---
    tbl = ranks[ranks["name"].isin(top_names)].copy()
    tbl["best_rank"] = tbl[weights].min(axis=1)
    tbl["worst_rank"] = tbl[weights].max(axis=1)
    tbl["rank_swing"] = tbl["worst_rank"] - tbl["best_rank"]
    tbl = tbl.sort_values("best_rank")
    tbl.to_csv(HERE / "weight_stability.csv", index=False)

    print("wrote weight_sensitivity_bump.png and weight_stability.csv")
    print("\nRank swing for top-10 counties (0 = perfectly stable):")
    print(tbl[["name", "best_rank", "worst_rank", "rank_swing"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
