"""
ofi_signal.py
does order flow imbalance at hedge execution predicts slippage quality?
run from research/ directory: python ofi_signal.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def analyze_ofi(hedge_log_path: str = "../data/processed/hedge_log.csv",
                save_fig: bool = True) -> dict:
    """
    classify hedge events by OFI direction and compare execution slippage:

    Sign convention in hedge_log:
      shares_traded < 0  →  engine bought shares (paid the ask)
      shares_traded > 0  →  engine sold  shares (received the bid)

    Favorable OFI:
      Buying  + OFI < 0  →  sellers dominating, better prices for buys
      Selling + OFI > 0  →  buyers  dominating, better prices for sells

    returns dict with group stats, t-test results, and pct_improvement
    """
    print(f"Loading hedge log from {hedge_log_path}...")
    df = pd.read_csv(hedge_log_path)
    print(f"  {len(df)} total hedge events")

    df = df[df["rolling_ofi_50"] != 0].copy()
    print(f"  {len(df)} after excluding zero-OFI rows")

    buying  = df["shares_traded"] < 0
    selling = df["shares_traded"] > 0
    df["favorable_ofi"] = ((buying  & (df["rolling_ofi_50"] < 0)) |
                           (selling & (df["rolling_ofi_50"] > 0)))

    def group_stats(subdf: pd.DataFrame) -> dict:
        return {
            "count":  len(subdf),
            "mean":   subdf["slippage_per_share"].mean(),
            "median": subdf["slippage_per_share"].median(),
            "std":    subdf["slippage_per_share"].std(),
            "total":  subdf["total_slippage"].sum(),
        }

    fav_df   = df[df["favorable_ofi"]]
    unfav_df = df[~df["favorable_ofi"]]
    fav_s    = group_stats(fav_df)
    unfav_s  = group_stats(unfav_df)

    t_stat, p_value = stats.ttest_ind(
        fav_df["slippage_per_share"].values,
        unfav_df["slippage_per_share"].values,
        equal_var=False,
    )
    pct_improv = (
        (unfav_s["mean"] - fav_s["mean"]) / unfav_s["mean"] * 100
        if unfav_s["mean"] != 0 else 0.0
    )

    print("\n=== OFI Execution Analysis ===")
    print(f"{'':24s} {'Favorable':>12s} {'Unfavorable':>12s}")
    print("─" * 50)
    print(f"{'Count':24s} {fav_s['count']:>12d} {unfav_s['count']:>12d}")
    print(f"{'Mean slip/share ($)':24s} {fav_s['mean']:>12.4f} {unfav_s['mean']:>12.4f}")
    print(f"{'Median slip/share ($)':24s} {fav_s['median']:>12.4f} {unfav_s['median']:>12.4f}")
    print(f"{'Total slippage ($)':24s} {fav_s['total']:>12.4f} {unfav_s['total']:>12.4f}")
    print(f"\n  t-stat = {t_stat:.3f},  p = {p_value:.4f}  "
          f"({'significant' if p_value < 0.05 else 'not significant'} at α = 0.05)")
    print(f"  Mean slippage improvement with favorable OFI: {pct_improv:+.1f}%")

    if save_fig:
        os.makedirs("../results/figures", exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # Left: bar chart of mean slippage per group
        ax = axes[0]
        groups = ["Favorable OFI", "Unfavorable OFI"]
        means  = [fav_s["mean"],   unfav_s["mean"]]
        errs   = [fav_s["std"],    unfav_s["std"]]
        colors = ["steelblue", "tomato"]
        ax.bar(groups, means, yerr=errs, capsize=6,
               color=colors, alpha=0.85, edgecolor="white", width=0.5)
        ax.set_title("Mean Slippage per Share by OFI Direction")
        ax.set_ylabel("Slippage per Share ($)")
        ax.set_xlabel("OFI Classification")
        y_top = max(m + e for m, e in zip(means, errs)) * 1.25
        ax.set_ylim(0, y_top)
        ax.annotate(f"{pct_improv:.1f}% improvement",
                    xy=(0.5, y_top * 0.82), xycoords=("axes fraction", "data"),
                    ha="center", fontsize=10, color="darkgreen", fontweight="bold")
        ax.annotate(f"p = {p_value:.3f}{'*' if p_value < 0.05 else ''}",
                    xy=(0.5, y_top * 0.68), xycoords=("axes fraction", "data"),
                    ha="center", fontsize=9, color="dimgray")

        # Right: scatter OFI vs slippage, coloured by trade direction
        ax = axes[1]
        buys  = df[df["shares_traded"] < 0]
        sells = df[df["shares_traded"] > 0]
        ax.scatter(buys["rolling_ofi_50"],  buys["slippage_per_share"],
                   alpha=0.45, s=18, color="steelblue",
                   label="Buy hedge (shares_traded < 0)")
        ax.scatter(sells["rolling_ofi_50"], sells["slippage_per_share"],
                   alpha=0.45, s=18, color="tomato",
                   label="Sell hedge (shares_traded > 0)")
        mean_slip = df["slippage_per_share"].mean()
        ax.axhline(mean_slip, color="black", linestyle="--", linewidth=1.0,
                   label=f"Overall mean = ${mean_slip:.4f}")
        ax.set_title("OFI vs Slippage per Share")
        ax.set_xlabel("Rolling OFI (50-event window)")
        ax.set_ylabel("Slippage per Share ($)")
        ax.legend(fontsize=8, markerscale=1.5)

        fig.tight_layout()
        path = "../results/figures/ofi_analysis.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved → {path}")
        plt.close(fig)

    return {
        "favorable":       fav_s,
        "unfavorable":     unfav_s,
        "t_statistic":     t_stat,
        "p_value":         p_value,
        "pct_improvement": pct_improv,
    }


if __name__ == "__main__":
    analyze_ofi()
