"""
frontier.py
Sweep lambda values across the C++ engine to build the efficient P&L frontier.
Run from the research/ directory: python frontier.py
"""
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LAMBDA_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


def run_engine(lambda_val: float,
               ob_file: str,
               msg_file: str,
               engine_path: str = "../engine/hedger") -> dict:
    """
    Run the C++ hedger for one lambda value.
    Reads summary.csv and tick_log.csv written by the engine.
    Returns a dict of aggregated results.
    Raises RuntimeError if the engine exits with a non-zero return code.
    """
    result = subprocess.run(
        [engine_path, ob_file, msg_file, str(lambda_val)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Engine failed for lambda={lambda_val}:\n{result.stderr.strip()}"
        )

    summary = pd.read_csv("../data/processed/summary.csv")
    row = summary.iloc[0]

    tick = pd.read_csv("../data/processed/tick_log.csv",
                       usecols=["delta_gap_shares"])
    delta_gap_var = float(tick["delta_gap_shares"].var())

    return {
        "lambda_val":              float(lambda_val),
        "total_tcost":             float(row["total_tcost"]),
        "total_hedges":            int(row["total_hedges"]),
        "net_pnl":                 float(row["mtm_pnl"]),
        "greek_net_pnl":           float(row["greek_net_pnl"]),
        "total_gamma_pnl":         float(row["total_gamma_pnl"]),
        "total_theta_pnl":         float(row["total_theta_pnl"]),
        "delta_gap_variance":      delta_gap_var,
        "avg_spread":              float(row["avg_spread"]),
        "avg_time_between_hedges": float(row["avg_time_between_hedges"]),
    }


def run_frontier(
    ob_file:     str = "../data/raw/AAPL-Lobster-Orderbook.csv",
    msg_file:    str = "../data/raw/AAPL-Lobster-Message.csv",
    engine_path: str = "../engine/hedger",
) -> pd.DataFrame:
    """
    Run the engine for every lambda in LAMBDA_VALUES.
    Saves results to ../data/processed/frontier_results.csv.
    Returns a DataFrame with one row per lambda.
    """
    os.makedirs("../results/figures", exist_ok=True)
    print(f"Running frontier sweep over {len(LAMBDA_VALUES)} lambda values...")
    print(f"  Engine : {engine_path}")
    print(f"  Data   : {ob_file}\n")

    rows = []
    for lam in LAMBDA_VALUES:
        try:
            r = run_engine(lam, ob_file, msg_file, engine_path)
            rows.append(r)
            print(f"  Lambda {lam:5.2f}: {r['total_hedges']:4d} hedges, "
                  f"net P&L ${r['net_pnl']:+7.2f}, "
                  f"tcost ${r['total_tcost']:6.2f}, "
                  f"δ-gap var {r['delta_gap_variance']:.4f}")
        except RuntimeError as e:
            print(f"  WARNING: skipping lambda={lam} — {e}")

    df = pd.DataFrame(rows)
    out_path = "../data/processed/frontier_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFrontier results saved → {out_path}")
    return df


def plot_frontier(df: pd.DataFrame, save_fig: bool = True):
    """
    Three-panel figure: efficient frontier, net P&L vs lambda, P&L decomposition.
    Saves to ../results/figures/efficient_frontier.png.
    """
    optimal_idx = df["net_pnl"].idxmax()
    optimal_lam = df.loc[optimal_idx, "lambda_val"]
    optimal_pnl = df.loc[optimal_idx, "net_pnl"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # --- Panel 1: Efficient Frontier ---
    ax = axes[0]
    ax.plot(df["total_tcost"], df["delta_gap_variance"],
            "o-", color="steelblue", linewidth=1.5, markersize=7)
    for _, row in df.iterrows():
        ax.annotate(
            f"λ={row['lambda_val']:.2g}",
            xy=(row["total_tcost"], row["delta_gap_variance"]),
            xytext=(5, 3), textcoords="offset points",
            fontsize=8, color="dimgray",
        )
    ax.set_title("Efficient Hedging Frontier:\nTransaction Costs vs Hedging Error")
    ax.set_xlabel("Total Transaction Costs ($)")
    ax.set_ylabel("Variance of Delta Gap (shares²)")
    ax.annotate("← optimal region (bottom-left)",
                xy=(0.04, 0.06), xycoords="axes fraction",
                fontsize=8, color="darkgreen", style="italic")

    # --- Panel 2: Net P&L vs Lambda ---
    ax = axes[1]
    ax.plot(df["lambda_val"], df["net_pnl"], "o-", color="steelblue",
            linewidth=1.5, markersize=7)
    ax.axvline(optimal_lam, color="tomato", linestyle="--", linewidth=1.3,
               label=f"Optimal λ = {optimal_lam:.2g}\n(net P&L = ${optimal_pnl:.2f})")
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_xscale("log")
    ax.set_title("Net P&L vs Lambda")
    ax.set_xlabel("Lambda (log scale)")
    ax.set_ylabel("Net P&L ($)")
    ax.legend(fontsize=9)

    # --- Panel 3: P&L Decomposition stacked bar ---
    ax = axes[2]
    x = np.arange(len(df))
    theta_vals = df["total_theta_pnl"].values          # positive (income)
    gamma_vals = df["total_gamma_pnl"].values          # negative (loss)
    tcost_vals = -df["total_tcost"].values             # negate → shown as negative bar

    ax.bar(x, theta_vals, label="Theta income",      color="mediumseagreen", alpha=0.85)
    ax.bar(x, gamma_vals, label="Gamma loss",        color="tomato",         alpha=0.85)
    ax.bar(x, tcost_vals, label="Transaction costs", color="darkorange",     alpha=0.85,
           bottom=gamma_vals)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l:.2g}" for l in df["lambda_val"].values], rotation=45)
    ax.set_title("P&L Decomposition by Lambda")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("P&L ($)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    if save_fig:
        os.makedirs("../results/figures", exist_ok=True)
        path = "../results/figures/efficient_frontier.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    df = run_frontier()
    plot_frontier(df)
