"""
ou_calibration.py
Ornstein-Uhlenbeck process applied AAPL historical realized, validate assumed vol in C++ sim
run from research/ directory: python ou_calibration.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf


def calibrate_ou(save_fig: bool = True) -> dict:
    """
    fit OU process to AAPL 21-day rolling realized vol (2010-01-01 to 2012-06-21)
    returns dict with keys: kappa, theta, xi
    """
    print("Downloading AAPL daily prices (2010-01-01 → 2012-06-21)...")
    raw = yf.download("AAPL", start="2010-01-01", end="2012-06-22",
                      auto_adjust=True, progress=False)
    prices = raw["Close"].squeeze()
    print(f"  {len(prices)} daily observations downloaded")

    # Daily log returns and 21-day rolling realized vol (annualised)
    log_ret = np.log(prices / prices.shift(1))
    realized_vol = log_ret.rolling(21).std() * np.sqrt(252)
    realized_vol = realized_vol.dropna()
    print(f"  {len(realized_vol)} realized-vol observations after dropping NaN")
    print(f"  Vol range: [{realized_vol.min():.4f}, {realized_vol.max():.4f}]")

    # AR(1) OLS regression: sigma_{t+1} = a + b * sigma_t + eps
    sigma_t  = realized_vol.values[:-1]
    sigma_t1 = realized_vol.values[1:]
    reg = stats.linregress(sigma_t, sigma_t1)
    b, a = reg.slope, reg.intercept
    residuals = sigma_t1 - (a + b * sigma_t)

    # OU parameter extraction from AR(1) coefficients
    dt    = 1.0 / 252.0
    kappa = -np.log(b) / dt
    theta = a / (1.0 - b)
    xi    = (np.std(residuals, ddof=1)
             * np.sqrt(2.0 * kappa / (1.0 - np.exp(-2.0 * kappa * dt))))

    print("\n=== Ornstein-Uhlenbeck Calibration Results ===")
    print(f"  κ (kappa) = {kappa:8.4f}  — mean reversion speed "
          f"(half-life = {np.log(2)/kappa*252:.1f} trading days)")
    print(f"  θ (theta) = {theta:8.4f}  — long-run mean volatility  ({theta*100:.1f}%)")
    print(f"  ξ (xi)    = {xi:8.4f}  — vol of vol")
    print(f"  R²        = {reg.rvalue**2:.4f}")

    if save_fig:
        os.makedirs("../results/figures", exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # Left: realized vol time series with long-run mean
        ax = axes[0]
        ax.plot(realized_vol.index, realized_vol.values,
                color="steelblue", linewidth=0.8, label="21-day realized vol")
        ax.axhline(theta, color="tomato", linestyle="--", linewidth=1.5,
                   label=f"θ = {theta:.3f}  ({theta*100:.1f}%)")
        ax.set_title("AAPL Realized Volatility (2010–2012)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Annualised Volatility")
        ax.legend()

        # Right: autocorrelation of realized vol
        ax = axes[1]
        n_lags = 40
        acf = [pd.Series(realized_vol.values).autocorr(lag=k) for k in range(1, n_lags + 1)]
        ax.bar(range(1, n_lags + 1), acf, color="steelblue", alpha=0.75)
        ax.axhline(0, color="black", linewidth=0.8)
        conf = 1.96 / np.sqrt(len(realized_vol))
        ax.axhline( conf, color="tomato", linestyle="--", linewidth=1.0, label="95% CI")
        ax.axhline(-conf, color="tomato", linestyle="--", linewidth=1.0)
        ax.set_title("Autocorrelation of Realized Volatility")
        ax.set_xlabel("Lag (trading days)")
        ax.set_ylabel("Autocorrelation")
        ax.legend()

        fig.tight_layout()
        path = "../results/figures/ou_calibration.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved → {path}")
        plt.close(fig)

    return {"kappa": kappa, "theta": theta, "xi": xi}


if __name__ == "__main__":
    calibrate_ou()
