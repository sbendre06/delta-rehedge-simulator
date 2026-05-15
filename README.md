# Friction-Aware Delta Hedging

A C++ simulation engine and Python analytics pipeline that finds the optimal rehedge threshold for an options market maker — balancing transaction costs against gamma exposure using real NASDAQ tick data and a calibrated stochastic volatility model.

---

## The Problem

Black-Scholes assumes continuous, costless hedging. In practice, every hedge crosses the bid-ask spread:

$$\text{Hedge Cost} = \Delta_{\text{shares}} \times \frac{\text{spread}}{2}$$

This creates a direct tension with the gamma P&L that motivates running the book in the first place:

$$\text{Gamma P\&L} \approx \frac{1}{2} \,\Gamma S^2 \left(\sigma_{\text{realized}}^2 - \sigma_{\text{implied}}^2\right)$$

Hedge too often and transaction costs dominate. Hedge too rarely and unhedged gamma exposure dominates. The optimal policy lives between these extremes and depends on three things that change continuously: gamma, current volatility, and the live bid-ask spread.

---

## Hedge Decision Rule

At each tick, the engine evaluates a single comparison:

$$\text{Hedge} \iff \underbrace{\frac{1}{2}\,\Gamma S^2 \sigma_t^2 \,\Delta t}_{\text{gamma risk from waiting}} > \underbrace{\Delta_{\text{shares}} \times \frac{\text{spread}}{2}}_{\text{cost of hedging now}}$$

The threshold is not static. $\sigma_t$ follows a calibrated Ornstein-Uhlenbeck process, so the rule responds dynamically to volatility regime changes — hedging more aggressively during vol spikes, less aggressively in calm periods.

---

## Stochastic Volatility Model

Realized volatility is modeled as an Ornstein-Uhlenbeck process:

$$d\sigma_t = \kappa(\theta - \sigma_t)\,dt + \xi\,dW_t$$

| Parameter | Meaning |
|-----------|---------|
| $\kappa$ | Mean reversion speed |
| $\theta$ | Long-run mean volatility |
| $\xi$ | Volatility of volatility |

Parameters are estimated via closed-form MLE on a historical realized vol time series (21-day rolling vol of SPY log returns). This replaces the constant-vol assumption with a dynamic, regime-aware input to the hedge threshold.

---

## Greeks

Delta and gamma are computed analytically from Black-Scholes at every tick:

$$\Delta = N(d_1), \qquad \Gamma = \frac{N'(d_1)}{S\,\sigma\sqrt{T}}$$

$$d_1 = \frac{\ln(S/K) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}}$$

where $N(\cdot)$ is the standard normal CDF, $N'(\cdot)$ is the standard normal PDF, and $S$, $K$, $T$, $r$, $\sigma$ are the standard Black-Scholes inputs. Both functions are implemented in C++ and called at every LOBSTER event.

---

## Order Flow Imbalance (OFI)

Beyond deciding *whether* to hedge, OFI determines *when* to execute. At each LOBSTER event, net order flow pressure is computed from changes in the best bid and ask queue sizes:

$$\text{OFI}_t = \sum_{i=t-w}^{t} \left[\mathbf{1}(\Delta \text{BidSize}_i > 0) - \mathbf{1}(\Delta \text{AskSize}_i < 0)\right]$$

over a rolling window $w$. When you need to sell shares and buyers are dominating (high OFI), you can often execute at mid or better rather than crossing the full spread. This timing signal reduces realized slippage without changing the hedge decision itself.

---

## Architecture

```
LOBSTER tick data (microsecond resolution)
        │
        ▼
┌───────────────────────────────┐
│   C++ Simulation Engine       │
│                               │
│  greeks.hpp   — Δ and Γ       │
│  simulator.cpp — hedge loop   │
│  main.cpp     — config/entry  │
│                               │
│  Output: hedge_log.csv        │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│   Python Analytics            │
│                               │
│  ou_calibration.py — fit OU   │
│  ofi_signal.py    — OFI calc  │
│  frontier.py      — P&L sweep │
│  analysis.ipynb   — results   │
└───────────────────────────────┘
```

The C++ engine handles 5–15M tick events per trading day and runs in seconds. The Python layer operates on the processed hedge log, which is small enough for pandas.

---

## Data

| Source | Usage |
|--------|-------|
| [LOBSTER](https://lobsterdata.com) | NASDAQ order book at microsecond resolution (top-of-book snapshots + message file) |
| SPY daily prices via `yfinance` | 21-day rolling realized vol for OU calibration |
| SPY options chain via `yfinance` | Implied vol and starting gamma for the simulated straddle |

LOBSTER files are gitignored. Download the free sample from [lobsterdata.com](https://lobsterdata.com) and place the message and orderbook CSVs in `data/raw/`.

---

## Getting Started

```bash
# Build the C++ engine
cd engine && make

# Run the simulation
./hedger --data ../data/raw/AAPL-Lobster-Orderbook.csv --output ../data/processed/hedge_log.csv

# Python analytics (from repo root)
pip install -r requirements.txt
jupyter notebook research/analysis.ipynb
```

---

## References

- Leland, H. E. (1985). *Option Pricing and Replication with Transactions Costs.* Journal of Finance, 40(5), 1283–1301.
- Garman, M. B., & Kohlhagen, S. W. (1983). *Foreign Currency Option Values.* Journal of International Money and Finance.
- LOBSTER: Limit Order Book System — The Efficient Reconstructor. Humboldt-Universität zu Berlin.
