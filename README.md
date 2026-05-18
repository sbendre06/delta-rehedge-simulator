# Hedging Tradeoffs: Slippage Costs vs Directional Risk in Options Market Making

A C++ simulation engine and Python analytics pipeline that investigates rehedging strategies for a short straddle market maker — balancing delta exposure against transaction costs using real NASDAQ tick data.

---

## Problem Setup

Black-Scholes assumes continuous, costless hedging. In practice every hedge crosses the bid-ask spread. A short straddle earns theta but loses on strong directional moves; every hedge to control that bleed costs money.

- **Hedge too often** → transaction costs dominate
- **Hedge too rarely** → unhedged delta risk and gamma exposure dominates

The engine finds the optimal point by sweeping a threshold parameter λ and measuring the full mark-to-market P&L at each level.

**Core hedge decision at every tick:**

$$\text{Hedge} \iff \underbrace{\frac{1}{2}\ \cdot |\Gamma|\ \cdot S^2\ \cdot \sigma^2\ \cdot \Delta t}_{\text{gamma risk from waiting}} > \lambda \cdot \underbrace{\Delta_{\text{shares}} \cdot \frac{\text{spread}}{2}}_{\text{cost of hedging now}}$$

---

## Repository Structure

```
├── engine/
│   ├── greeks.hpp          Black-Scholes greeks and option prices
│   ├── simulator.cpp       Core hedging loop, P&L tracking, CSV output
│   ├── main.cpp            CLI entry point
│   └── Makefile
│
├── research/
│   ├── ou_calibration.py   Fit OU process to historical AAPL realized vol
│   ├── ofi_signal.py       OFI execution quality analysis
│   ├── frontier.py         Efficient frontier
│   └── analysis.ipynb      Master results notebook
│
├── data/
│   ├── raw/                LOBSTER files (gitignored)
│   └── processed/          Engine output CSVs (gitignored)
│
└── results/
    └── figures/
```

---

## Data

| Source | What it provides |
|--------|-----------------|
| [LOBSTER](https://lobsterdata.com) | AAPL NASDAQ order book at microsecond resolution — top-of-book bid/ask prices, sizes, and a message file linking each snapshot to its causing event (new order, cancellation, execution). Free academic sample covers one full trading day. Place files in `data/raw/`. |
| `yfinance`: AAPL daily prices 2010–2012 | Used by `ou_calibration.py` to estimate OU parameters from historical realized volatility |

The simulation uses **June 21 2012** AAPL data. Simulation parameters match that date:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Strike | $585.00 | AAPL was trading ~$585 that day |
| Expiry | 30 days | Standard liquid expiry |
| Implied vol | 25% | Typical AAPL implied vol for that period |
| Risk-free rate | 1.5% | June 2012 Fed funds environment |
| Contract size | 100 shares | Standard US equity option |

---

## C++ Engine

### Build

```bash
cd engine
make
```

Requires `g++` with C++17. On macOS: `xcode-select --install`. On Linux: `sudo apt install g++`.

### Run

```bash
cd engine
./hedger <orderbook_file> <message_file> [lambda]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `orderbook_file` | Path to LOBSTER orderbook CSV |
| `message_file` | Path to LOBSTER message CSV |
| `lambda` | Hedge threshold multiplier (optional, default `1.0`) |

**Example:**

```bash
./hedger ../data/raw/AAPL-Lobster-Orderbook.csv \
         ../data/raw/AAPL-Lobster-Message.csv \
         1.0
```

Processes ~400k tick events in a few seconds and writes three files to `data/processed/`:

| Output file | Contents |
|-------------|----------|
| `tick_log.csv` | One row per valid market event — greeks, delta gap, P&L components, OFI, MTM P&L |
| `hedge_log.csv` | One row per hedge execution — timing, size, execution price, slippage, OFI |
| `summary.csv` | Single end-of-day row — total hedges, gamma P&L, theta P&L, transaction costs, MTM P&L |

### P&L Accounting

- **Mark-to-market** (`mtm_pnl`): `initial straddle value − current straddle value + hedge inventory value + cumulative hedge cash flows`. The primary figure. Slippage is embedded naturally because hedge trades execute at bid/ask rather than mid.

---

## Python Analytics

All scripts run from the `research/` directory.

### Dependencies

```bash
pip install pandas numpy matplotlib scipy yfinance
```

### ou_calibration.py

Fits an Ornstein-Uhlenbeck process to AAPL 21-day rolling realized volatility (2010–2012):

$$d\sigma_t = \kappa(\theta - \sigma_t) dt + \xi dW_t$$

Parameters estimated via OLS on the AR(1) discretisation. Produces `results/figures/ou_calibration.png`.

```bash
python ou_calibration.py
```

### ofi_signal.py

Classifies each hedge event as favorable or unfavorable based on whether rolling order flow imbalance aligned with the required trade direction. Runs a Welch t-test on slippage distributions and produces `results/figures/ofi_analysis.png`.

```bash
python ofi_signal.py
```

### frontier.py

Sweeps λ ∈ {0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0} by re-running the C++ engine for each value. Produces the efficient frontier (transaction costs vs. delta-gap variance), net P&L vs. λ, and a P&L decomposition chart.

```bash
python frontier.py
```

### analysis.ipynb

Shows combination of all previous results in one notebook.

```bash
jupyter notebook analysis.ipynb
```

---

## Key Findings & References

Read whitepaper for full results reporting and discussion.
