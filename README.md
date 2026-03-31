# Portfolio Risk System — VaR & CVaR

A full institutional-style market risk pipeline for an 8-asset, $1M portfolio. Covers data ingestion, three VaR methods, CVaR (Expected Shortfall), statistical backtesting, and stress testing across historical crises — assembled into a single risk dashboard.

---

## Portfolio

| Asset | Ticker | Role |
|---|---|---|
| S&P 500 ETF | SPY | US Large Cap Equity |
| Nasdaq-100 ETF | QQQ | US Tech Equity |
| Intl Developed ETF | EFA | International Equity |
| Gold ETF | GLD | Safe Haven / Inflation Hedge |
| Long Treasury ETF | TLT | Duration / Flight-to-Safety |
| Investment Grade Bond ETF | LQD | Fixed Income |
| ExxonMobil | XOM | Energy / Commodity Exposure |
| Real Estate ETF | VNQ | Real Estate |

**Equal-weighted (12.5% each) | $1,000,000 portfolio | 2015–2024 (2,514 trading days)**

---

## Pipeline

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
 Data      VaR       CVaR     Backtest   Stress    Dashboard
```

Each phase saves results to `portfolio_data.pkl` which is loaded by the next phase.

---

## Results

### Phase 1 — Portfolio Construction

Diversification confirmed: portfolio annualised volatility (11.7%) is lower than every individual asset.

| Asset | Ann. Return | Ann. Vol | Sharpe |
|---|---|---|---|
| SPY | 12.29% | 17.68% | 0.695 |
| QQQ | 16.93% | 21.88% | 0.774 |
| EFA | 5.12% | 17.38% | 0.295 |
| GLD | 7.48% | 14.13% | 0.530 |
| TLT | -1.20% | 15.31% | -0.078 |
| LQD | 2.23% | 8.60% | 0.259 |
| XOM | 5.67% | 27.79% | 0.204 |
| VNQ | 4.71% | 21.02% | 0.224 |
| **Portfolio** | **6.66%** | **11.71%** | **0.568** |

Portfolio excess kurtosis = **16.99** — fat tails are present and the normality assumption is violated. This motivates using Historical Simulation and CVaR over parametric methods.

---

### Phase 2 — Value at Risk (Three Methods)

| Method | 95% VaR | 99% VaR | 95% VaR ($) | 99% VaR ($) |
|---|---|---|---|---|
| Historical Simulation | 1.061% | 1.889% | $10,610 | $18,886 |
| Parametric | 1.213% | 1.716% | $12,132 | $17,159 |
| Monte Carlo | 1.189% | 1.701% | $11,891 | $17,007 |

At 99%, Parametric VaR is **lower** than Historical Simulation — the normal distribution underestimates tail severity. This gap is model risk.

---

### Phase 3 — CVaR / Expected Shortfall

CVaR answers the question VaR cannot: *given we are already having a bad day beyond VaR, how bad is it on average?*

| Method | 95% VaR ($) | 95% CVaR ($) | 99% VaR ($) | 99% CVaR ($) |
|---|---|---|---|---|
| Historical Simulation | $10,610 | $17,442 | $18,886 | **$31,791** |
| Parametric | $12,132 | $15,215 | $17,159 | $19,659 |
| Monte Carlo | $11,891 | $15,127 | $17,007 | $19,787 |

Historical CVaR/VaR ratio at 99% = **1.68x** vs the theoretical normal ratio of 1.16x. The empirical tail is 45% fatter than normality assumes. Basel III replaced VaR with CVaR (Expected Shortfall) as the regulatory standard for this exact reason.

---

### Phase 4 — Backtesting (Kupiec + Christoffersen)

Rolling 252-day out-of-sample VaR forecasts tested against actual returns.

**99% VaR Backtest Results**

| Method | Exceptions | Expected | Actual Rate | Kupiec | Christoffersen |
|---|---|---|---|---|---|
| Historical Sim | 35 | 22.6 | 1.55% | FAIL | FAIL |
| Parametric | 50 | 22.6 | 2.21% | FAIL | FAIL |
| Monte Carlo | 49 | 22.6 | 2.17% | FAIL | FAIL |

**95% VaR Backtest Results**

| Method | Exceptions | Expected | Actual Rate | Kupiec | Christoffersen |
|---|---|---|---|---|---|
| Historical Sim | 123 | 113.1 | 5.44% | PASS | FAIL |
| Parametric | 106 | 113.1 | 4.69% | PASS | FAIL |
| Monte Carlo | 105 | 113.1 | 4.64% | PASS | FAIL |

All models pass Kupiec at 95% (correct frequency) but fail Christoffersen at both levels (exceptions cluster). For Historical Simulation at 99%: π11 = **14.3%** vs π01 = **1.35%** — after a VaR breach, another breach the next day is 10x more likely. Static models have no memory of volatility regimes.

---

### Phase 5 — Stress Testing

Historical crises and a hypothetical liquidity shock applied to the portfolio, compared against VaR/CVaR thresholds.

| Scenario | Worst Single Day | Dollar Loss | vs 99% VaR |
|---|---|---|---|
| GFC 2008–09 (simulated) | 0.23% avg/day | $288,750 total | — |
| COVID Crash (Feb–Mar 2020) | **7.81%** | **$78,102** | **4.14x** |
| Rate Shock 2022 | 3.75% | $37,526 | 1.99x |
| Hypothetical Liquidity Shock | 10.38% | $103,750 | 5.49x |
| 99% VaR (reference) | 1.89% | $18,886 | 1.00x |
| 99% CVaR (reference) | 3.18% | $31,791 | 1.68x |

COVID's worst single day was **4x the 99% VaR**. The 2022 rate shock broke the stock-bond negative correlation embedded in the covariance matrix — both TLT and LQD fell alongside equities, invalidating the diversification assumption.

---

### Phase 6 — Risk Dashboard

All results assembled into a single publication-quality figure (`risk_dashboard.png`) with 6 panels: cumulative return and drawdown, VaR/CVaR summary table, rolling VaR vs actual returns, exception timeline, stress test bar chart, and backtesting results table.

---

## Setup

```bash
git clone <repo-url>
cd "VaR CVaR"

python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Running

Run notebooks in order — each phase saves to `portfolio_data.pkl` which the next phase loads:

```
phase_1.ipynb
phase_2.ipynb
phase_3_VaR_ES.ipynb
phase_4_backtest_kupiek.ipynb
phase_5_stress_testing.ipynb
phase_6_risk_dashboard.ipynb
```

---

## Known Limitations / Planned Work

- **Static volatility** — rolling window weights all days equally; GARCH(1,1) would fix the clustering failures in backtesting
- **Monte Carlo assumes normality** — correlated shocks drawn from normal distribution; fat-tailed draws (Student-t) would make it a genuinely independent method
- **Static covariance matrix** — breaks during regime changes (2022); DCC-GARCH or regime-switching needed
- **Equal weighting only** — mean-variance optimization and efficient frontier analysis not yet implemented
- **1-day horizon only** — multi-day VaR requires simulated paths, not √T scaling
