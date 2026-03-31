# Multi-Asset Portfolio Risk Engine

**Live app:** [multi-asset-portfolio-risk-engine.streamlit.app](https://multi-asset-portfolio-risk-engine-var-cvar-backtesting-stress.streamlit.app/)

A full institutional-style market risk pipeline for an 8-asset, $1M portfolio. Covers data ingestion, three VaR methods, CVaR (Expected Shortfall), statistical backtesting, stress testing across historical crises, GARCH(1,1)-t dynamic volatility modelling, and mean-variance portfolio optimization — all assembled into a single risk research project across 8 notebook phases.

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
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8
 Data      VaR       CVaR     Backtest   Stress    Dashboard   GARCH-t   Optimizer
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

All models pass Kupiec at 95% (correct frequency) but fail Christoffersen at both levels — exceptions cluster. For Historical Simulation at 99%: π11 = **14.3%** vs π01 = **1.35%** — after a VaR breach, another breach the next day is 10x more likely. Static models have no memory of volatility regimes.

---

### Phase 5 — Stress Testing

Historical crises and a hypothetical liquidity shock applied to the portfolio, compared against VaR/CVaR thresholds.

| Scenario | Worst Single Day | Dollar Loss | vs 99% VaR |
|---|---|---|---|
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

### Phase 7 — GARCH(1,1) with Student-t Innovations

#### Why static VaR models fail

Phase 4 revealed two distinct failure modes in all three models:

1. **Wrong frequency (Kupiec)** — too many exceptions at 99%, because normal-distribution-based thresholds underestimate how large tail losses actually are
2. **Clustered exceptions (Christoffersen)** — after one VaR breach, another is 10x more likely the next day (π11 = 14.3%). Static windows are blind to volatility regimes

#### The fix: GARCH(1,1)-t

GARCH(1,1) models time-varying conditional volatility:

```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

After a large shock, σ_t rises — so the next day's VaR threshold automatically widens. Pairing this with **Student-t innovations** (instead of normal) adds fat-tailed quantiles that reflect the portfolio's actual excess kurtosis of 16.99.

#### Results

| Method | Exceptions | Expected | Act. Rate | π11 | Kupiec | Christoffersen |
|---|---|---|---|---|---|---|
| Hist Sim (static) | 35 | 22.6 | 1.55% | 0.1429 | FAIL | FAIL |
| Parametric (static) | 50 | 22.6 | 2.21% | 0.1200 | FAIL | FAIL |
| MC (static) | 49 | 22.6 | 2.17% | 0.0816 | FAIL | FAIL |
| GARCH-Normal | 43 | 22.6 | 1.90% | 0.0465 | FAIL | FAIL |
| **GARCH-t** | **24** | **22.6** | **1.06%** | **0.0417** | **PASS** | **PASS** |

GARCH-t is the only model that passes both regulatory tests. The progression shows what each assumption costs:

- **GARCH-Normal** fixed clustering (π11: 0.143 → 0.047) but not frequency — dynamic vol alone isn't enough when the tail distribution is wrong
- **GARCH-t** fixed both — 24 exceptions vs 22.6 expected, and π11 collapsed to near the unconditional rate

This mirrors the industry's own trajectory: pre-2008 banks used parametric VaR with normal distributions, underestimated tail risk, and failed backtests they didn't run. Basel III's response was to mandate CVaR and more robust volatility models.

---

### Phase 8 — Efficient Frontier & Portfolio Optimization

Phases 1–7 treated equal-weighting as given. Phase 8 asks: given the same 8 assets, what is the optimal allocation?

The mean-variance optimization problem is solved with long-only constraints (no shorting) across 60 target return levels to trace the efficient frontier, then three specific portfolios are extracted:

**Minimum Variance** — minimises portfolio variance, ignoring return. Dominated by LQD (lowest individual vol at 8.6%), GLD and TLT (low correlation with equities). Equities (SPY, QQQ, XOM) receive minimal allocation. Result: significantly lower vol and tail risk than equal-weight, at the cost of return.

**Maximum Sharpe** — the tangency portfolio, maximising return per unit of risk. Concentrates in QQQ (Sharpe 0.774) and SPY (Sharpe 0.695). TLT receives near-zero weight due to its negative Sharpe over the period. Result: highest return and Sharpe, but carries more equity concentration.

**Equal Weight (benchmark)** — sits inside the frontier, confirming it is mean-variance inefficient. Both optimised portfolios dominate it on a risk-adjusted basis.

**Portfolio comparison across all three:**

| Portfolio | Ann. Return | Ann. Vol | Sharpe | 99% VaR ($) | 99% CVaR ($) |
|---|---|---|---|---|---|
| Equal Weight | 6.66% | 11.71% | 0.568 | $18,886 | $31,791 |
| Min Variance | lower | lower | comparable | lower | lower |
| Max Sharpe | higher | higher | higher | higher | higher |

The key finding: **portfolio construction choices directly affect tail risk**, not just normal-times volatility. Minimum variance reduces 99% VaR and CVaR materially — meaningful in a risk management context. Maximum Sharpe improves the return-risk tradeoff but accepts higher tail exposure as a consequence of equity concentration.

---

## Interactive App

The research has also been packaged into an interactive Streamlit app that lets you run the full analysis on any set of tickers, date range, and portfolio weights in real time — VaR/CVaR across all three methods, rolling backtests with Kupiec and Christoffersen results, GARCH-t conditional volatility forecasting, and mean-variance portfolio optimization with the efficient frontier.

**Live:** [multi-asset-portfolio-risk-engine.streamlit.app](https://multi-asset-portfolio-risk-engine-var-cvar-backtesting-stress.streamlit.app/)

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

## Running the Notebooks

Run notebooks in order — each phase saves to `portfolio_data.pkl` which the next phase loads:

```
notebooks/phase_1.ipynb
notebooks/phase_2.ipynb
notebooks/phase_3_VaR_ES.ipynb
notebooks/phase_4_backtest_kupiek.ipynb
notebooks/phase_5_stress_testing.ipynb
notebooks/phase_6_risk_dashboard.ipynb
notebooks/phase_7_GARCH.ipynb
notebooks/phase_8_efficient_frontier.ipynb
```

---

## Remaining Limitations

- **Static covariance matrix** — correlation is assumed constant; DCC-GARCH or regime-switching would capture correlation dynamics across market states
- **1-day horizon only** — multi-day VaR requires simulated return paths, not simple √T scaling
- **Monte Carlo normality** — MC still draws normal shocks; Student-t draws would make it a genuinely independent method from parametric
- **Mean-variance assumptions** — the optimizer uses historical means and covariances as proxies for forward-looking expectations, which is a known limitation of Markowitz-style optimization
