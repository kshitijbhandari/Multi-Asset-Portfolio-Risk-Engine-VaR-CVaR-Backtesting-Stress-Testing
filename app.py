import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import yfinance as yf
from scipy.stats import norm, chi2
from scipy.stats import t as student_t
from scipy.optimize import minimize
from arch import arch_model

DARK_BG   = "#0e1117"
DARK_AX   = "#161b2a"
GRID_COL  = "#2a3347"
TEXT_COL  = "#e2e8f0"
BLUE      = "#3b82f6"
BLUE_LIGHT= "#60a5fa"
ACCENT    = "#1d4ed8"

sns.set_theme(style="dark", rc={
    "axes.facecolor":    DARK_AX,
    "figure.facecolor":  DARK_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "text.color":        TEXT_COL,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.6,
})
plt.rcParams.update({
    "font.size":          8,
    "figure.facecolor":   DARK_BG,
    "axes.facecolor":     DARK_AX,
    "axes.edgecolor":     GRID_COL,
    "axes.labelcolor":    TEXT_COL,
    "xtick.color":        TEXT_COL,
    "ytick.color":        TEXT_COL,
    "text.color":         TEXT_COL,
    "grid.color":         GRID_COL,
    "grid.linewidth":     0.6,
    "legend.facecolor":   DARK_AX,
    "legend.edgecolor":   GRID_COL,
    "legend.labelcolor":  TEXT_COL,
})

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Risk Engine",
    layout="wide",
)

st.markdown("""
<style>
/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #161b2a;
    border-radius: 8px;
    padding: 4px 6px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #94a3b8;
    border-radius: 6px;
    font-weight: 500;
    padding: 6px 18px;
}
.stTabs [aria-selected="true"] {
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
}
/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1e2d45;
}
/* ── Metric cards ── */
[data-testid="metric-container"] {
    background-color: #161b2a;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: 12px 16px;
}
/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px; }
/* ── Main background ── */
.stApp { background-color: #0e1117; }
/* ── st.info box ── */
.stAlert { background-color: #1a2744; border-left: 4px solid #3b82f6; }
</style>
""", unsafe_allow_html=True)

st.title("Portfolio Risk Engine")
st.markdown("VaR · CVaR · Backtesting · GARCH-t · Portfolio Optimization")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Portfolio Settings")

DEFAULT_TICKERS = ["SPY", "QQQ", "EFA", "GLD", "TLT", "LQD", "XOM", "VNQ"]

ticker_input = st.sidebar.text_area(
    "Tickers (one per line)",
    value="\n".join(DEFAULT_TICKERS),
    height=170,
)
tickers = [t.strip().upper() for t in ticker_input.split("\n") if t.strip()]

if len(tickers) < 2:
    st.sidebar.error("Enter at least 2 tickers.")
    st.stop()

st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start", value=pd.Timestamp("2015-01-01"))
end_date   = col2.date_input("End",   value=pd.Timestamp("2024-12-31"))

portfolio_value = st.sidebar.number_input(
    "Portfolio Value ($)", value=1_000_000, step=100_000, min_value=10_000
)
confidence = st.sidebar.selectbox(
    "Confidence Level", [0.95, 0.99], index=1,
    format_func=lambda x: f"{int(x*100)}%"
)
lookback = st.sidebar.slider("Rolling Lookback (days)", 126, 504, 252, step=21)

st.sidebar.markdown("---")
st.sidebar.markdown("**Weights** (auto-normalised to 1)")
weights_raw = {}
for t in tickers:
    weights_raw[t] = st.sidebar.number_input(
        t, min_value=0.0, max_value=1.0,
        value=round(1 / len(tickers), 3), step=0.05, key=f"w_{t}"
    )

total = sum(weights_raw.values())
if total == 0:
    weights_arr = np.array([1 / len(tickers)] * len(tickers))
else:
    weights_arr = np.array([weights_raw[t] / total for t in tickers])

st.sidebar.caption(f"Sum: {total:.3f} → normalised to 1.0")

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading price data...")
def load_data(tickers_tuple, start, end):
    raw = yf.download(
        list(tickers_tuple), start=str(start), end=str(end),
        auto_adjust=True, progress=False
    )["Close"]
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers_tuple[0])
    raw = raw[list(tickers_tuple)].dropna()
    log_ret = np.log(raw / raw.shift(1)).dropna()
    return raw, log_ret

try:
    prices, log_returns = load_data(tuple(tickers), start_date, end_date)
except Exception as e:
    st.error(f"Data download failed: {e}")
    st.stop()

if prices.empty or log_returns.empty:
    st.error("No data returned for the selected tickers and date range. Check tickers are valid and dates are not in the future.")
    st.stop()

missing = [t for t in tickers if t not in prices.columns]
if missing:
    st.error(f"Could not find data for: {', '.join(missing)}. Remove them and try again.")
    st.stop()

portfolio_returns = log_returns.dot(weights_arr)
returns  = portfolio_returns.values
dates    = portfolio_returns.index
cov_ann  = log_returns.cov() * 252
mu_ann   = log_returns.mean() * 252
N        = len(tickers)

# ── Core functions ─────────────────────────────────────────────────────────────
def hs_var_cvar(rets, c):
    var  = -np.percentile(rets, (1 - c) * 100)
    tail = rets[rets < -var]
    cvar = -tail.mean() if len(tail) > 0 else var
    return var, cvar

def param_var_cvar(w, cov, c):
    sigma = np.sqrt(w @ (cov.values / 252) @ w)
    z     = norm.ppf(1 - c)
    var   = -z * sigma
    cvar  = sigma * norm.pdf(z) / (1 - c)
    return var, cvar

def mc_var_cvar(w, cov, c, n=10_000):
    np.random.seed(42)
    L   = np.linalg.cholesky(cov.values / 252)
    sim = np.random.standard_normal((n, len(w))) @ L.T @ w
    var  = -np.percentile(sim, (1 - c) * 100)
    tail = sim[sim < -var]
    cvar = -tail.mean()
    return var, cvar

def kupiec(exc, c, lb):
    e = exc[lb:]
    T, N_ = len(e), e.sum()
    if N_ == 0 or N_ == T:
        return {"N": N_, "T": T, "p_hat": N_/T, "p_value": np.nan, "reject": True}
    p_hat  = N_ / T
    p_star = 1 - c
    LR = -2 * (
        np.log((1-p_star)**(T-N_) * p_star**N_)
        - np.log((1-p_hat)**(T-N_) * p_hat**N_)
    )
    pv = 1 - chi2.cdf(LR, 1)
    return {"N": N_, "T": T, "p_hat": p_hat, "p_value": pv, "reject": pv < 0.05}

def christoffersen(exc, c, lb):
    e = exc[lb:].astype(int)
    T = len(e)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, T):
        p, cur = e[i-1], e[i]
        if   p == 0 and cur == 0: n00 += 1
        elif p == 0 and cur == 1: n01 += 1
        elif p == 1 and cur == 0: n10 += 1
        else:                     n11 += 1
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi   = (n01 + n11) / T
    eps  = 1e-10
    LR_ind = max(-2 * (
        np.log(pi+eps)*(n01+n11) + np.log(1-pi+eps)*(n00+n10)
        - np.log(pi01+eps)*n01   - np.log(1-pi01+eps)*n00
        - np.log(pi11+eps)*n11   - np.log(1-pi11+eps)*n10
    ), 0)
    k = kupiec(exc, c, lb)
    LR_cc = (k["p_value"] and -2*np.log((1-1+c)**(k["T"]-k["N"])*(1-c)**k["N"] / ((1-k["p_hat"])**(k["T"]-k["N"])*k["p_hat"]**k["N"])) or 0) + LR_ind
    LR_uc = 0 if np.isnan(k["p_value"]) else chi2.ppf(1 - k["p_value"], 1)
    LR_cc = LR_uc + LR_ind
    pv_cc = 1 - chi2.cdf(LR_cc, 2)
    return {"pi01": pi01, "pi11": pi11, "p_value_cc": pv_cc, "reject_cc": pv_cc < 0.05}

def portfolio_opt_metrics(w, mu, cov):
    ret = float(np.dot(w, mu))
    vol = float(np.sqrt(w @ cov.values @ w))
    return ret, vol, ret / vol if vol > 0 else 0

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_var, tab_bt, tab_garch, tab_opt = st.tabs([
    "Overview", "VaR & CVaR", "Backtest", "GARCH-t", "Optimizer"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Portfolio Overview")

    # Normalised prices
    norm_prices = prices / prices.iloc[0] * 100
    PRICE_COLORS = [
        "#3b82f6","#60a5fa","#93c5fd","#1d4ed8","#2563eb",
        "#7dd3fc","#38bdf8","#0ea5e9","#0284c7","#0369a1",
    ]
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, t in enumerate(tickers):
        ax.plot(norm_prices.index, norm_prices[t], label=t,
                linewidth=1.2, color=PRICE_COLORS[i % len(PRICE_COLORS)])
    ax.set_title("Normalised Prices — Base 100")
    ax.set_ylabel("Index Level")
    ax.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Stats table
    from scipy.stats import skew, kurtosis
    combined = pd.concat([log_returns, portfolio_returns.rename("Portfolio")], axis=1)
    stats = pd.DataFrame({
        "Ann. Return (%)": (combined.mean() * 252 * 100).round(2),
        "Ann. Vol (%)":    (combined.std() * np.sqrt(252) * 100).round(2),
        "Sharpe":          ((combined.mean() / combined.std()) * np.sqrt(252)).round(3),
        "Skewness":        combined.apply(skew).round(3),
        "Exc. Kurtosis":   combined.apply(kurtosis).round(3),
        "Min Day (%)":     (combined.min() * 100).round(3),
        "Max Day (%)":     (combined.max() * 100).round(3),
    })
    st.dataframe(stats, use_container_width=True)

    # Correlation heatmap
    st.markdown("**Correlation Matrix**")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(log_returns.corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, vmin=-1, vmax=1,
                linewidths=0.5, annot_kws={"size": 8}, ax=ax,
                linecolor=DARK_BG)
    ax.set_title("Asset Correlation (Daily Log Returns)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: VaR & CVaR
# ══════════════════════════════════════════════════════════════════════════════
with tab_var:
    st.subheader("Value at Risk & CVaR")

    c = confidence
    hs_v,  hs_cv  = hs_var_cvar(returns, c)
    pm_v,  pm_cv  = param_var_cvar(weights_arr, cov_ann, c)
    mc_v,  mc_cv  = mc_var_cvar(weights_arr, cov_ann, c)

    # Comparison table
    df_var = pd.DataFrame({
        "VaR (%)":    [f"{hs_v*100:.4f}", f"{pm_v*100:.4f}", f"{mc_v*100:.4f}"],
        "CVaR (%)":   [f"{hs_cv*100:.4f}", f"{pm_cv*100:.4f}", f"{mc_cv*100:.4f}"],
        "VaR ($)":    [f"${hs_v*portfolio_value:,.0f}", f"${pm_v*portfolio_value:,.0f}", f"${mc_v*portfolio_value:,.0f}"],
        "CVaR ($)":   [f"${hs_cv*portfolio_value:,.0f}", f"${pm_cv*portfolio_value:,.0f}", f"${mc_cv*portfolio_value:,.0f}"],
    }, index=["Historical Sim", "Parametric", "Monte Carlo"])
    st.dataframe(df_var, use_container_width=True)

    # Distribution plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(returns * 100, bins=80, density=True,
            color=BLUE, alpha=0.45, label="Empirical")

    ax.axvline(-hs_v  * 100, color="#f59e0b", lw=2,   ls="--", label=f"HS VaR {int(c*100)}% = {hs_v*100:.2f}%")
    ax.axvline(-hs_cv * 100, color="#ef4444", lw=2,   ls="-",  label=f"HS CVaR = {hs_cv*100:.2f}%")
    ax.axvline(-pm_v  * 100, color="#22c55e", lw=1.5, ls="--", label=f"Param VaR = {pm_v*100:.2f}%")
    ax.axvline(-mc_v  * 100, color="#a855f7", lw=1.5, ls="--", label=f"MC VaR = {mc_v*100:.2f}%")

    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.set_title(f"Return Distribution with VaR/CVaR Thresholds ({int(c*100)}% confidence)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # CVaR tail visualisation
    fig, ax = plt.subplots(figsize=(12, 3))
    tail = returns[returns < -hs_v]
    ax.hist(returns * 100, bins=80, density=True, color=BLUE, alpha=0.35, label="Full distribution")
    ax.hist(tail * 100, bins=20, density=False,
            weights=np.ones(len(tail)) / len(returns),
            color="#ef4444", alpha=0.85, label="Tail losses (beyond VaR)")
    ax.axvline(-hs_v  * 100, color="#f59e0b", lw=2, ls="--", label=f"VaR = {hs_v*100:.2f}%")
    ax.axvline(tail.mean() * 100, color="#ef4444", lw=2, label=f"CVaR = {-tail.mean()*100:.2f}%")
    ax.set_xlabel("Daily Return (%)")
    ax.set_title("CVaR = Mean of the Tail Beyond VaR")
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.subheader("Backtesting — Kupiec & Christoffersen")
    st.caption("Rolling out-of-sample VaR vs actual returns. Only HS and Parametric (MC rolling is too slow for a live app).")

    c = confidence

    @st.cache_data(show_spinner="Running rolling backtest...")
    def run_backtest(returns_tuple, lookback, confidence):
        rets = np.array(returns_tuple)
        n    = len(rets)
        z    = norm.ppf(1 - confidence)
        roll_hs    = np.full(n, np.nan)
        roll_param = np.full(n, np.nan)
        for t in range(lookback, n):
            w = rets[t - lookback : t]
            roll_hs[t]    = -np.percentile(w, (1 - confidence) * 100)
            roll_param[t] = -z * w.std()
        exc_hs    = np.zeros(n, dtype=int)
        exc_param = np.zeros(n, dtype=int)
        v = ~np.isnan(roll_hs)
        exc_hs[v]    = (rets[v] < -roll_hs[v]).astype(int)
        exc_param[v] = (rets[v] < -roll_param[v]).astype(int)
        return roll_hs, roll_param, exc_hs, exc_param

    roll_hs, roll_param, exc_hs, exc_param = run_backtest(
        tuple(returns), lookback, c
    )

    # Results table
    rows = []
    for name, exc in [("Historical Sim", exc_hs), ("Parametric", exc_param)]:
        k  = kupiec(exc, c, lookback)
        cr = christoffersen(exc, c, lookback)
        rows.append({
            "Method":      name,
            "Exceptions":  k["N"],
            "Expected":    round((1 - c) * k["T"], 1),
            "Act. Rate":   f"{k['p_hat']:.3%}",
            "pi11":        f"{cr['pi11']:.4f}",
            "Kupiec":      "PASS" if not k["reject"]    else "FAIL",
            "Christoff":   "PASS" if not cr["reject_cc"] else "FAIL",
        })
    df_bt = pd.DataFrame(rows).set_index("Method")
    st.dataframe(df_bt, use_container_width=True)

    # Rolling VaR plot
    valid_dates  = dates[lookback:]
    actual_valid = returns[lookback:] * 100

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, (name, roll, exc, color) in zip(axes, [
        ("Historical Simulation", roll_hs,    exc_hs,    BLUE_LIGHT),
        ("Parametric",            roll_param, exc_param, "#22c55e"),
    ]):
        exc_ = exc[lookback:]
        ax.plot(valid_dates, actual_valid, color="#64748b", lw=0.5, alpha=0.7, label="Actual")
        ax.plot(valid_dates, -roll[lookback:] * 100, color=color, lw=1.2,
                label=f"{int(c*100)}% VaR")
        ax.scatter(valid_dates[exc_ == 1], actual_valid[exc_ == 1],
                   color="#ef4444", s=10, zorder=5, label=f"Exceptions ({exc_.sum()})")
        ax.axhline(0, color=GRID_COL, lw=0.5)
        ax.set_ylabel("Return (%)")
        ax.set_title(f"{name} — Rolling {int(c*100)}% VaR")
        ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: GARCH-t
# ══════════════════════════════════════════════════════════════════════════════
with tab_garch:
    st.subheader("GARCH(1,1) with Student-t Innovations")

    @st.cache_data(show_spinner="Fitting GARCH-t model...")
    def fit_garch_t(returns_tuple):
        rets_pct = np.array(returns_tuple) * 100
        am  = arch_model(rets_pct, vol="Garch", p=1, q=1, mean="Zero", dist="t")
        res = am.fit(disp="off")
        cond_vol = res.conditional_volatility
        if hasattr(cond_vol, "values"):
            cond_vol = cond_vol.values
        cond_vol = np.array(cond_vol, dtype=float)
        return (
            float(res.params["omega"]),
            float(res.params["alpha[1]"]),
            float(res.params["beta[1]"]),
            float(res.params["nu"]),
            cond_vol,
        )

    omega, alpha, beta, nu, cond_vol = fit_garch_t(tuple(returns))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("omega (ω)", f"{omega:.6f}")
    col2.metric("alpha (α)", f"{alpha:.4f}")
    col3.metric("beta (β)",  f"{beta:.4f}")
    col4.metric("nu (ν) — degrees of freedom", f"{nu:.2f}")

    st.caption(f"alpha + beta = {alpha+beta:.4f}  (< 1 = stationary).  Low nu confirms fat tails.")

    # Current VaR forecast
    z_t    = float(student_t.ppf(1 - confidence, df=nu))
    sigma_today = cond_vol[-1] / 100
    garch_var_today = -z_t * sigma_today

    col1, col2, col3 = st.columns(3)
    col1.metric(f"GARCH-t {int(confidence*100)}% VaR (today)",
                f"{garch_var_today*100:.4f}%",
                f"${garch_var_today*portfolio_value:,.0f}")
    hs_v, _ = hs_var_cvar(returns, confidence)
    col2.metric("Static HS VaR (full sample)", f"{hs_v*100:.4f}%",
                f"${hs_v*portfolio_value:,.0f}")
    col3.metric("Current conditional vol (daily)",
                f"{cond_vol[-1]:.4f}%")

    # Conditional vol chart
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    axes[0].fill_between(dates, returns * 100, 0,
                         where=(returns >= 0), color=BLUE,      alpha=0.65, label="Gain")
    axes[0].fill_between(dates, returns * 100, 0,
                         where=(returns < 0),  color="#ef4444", alpha=0.65, label="Loss")
    axes[0].set_ylabel("Return (%)")
    axes[0].set_title("Portfolio Daily Returns")
    axes[0].legend(fontsize=7)

    axes[1].plot(dates, cond_vol, color="#f59e0b", lw=1.0,
                 label="GARCH-t Conditional Vol (%)")
    axes[1].set_ylabel("Conditional Vol (%)")
    axes[1].set_title("GARCH(1,1)-t Conditional Volatility — Spikes During Crises")
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info(
        f"**Interpretation:** nu = {nu:.1f} < 10 confirms fat tails — the normal distribution "
        f"underestimates tail risk. The t-quantile at {int(confidence*100)}% is "
        f"{z_t:.4f} vs normal {norm.ppf(1-confidence):.4f} — "
        f"{abs(z_t/norm.ppf(1-confidence)):.2f}x more conservative."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════
with tab_opt:
    st.subheader("Efficient Frontier & Portfolio Optimization")

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds_opt  = [(0, 1)] * N
    w0          = np.array([1 / N] * N)

    @st.cache_data(show_spinner="Optimising portfolios...")
    def optimise(mu_tuple, cov_tuple, n):
        mu  = np.array(mu_tuple)
        cov = pd.DataFrame(np.array(cov_tuple).reshape(n, n))
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bnds = [(0, 1)] * n
        w0_  = np.array([1/n] * n)

        # Min Variance
        res_mv = minimize(lambda w: w @ cov.values @ w, w0_,
                          method="SLSQP", bounds=bnds, constraints=cons)
        w_mv = res_mv.x

        # Max Sharpe
        def neg_sharpe(w):
            r = np.dot(w, mu)
            v = np.sqrt(w @ cov.values @ w)
            return -r / v if v > 0 else 0
        res_ms = minimize(neg_sharpe, w0_, method="SLSQP", bounds=bnds, constraints=cons)
        w_ms = res_ms.x

        # Frontier
        mv_ret = float(np.dot(w_mv, mu))
        fr_vols, fr_rets, fr_ws = [], [], []
        for target in np.linspace(mv_ret, float(mu.max()), 60):
            c_fr = cons + [{"type": "eq", "fun": lambda w, t=target: np.dot(w, mu) - t}]
            res = minimize(lambda w: w @ cov.values @ w, w0_,
                           method="SLSQP", bounds=bnds, constraints=c_fr)
            if res.success:
                fr_vols.append(np.sqrt(res.fun))
                fr_rets.append(target)
                fr_ws.append(res.x)

        return w_mv, w_ms, np.array(fr_vols), np.array(fr_rets), fr_ws

    w_mv, w_ms, fr_vols, fr_rets, fr_ws = optimise(
        tuple(mu_ann.values), tuple(cov_ann.values.flatten()), N
    )

    # Metrics for each portfolio
    rets_mv = log_returns.dot(w_mv)
    rets_ms = log_returns.dot(w_ms)
    rets_ew = portfolio_returns

    rows_opt = []
    for name, w, rets in [
        ("Equal Weight", weights_arr, rets_ew),
        ("Min Variance",  w_mv, rets_mv),
        ("Max Sharpe",    w_ms, rets_ms),
    ]:
        r, v, s = portfolio_opt_metrics(w, mu_ann, cov_ann)
        var95, cvar95 = hs_var_cvar(rets.values, 0.95)
        var99, cvar99 = hs_var_cvar(rets.values, 0.99)
        rows_opt.append({
            "Portfolio":    name,
            "Ann. Return":  f"{r*100:.2f}%",
            "Ann. Vol":     f"{v*100:.2f}%",
            "Sharpe":       f"{s:.3f}",
            "95% VaR ($)":  f"${var95*portfolio_value:,.0f}",
            "99% VaR ($)":  f"${var99*portfolio_value:,.0f}",
            "95% CVaR ($)": f"${cvar95*portfolio_value:,.0f}",
            "99% CVaR ($)": f"${cvar99*portfolio_value:,.0f}",
        })

    st.dataframe(pd.DataFrame(rows_opt).set_index("Portfolio"), use_container_width=True)

    # Frontier plot
    ew_r, ew_v, _ = portfolio_opt_metrics(weights_arr, mu_ann, cov_ann)
    mv_r, mv_v, _ = portfolio_opt_metrics(w_mv, mu_ann, cov_ann)
    ms_r, ms_v, _ = portfolio_opt_metrics(w_ms, mu_ann, cov_ann)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fr_vols * 100, fr_rets * 100, color=BLUE, lw=2.5, label="Efficient Frontier")

    for i, t in enumerate(tickers):
        v = np.sqrt(cov_ann.values[i, i])
        r = mu_ann.iloc[i]
        ax.scatter(v * 100, r * 100, s=50, color="#94a3b8", zorder=4)
        ax.annotate(t, (v*100, r*100), textcoords="offset points",
                    xytext=(5, 3), fontsize=8, color="#94a3b8")

    ax.scatter(ew_v*100, ew_r*100, s=120, color=BLUE_LIGHT, marker="o",
               edgecolors=TEXT_COL, lw=0.8, zorder=5, label="Equal Weight")
    ax.scatter(mv_v*100, mv_r*100, s=120, color="#22c55e", marker="D",
               edgecolors=TEXT_COL, lw=0.8, zorder=5, label="Min Variance")
    ax.scatter(ms_v*100, ms_r*100, s=180, color="#f59e0b", marker="*",
               edgecolors=TEXT_COL, lw=0.8, zorder=5, label="Max Sharpe")

    ax.set_xlabel("Annualised Volatility (%)")
    ax.set_ylabel("Annualised Return (%)")
    ax.set_title("Efficient Frontier")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Weight comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (title, w, color) in zip(axes, [
        ("Equal Weight", weights_arr, BLUE_LIGHT),
        ("Min Variance", w_mv,        "#22c55e"),
        ("Max Sharpe",   w_ms,        "#f59e0b"),
    ]):
        bars = ax.bar(tickers, w * 100, color=color, alpha=0.85,
                      edgecolor=GRID_COL, linewidth=0.5)
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=7)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Weight (%)")
        ax.set_ylim(0, max(w * 100) * 1.3)
        ax.axhline(100/N, color="#64748b", lw=1, ls="--", alpha=0.6)
    plt.suptitle("Portfolio Weight Allocations", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
