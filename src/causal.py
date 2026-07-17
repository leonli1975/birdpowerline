"""Causal inference analysis for bird-powerline conflict mitigation.

Direction 4 from rptask.md:
"基于因果推断的鸟类-线路冲突驱动力分析"

Methods:
  1. Interrupted Time Series (ITS) — primary
  2. Difference-in-Differences (DiD)
  3. Causal discovery (PCMCI-lite)
"""

import numpy as np
import os

# ============================================================
# 1. Construct realistic monthly fault time series
# ============================================================

MONTHS = [
    "2017-01","2017-02","2017-03","2017-04","2017-05","2017-06",
    "2017-07","2017-08","2017-09","2017-10","2017-11","2017-12",
    "2018-01","2018-02","2018-03","2018-04","2018-05","2018-06",
    "2018-07","2018-08","2018-09","2018-10","2018-11","2018-12",
    "2019-01","2019-02","2019-03","2019-04","2019-05","2019-06",
    "2019-07","2019-08","2019-09","2019-10","2019-11","2019-12",
    "2020-01","2020-02","2020-03","2020-04","2020-05","2020-06",
    "2020-07","2020-08","2020-09","2020-10","2020-11","2020-12",
    "2021-01","2021-02","2021-03","2021-04","2021-05","2021-06",
    "2021-07","2021-08","2021-09","2021-10","2021-11","2021-12",
    "2022-01","2022-02","2022-03","2022-04","2022-05","2022-06",
    "2022-07","2022-08","2022-09","2022-10","2022-11","2022-12",
    "2023-01","2023-02","2023-03","2023-04","2023-05","2023-06",
    "2023-07","2023-08","2023-09","2023-10","2023-11","2023-12",
    "2024-01","2024-02","2024-03",
]
N_MONTHS = len(MONTHS)
MONTH_IDX = {m: i for i, m in enumerate(MONTHS)}

# Quarterly aggregation
QUARTERS = [
    "2017-Q1","2017-Q2","2017-Q3","2017-Q4",
    "2018-Q1","2018-Q2","2018-Q3","2018-Q4",
    "2019-Q1","2019-Q2","2019-Q3","2019-Q4",
    "2020-Q1","2020-Q2","2020-Q3","2020-Q4",
    "2021-Q1","2021-Q2","2021-Q3","2021-Q4",
    "2022-Q1","2022-Q2","2022-Q3","2022-Q4",
    "2023-Q1","2023-Q2","2023-Q3","2023-Q4",
    "2024-Q1",
]
N_QUARTERS = len(QUARTERS)

# Intervention time points
INTERVENTION_2020 = 12  # 2020-Q1 (first dynamic spikes deployed March 2020)
INTERVENTION_2023 = 24  # 2023-Q2 (comprehensive deployment April 2023)


def build_fault_series_quarterly():
    """Build quarterly fault time series using documented seasonal patterns.

    Uses realistic fault counts derived from document data,
    distributed according to known seasonal peaking patterns.
    """
    rng = np.random.RandomState(42)

    # Seasonal weights per quarter (from national data)
    # Q1 (Jan-Mar): spring migration peak
    # Q2 (Apr-Jun): breeding peak — highest
    # Q3 (Jul-Sep): moderate
    # Q4 (Oct-Dec): autumn secondary peak
    q_weights_base = np.array([1.2, 1.8, 0.8, 1.0])  # relative weights
    q_weights_base /= q_weights_base.sum()

    def quarterly_from_annual(annual_count, year, scale=1.0):
        w = q_weights_base + rng.uniform(-0.1, 0.1, 4)
        w = np.maximum(w, 0); w /= w.sum()
        counts = rng.poisson(annual_count * w * scale)
        return counts.tolist()

    # Annual fault counts per line (from documents)
    # 若真线 annual totals
    annual_treated = {
        2017: 9, 2018: 9, 2019: 8,     # baseline: 26 in 3 years
        2020: 5, 2021: 4, 2022: 3,     # reduced: 12 in 3 years (54% drop)
        2023: 0, 2024: 0,              # post-comprehensive: 0
    }
    annual_control = {
        2017: 7, 2018: 8, 2019: 7,     # control lines also high
        2020: 4, 2021: 4, 2022: 4,     # some reduction (they got partial treatment too)
        2023: 1, 2024: 0,
    }

    treated = []
    control = []
    for year in range(2017, 2025):
        treated.extend(quarterly_from_annual(annual_treated.get(year, 0), year))
        control.extend(quarterly_from_annual(annual_control.get(year, 0), year))

    # Trim
    treated = treated[:N_QUARTERS]
    control = control[:N_QUARTERS]

    interventions = {
        INTERVENTION_2020: "2020-Q1: Dynamic spikes + smart deterrents deployed on 110kV lines",
        INTERVENTION_2023: "2023-Q2: Comprehensive deployment (baffles+nests+video+deterrents)",
    }
    return treated, control, QUARTERS, interventions


# ============================================================
# 2. Interrupted Time Series (ITS) Analysis
# ============================================================

def its_analysis(treated, control, interventions, dates):
    """Interrupted Time Series with Poisson regression.

    Fits ITS model on 2017-2022 data (pre-2023 comprehensive).
    2023 intervention is a structural break (hard zero), not a coefficient.
    """
    from scipy.stats import chi2

    n = len(treated)
    t = np.arange(n)

    # Only model pre-2023 (first 24 quarters: 2017-2022)
    n_pre2023 = INTERVENTION_2023
    I1 = np.array([1.0 if i >= INTERVENTION_2020 else 0.0 for i in range(n_pre2023)])

    X = np.column_stack([np.ones(n_pre2023), np.arange(n_pre2023), I1])
    y = np.array(treated[:n_pre2023], dtype=float)

    # Add small epsilon to handle potential zeros
    y = y + 1e-6

    # Poisson IRLS
    beta = np.zeros(3)
    for _ in range(50):
        eta = X @ beta
        mu = np.exp(np.clip(eta, -10, 10))
        W = np.diag(mu)
        z = eta + (y - mu) / np.maximum(mu, 1e-6)
        try:
            beta_new = np.linalg.solve(X.T @ W @ X, X.T @ W @ z)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ z, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            break
        beta = beta_new

    mu_final = np.exp(X @ beta)
    W_final = np.diag(mu_final)
    try:
        cov = np.linalg.inv(X.T @ W_final @ X)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(X.T @ W_final @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z_stats = beta / np.maximum(np.abs(se), 1e-10)
    p_values = 2 * (1 - chi2.cdf(z_stats**2, 1) ** 0.5)

    # Build full-length fitted and counterfactual
    fitted = np.concatenate([mu_final, np.zeros(n - n_pre2023)])
    # Counterfactual: extend pre-2020 trend through 2023-2024
    cf = np.zeros(n)
    X_cf = np.column_stack([np.ones(n), np.arange(n), np.zeros(n)])
    cf = np.exp(np.clip(X_cf @ np.array([beta[0], beta[1], 0]), -10, 10))
    cf[n_pre2023:] = 0  # structural break: zero after 2023

    actual_total = np.sum(treated[n_pre2023:])
    cf_total = np.sum(cf[n_pre2023:])
    avoided = max(0, cf_total - actual_total)

    return {
        "coefficients": {
            "intercept": (float(beta[0]), float(se[0]), float(p_values[0])),
            "baseline_trend": (float(beta[1]), float(se[1]), float(p_values[1])),
            "level_change_2020": (float(beta[2]), float(se[2]), float(p_values[2])),
        },
        "effect_sizes": {
            "level_change_2020_coef": float(beta[2]),
            "mean_qtr_pre2020": float(np.mean(treated[:INTERVENTION_2020])),
            "mean_qtr_2020_2022": float(np.mean(treated[INTERVENTION_2020:INTERVENTION_2023])),
            "mean_qtr_post2023": float(np.mean(treated[INTERVENTION_2023:])),
            "reduction_2020_vs_baseline_pct": float(
                100 * (np.mean(treated[:INTERVENTION_2020]) -
                       np.mean(treated[INTERVENTION_2020:INTERVENTION_2023])) /
                max(np.mean(treated[:INTERVENTION_2020]), 0.01)),
            "reduction_2023_vs_baseline_pct": 100.0,  # zero faults = 100% reduction
            "cumulative_avoided_2023_2024": float(avoided),
            "structural_break_2023": True,
        },
        "fitted": fitted.tolist(),
        "counterfactual": cf.tolist(),
        "actual": np.array(treated).tolist(),
        "dates": dates,
        "interventions": interventions,
        "model": "Poisson (2017-2022) + structural break (2023+)",
    }


# ============================================================
# 3. Difference-in-Differences (DiD)
# ============================================================

def did_analysis(treated, control, interventions, dates):
    """Difference-in-Differences for the 2023 comprehensive intervention.

    若真线 (treated) vs other 110kV lines (control) before/after 2023-Q2.
    """
    # Pre: 2022-Q1 to 2023-Q1 (5 quarters before comprehensive)
    pre_start = 20   # 2022-Q1
    pre_end = 23     # 2023-Q1
    # Post: 2023-Q2 to 2024-Q1 (4 quarters after)
    post_start = 24  # 2023-Q2
    post_end = 27    # 2024-Q1

    pre_t = np.array(treated[pre_start:pre_end+1])
    pre_c = np.array(control[pre_start:pre_end+1])
    post_t = np.array(treated[post_start:post_end+1])
    post_c = np.array(control[post_start:post_end+1])

    pre_t_mean = np.mean(pre_t)
    pre_c_mean = np.mean(pre_c)
    post_t_mean = np.mean(post_t)
    post_c_mean = np.mean(post_c)

    did = (post_t_mean - pre_t_mean) - (post_c_mean - pre_c_mean)

    # Bootstrap CI
    rng = np.random.RandomState(42)
    did_boot = []
    for _ in range(2000):
        bt_pre = rng.choice(pre_t, size=len(pre_t), replace=True)
        bt_post = rng.choice(post_t, size=len(post_t), replace=True)
        bc_pre = rng.choice(pre_c, size=len(pre_c), replace=True)
        bc_post = rng.choice(post_c, size=len(post_c), replace=True)
        did_boot.append((bt_post.mean() - bt_pre.mean()) - (bc_post.mean() - bc_pre.mean()))

    did_boot = np.array(did_boot)
    ci_lower = np.percentile(did_boot, 2.5)
    ci_upper = np.percentile(did_boot, 97.5)

    return {
        "pre_treated_mean": float(pre_t_mean),
        "pre_control_mean": float(pre_c_mean),
        "post_treated_mean": float(post_t_mean),
        "post_control_mean": float(post_c_mean),
        "did_estimate": float(did),
        "ci_95": (float(ci_lower), float(ci_upper)),
        "significant": ci_upper < 0,
        "interpretation": (
            f"After comprehensive deployment, 若真线 faults changed by "
            f"{post_t_mean - pre_t_mean:+.2f}/quarter vs control lines "
            f"{post_c_mean - pre_c_mean:+.2f}/quarter. "
            f"DiD = {did:.2f} (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])."
            f"{' Significant reduction.' if ci_upper < 0 else ' Not significant at 0.05.'}"
        ),
    }


# ============================================================
# 4. Causal Discovery (PCMCI-lite)
# ============================================================

def causal_discovery(treated, control, dates):
    """Simplified causal discovery using Granger-style lagged correlation."""
    from scipy.stats import pearsonr

    n = len(treated)

    # Measure indicator (binary)
    measure = np.array([1.0 if i >= INTERVENTION_2020 else 0.0 for i in range(n)])
    measure2 = np.array([1.0 if i >= INTERVENTION_2023 else 0.0 for i in range(n)])

    # Breed/non-breed season indicator (Q1-Q2 = breeding)
    q_idx = np.array([int(d.split("-Q")[1]) for d in dates])
    breed_season = np.array([1.0 if q in (1, 2) else 0.0 for q in q_idx])

    y = np.array(treated)
    lags = [1, 2, 3, 4]  # up to 4 quarters
    results = []

    for lag in lags:
        y_cur = y[lag:]
        n_cur = len(y_cur)

        # measure → fault
        m_lag = measure[:-lag]
        if np.std(m_lag) > 0:
            r, p = pearsonr(m_lag, y_cur)
            results.append({
                "from": f"measure(t-{lag}Q)", "to": "fault(t)",
                "pearson_r": float(r), "p_value": float(p),
                "significant": p < 0.05,
            })

        # breeding season → fault
        b_lag = breed_season[:-lag]
        r, p = pearsonr(b_lag, y_cur)
        results.append({
            "from": f"breeding_season(t-{lag}Q)", "to": "fault(t)",
            "pearson_r": float(r), "p_value": float(p),
            "significant": p < 0.05,
        })

    return results


# ============================================================
# 5. Report generation
# ============================================================

def run_full_analysis():
    """Run all causal analyses and return results dict."""
    treated, control, dates, interventions = build_fault_series_quarterly()

    print("=== Interrupted Time Series (Poisson) ===")
    its = its_analysis(treated, control, interventions, dates)
    for k, (b, se, p) in its["coefficients"].items():
        sig = "*" if p < 0.1 else ("**" if p < 0.05 else "***" if p < 0.01 else "")
        print(f"  {k}: β={b:+.3f} se={se:.3f} p={p:.4f} {sig}")
    eff = its["effect_sizes"]
    print(f"  Mean faults/qtr: pre2020={eff['mean_qtr_pre2020']:.2f} "
          f"2020-22={eff['mean_qtr_2020_2022']:.2f} "
          f"post2023={eff['mean_qtr_post2023']:.2f}")
    print(f"  Reduction 2020 vs baseline: {eff['reduction_2020_vs_baseline_pct']:.1f}%")
    print(f"  Structural break 2023+: 100% reduction (zero faults)")

    print("\n=== Difference-in-Differences (DiD) ===")
    did = did_analysis(treated, control, interventions, dates)
    print(f"  DiD estimate: {did['did_estimate']:.3f}")
    print(f"  95% CI: [{did['ci_95'][0]:.3f}, {did['ci_95'][1]:.3f}]")
    print(f"  Significant: {did['significant']}")

    print("\n=== Causal Discovery ===")
    cd = causal_discovery(treated, control, dates)
    for r in cd:
        sig = "*" if r["significant"] else " "
        print(f"  {r['from']} → {r['to']}: r={r['pearson_r']:.3f} p={r['p_value']:.4f} {sig}")

    return {
        "treated": treated,
        "control": control,
        "dates": dates,
        "interventions": interventions,
        "its": its,
        "did": did,
        "causal_discovery": cd,
    }
