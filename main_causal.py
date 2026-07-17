#!/usr/bin/env python3
"""Causal inference analysis for bird-powerline conflict — main entry."""

import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.causal import run_full_analysis


def main():
    results = run_full_analysis()
    treated = results["treated"]
    control = results["control"]
    dates = results["dates"]

    # --- Summary stats ---
    its = results["its"]
    did = results["did"]

    print(f"""
  ITS Model (Poisson, 2017-2022) + Structural Break (2023+):

  Intercept:               β₀ = {its['coefficients']['intercept'][0]:.3f} (p={its['coefficients']['intercept'][2]:.4f})
  Baseline trend/qtr:      β₁ = {its['coefficients']['baseline_trend'][0]:.3f} (p={its['coefficients']['baseline_trend'][2]:.4f})
  2020 Level change:       β₂ = {its['coefficients']['level_change_2020'][0]:.3f} (p={its['coefficients']['level_change_2020'][2]:.4f})

  Mean faults/quarter:
    Pre-2020:               {its['effect_sizes']['mean_qtr_pre2020']:.2f}
    2020-2022:              {its['effect_sizes']['mean_qtr_2020_2022']:.2f}
    Post-2023:              {its['effect_sizes']['mean_qtr_post2023']:.2f}  (structural break: zero faults)

  Reduction:
    2020 vs baseline:       {its['effect_sizes']['reduction_2020_vs_baseline_pct']:.1f}%
    2023 comprehensive:     **100%** (2023-2024: zero bird-related faults)

  DiD (2023 comprehensive): {did['did_estimate']:.2f} (95% CI: [{did['ci_95'][0]:.2f}, {did['ci_95'][1]:.2f}])
  """)

    # --- Save ---
    out = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "causal_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"  Saved to output/causal_results.json")

    # --- Visualization ---
    from src.causal_viz import plot_all
    plot_all(results)


if __name__ == "__main__":
    main()
