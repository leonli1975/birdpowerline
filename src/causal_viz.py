"""Visualization for causal inference analysis."""

import os, logging
import numpy as np
import matplotlib
matplotlib.use("Agg")

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
_CJK_FONT = None
for _c in ["Noto Sans SC","SimHei","WenQuanYi Micro Hei"]:
    try:
        from matplotlib.font_manager import findfont, FontProperties
        _fp = FontProperties(family=_c)
        if findfont(_fp, fallback_to_default=False):
            _CJK_FONT = _c; break
    except: continue
if _CJK_FONT:
    matplotlib.rcParams["font.family"] = _CJK_FONT
    matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
CB = ["#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9",
      "#D55E00","#F0E442","#000000"]


def plot_all(results):
    os.makedirs(OUT, exist_ok=True)
    plot_its(results)
    plot_did(results)
    plot_causal_graph(results)
    print(f"  Causal plots saved to {OUT}/")


def plot_its(results):
    """ITS plot: quarterly actual vs fitted vs counterfactual."""
    treated = results["treated"]
    its = results["its"]
    dates = results["dates"]
    interventions = results["interventions"]

    n = len(treated)
    x = np.arange(n)

    # Annual aggregation for inset
    n_years = n // 4
    annual_actual = [sum(treated[i*4:(i+1)*4]) for i in range(n_years)]
    annual_cf = [sum(its["counterfactual"][i*4:(i+1)*4]) for i in range(n_years)]
    years = list(range(2017, 2017 + n_years))

    fig, ax = plt.subplots(figsize=(12, 5))

    # Quarterly data
    ax.bar(x, treated, color=CB[0], alpha=0.5, width=0.6, label="Actual faults (quarterly)")
    ax.plot(x, its["fitted"], color=CB[1], linewidth=2.5, label="ITS fitted (Poisson)")
    ax.plot(x, its["counterfactual"], color=CB[2], linestyle="--",
            linewidth=2, label="Counterfactual (no measures)")

    # Intervention lines
    for ti, label in interventions.items():
        ax.axvline(x=ti - 0.5, color="red", linestyle=":", alpha=0.7, linewidth=2)
        ax.text(ti, ax.get_ylim()[1]*0.9,
                f"{dates[ti]}\n{label.split(':')[0][:20]}",
                fontsize=7, color="red", rotation=90, va="top")

    ax.set_xlabel("Quarter (2017-Q1 to 2024-Q1)")
    ax.set_ylabel("Fault Count")
    ax.set_title("Interrupted Time Series: Bird-Related Faults (若真线 110kV)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-0.5, n - 0.5)

    # Remove CJK from labels to avoid font issues
    ax.set_title("Interrupted Time Series: Bird-Related Faults (Ruozhen Line 110kV)")
    ax.set_xlabel("Quarter (2017-Q1 to 2024-Q1)")

    # Inset: annual bar chart
    inset = fig.add_axes([0.62, 0.55, 0.30, 0.30])
    bw = 0.35
    inset.bar(np.array(years) - bw/2, annual_actual, bw, label="Actual", color=CB[0], alpha=0.8)
    inset.bar(np.array(years) + bw/2, annual_cf, bw, label="CF (no measures)",
              color=CB[2], alpha=0.5, hatch="//")
    inset.set_title("Annual Total", fontsize=9)
    inset.set_xticks(years)
    inset.set_xticklabels(years, fontsize=7, rotation=45)
    inset.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "causal_its.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_did(results):
    """DiD plot: pre-post comparison for treated vs control."""
    did = results["did"]
    fig, ax = plt.subplots(figsize=(7, 5))

    groups = ["Ruozhen (Treated)", "Other 110kV (Control)"]
    pre_vals = [did["pre_treated_mean"], did["pre_control_mean"]]
    post_vals = [did["post_treated_mean"], did["post_control_mean"]]

    x = np.arange(len(groups))
    w = 0.35

    bars1 = ax.bar(x - w/2, pre_vals, w, label="Pre (2022-Q1 to 2023-Q1)",
                   color=CB[0], alpha=0.8)
    bars2 = ax.bar(x + w/2, post_vals, w, label="Post (2023-Q2 to 2024-Q1)",
                   color=CB[2], alpha=0.8)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", fontsize=11)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("Mean Quarterly Faults")
    ax.set_title(f"DiD Estimate = {did['did_estimate']:.2f} "
                 f"(95% CI: [{did['ci_95'][0]:.2f}, {did['ci_95'][1]:.2f}])")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "causal_did.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_causal_graph(results):
    """Simple causal discovery visualization."""
    cd = results["causal_discovery"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    sig_edges = [r for r in cd if r["significant"]]
    all_edges = cd

    text = "Causal Discovery (PCMCI-lite, lag=1-3 months)\n\n"
    for r in all_edges:
        sig = "✓" if r["significant"] else "✗"
        text += f"  {sig} {r['from']:20s} → {r['to']:10s} "
        text += f"r={r['pearson_r']:+.3f} (p={r['p_value']:.4f})\n"

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    ax.set_title("Causal Discovery Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "causal_discovery.png"), dpi=150, bbox_inches="tight")
    plt.close()
