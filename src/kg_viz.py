"""Visualization for temporal knowledge graph reasoning results."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

_CJK_FONT = None
for _candidate in ["Noto Sans SC", "SimHei", "WenQuanYi Micro Hei",
                   "WenQuanYi Zen Hei", "Source Han Sans SC", "Microsoft YaHei"]:
    try:
        from matplotlib.font_manager import findfont, FontProperties
        _fp = FontProperties(family=_candidate)
        _found = findfont(_fp, fallback_to_default=False)
        if _found and os.path.exists(_found):
            _CJK_FONT = _candidate
            break
    except Exception:
        continue
if _CJK_FONT:
    matplotlib.rcParams["font.family"] = _CJK_FONT
    matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
CB = ["#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9",
      "#D55E00","#F0E442","#000000"]


def ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_link_prediction_metrics(metrics, save=True):
    """Bar chart of MRR, Hits@1,3,10."""
    ensure_dir()
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
    vals = [metrics.get(n, 0) for n in names]
    bars = ax.bar(names, vals, color=CB[:4], width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Temporal Link Prediction Performance")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, "kg_link_pred.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()


def plot_threat_evolution(model, e2i, i2e, r2i, timestamps, save=True):
    """Plot threat probability evolution for key bird-line pairs."""
    ensure_dir()
    threat_r = r2i.get("threatens")
    if threat_r is None:
        return

    pairs = [
        ("Bu_hemilasius", "line_ruozhen", "大鵟→若真线"),
        ("Bu_hemilasius", "line_ruotang", "大鵟→若唐线"),
        ("Mi_migrans", "line_ruozhen", "黑鸢→若真线"),
        ("Fa_tinnunculus", "line_ruozhen", "红隼→若真线"),
    ]
    n = len(timestamps)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (bird, line, label) in enumerate(pairs):
        if bird not in e2i or line not in e2i:
            continue
        b_id, l_id = e2i[bird], e2i[line]
        probs = []
        for tau in range(n):
            s = model.score(b_id, threat_r, l_id, tau)
            probs.append(model._sigmoid(s))
        ax.plot(timestamps, probs, "o-", label=label, color=CB[i], linewidth=2, markersize=6)

    # Annotate measure deployment
    ax.axvline(x=2020, color="gray", linestyle="--", alpha=0.5, label="First measures (2020)")
    ax.axvline(x=2023, color="red", linestyle="--", alpha=0.5, label="Comprehensive (2023)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Threat Probability")
    ax.set_title("Temporal Threat Evolution (Learned from KG Embeddings)")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, "kg_threat_evolution.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()


def plot_measure_scores(scores_dict, save=True):
    """Bar chart of measure recommendation scores."""
    ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(scores_dict.keys())
    vals = list(scores_dict.values())
    colors = [CB[2] if "recommend NEW" not in n else CB[1] for n in names]

    bars = ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Score")
    ax.set_title("Measure Recommendation Scores (大鵟 → nest shorting, 2024)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, "kg_measure_recs.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()


def plot_training_curve(history, save=True):
    """Plot training loss curve."""
    ensure_dir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"], color=CB[0], linewidth=1, alpha=0.7)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Margin Loss")
    axes[0].set_title("Training Loss")

    if history["valid_mrr"]:
        epochs = [20*(i+1) for i in range(len(history["valid_mrr"]))]
        axes[1].plot(epochs, history["valid_mrr"], "o-", color=CB[2], linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MRR")
        axes[1].set_title("Validation MRR")
        axes[1].set_ylim(0, 1)

    plt.suptitle("TComplEx Training Dynamics")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, "kg_training_curve.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()
