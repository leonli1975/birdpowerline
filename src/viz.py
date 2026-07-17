"""Visualization for multi-layer temporal network analysis."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

# Use CJK font if available
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

import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

# Color-blind friendly palette
CB_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
             "#D55E00", "#F0E442", "#000000"]


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_supra_adjacency(model, season, save=True):
    """Plot the supral adjacency matrix for a given season."""
    ensure_output_dir()
    A = model.get_snapshot(season)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(A, cmap="YlOrRd", aspect="auto", interpolation="nearest")

    # Layer boundaries
    nB, nT = model.n_birds, model.n_towers
    n_total = model.n_total
    ax.axhline(y=nB - 0.5, color="black", linewidth=1.5)
    ax.axhline(y=nB + nT - 0.5, color="black", linewidth=1.5)
    ax.axvline(x=nB - 0.5, color="black", linewidth=1.5)
    ax.axvline(x=nB + nT - 0.5, color="black", linewidth=1.5)

    # Labels
    mid_b = nB / 2
    mid_t = nB + nT / 2
    mid_m = nB + nT + model.n_measures / 2
    ax.text(mid_b, -2, "Birds", ha="center", fontsize=12, fontweight="bold")
    ax.text(mid_t, -2, "Towers", ha="center", fontsize=12, fontweight="bold")
    ax.text(mid_m, -2, "Measures", ha="center", fontsize=12, fontweight="bold")
    ax.text(-2.5, mid_b, "Birds", va="center", rotation=90, fontsize=12, fontweight="bold")
    ax.text(-2.5, mid_t, "Towers", va="center", rotation=90, fontsize=12, fontweight="bold")
    ax.text(-2.5, mid_m, "Measures", va="center", rotation=90, fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Edge Weight")
    ax.set_title(f"Supra-adjacency Matrix — {season.capitalize()}", fontsize=14)
    plt.tight_layout()
    if save:
        fp = os.path.join(OUTPUT_DIR, f"supra_adj_{season}.png")
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def plot_centrality_comparison(model, results, save=True):
    """Bar chart comparing degree centrality across seasons for top nodes."""
    ensure_output_dir()
    dc = results["degree_centrality"]
    layers = ["bird", "tower", "measure"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, layer in zip(axes, layers):
        # Collect all nodes' seasonal data
        nodes = getattr(model, f"{layer}_ids")
        data = {n: [dc[s][layer].get(n, 0) for s in model.seasons] for n in nodes}
        # Pick top 6 nodes by summer degree
        sorted_nodes = sorted(data.keys(),
                              key=lambda n: data[n][model.seasons.index("summer")],
                              reverse=True)[:6]

        x = np.arange(len(model.seasons))
        width = 0.12
        for i, n in enumerate(sorted_nodes):
            vals = data[n]
            ax.bar(x + i * width, vals, width,
                   label=model.data[f"{layer}_layer"]["nodes"].get(n, {}).get("name_cn", n),
                   color=CB_COLORS[i % len(CB_COLORS)])
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels([s.capitalize() for s in model.seasons])
        ax.set_title(f"{layer.capitalize()} Layer Degree Centrality")
        ax.set_ylabel("Degree Centrality")
        ax.legend(fontsize=7, loc="upper right")

    plt.suptitle("Seasonal Degree Centrality Dynamics", fontsize=14)
    plt.tight_layout()
    if save:
        fp = os.path.join(OUTPUT_DIR, "centrality_comparison.png")
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def plot_cross_layer_coupling(model, results, save=True):
    """Plot cross-layer coupling strength across seasons."""
    ensure_output_dir()
    coupling = results["cross_layer_coupling"]
    fig, ax = plt.subplots(figsize=(7, 4))
    seasons = model.seasons
    vals = [coupling[s] for s in seasons]
    ax.bar(range(len(seasons)), vals, color=CB_COLORS[:len(seasons)])
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels([s.capitalize() for s in seasons])
    ax.set_ylabel("Cross-layer Coupling Strength")
    ax.set_title("Cross-layer Coupling Strength Across Seasons")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    if save:
        fp = os.path.join(OUTPUT_DIR, "cross_layer_coupling.png")
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def plot_risk_heatmap(model, save=True):
    """Plot tower risk heatmap across seasons."""
    ensure_output_dir()
    from src.analysis import compute_risk_ranking

    data = np.zeros((model.n_towers, len(model.seasons)))
    for j, s in enumerate(model.seasons):
        risk = compute_risk_ranking(model, s)
        for t, score in risk.items():
            i = model.tower_idx[t]
            data[i, j] = score

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    # Annotate zones
    for i, tid in enumerate(model.tower_ids):
        zone = model.data["tower_layer"]["nodes"][tid]["zone"]
        color = {"zone_A": "blue", "zone_B": "green",
                 "zone_C": "purple", "zone_D": "orange"}[zone]
        ax.text(-0.5, i, tid, fontsize=7, ha="right", va="center", color=color)

    ax.set_xticks(range(len(model.seasons)))
    ax.set_xticklabels([s.capitalize() for s in model.seasons])
    ax.set_ylabel("Tower")
    ax.set_title("Tower Risk Heatmap Across Seasons")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Risk Score")
    plt.tight_layout()
    if save:
        fp = os.path.join(OUTPUT_DIR, "risk_heatmap.png")
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def plot_community_summary(model, results, save=True):
    """Plot community detection summary across seasons."""
    ensure_output_dir()
    comm = results["communities"]
    seasons = model.seasons

    n_comm = [comm[s]["n_communities"] for s in seasons]
    n_cross = [comm[s]["cross_layer_communities"] for s in seasons]
    modularity = [comm[s]["modularity"] for s in seasons]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(seasons, n_comm, color=CB_COLORS[:len(seasons)])
    axes[0].set_title("Number of Communities")
    axes[0].set_ylabel("Count")

    axes[1].bar(seasons, n_cross, color=CB_COLORS[:len(seasons)])
    axes[1].set_title("Cross-layer Communities")
    axes[1].set_ylabel("Count")

    axes[2].bar(seasons, modularity, color=CB_COLORS[:len(seasons)])
    axes[2].set_title("Modularity")
    axes[2].set_ylabel("Q")

    plt.suptitle("Community Structure Across Seasons", fontsize=14)
    plt.tight_layout()
    if save:
        fp = os.path.join(OUTPUT_DIR, "community_summary.png")
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def plot_temporal_jaccard(model, stability, save=True):
    """Plot Jaccard similarity of top-k nodes between consecutive seasons."""
    ensure_output_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, vals in stability["jaccard_similarity"].items():
        ax.plot(model.seasons, vals, "o-", label=k, linewidth=2)
    ax.set_xlabel("Season")
    ax.set_ylabel("Jaccard Similarity")
    ax.set_title("Temporal Stability: Top-k Node Consistency Between Seasons")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save:
        fp = os.path.join(OUTPUT_DIR, "temporal_jaccard.png")
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def generate_all_plots(model, results, stability):
    """Generate all visualization outputs."""
    ensure_output_dir()
    print("Generating plots...")
    plot_supra_adjacency(model, "spring")
    plot_supra_adjacency(model, "summer")
    plot_supra_adjacency(model, "autumn")
    plot_supra_adjacency(model, "winter")
    plot_centrality_comparison(model, results)
    plot_cross_layer_coupling(model, results)
    plot_risk_heatmap(model)
    plot_community_summary(model, results)
    plot_temporal_jaccard(model, stability)
    print(f"All plots saved to {OUTPUT_DIR}/")
