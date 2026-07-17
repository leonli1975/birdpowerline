"""Analysis functions: centrality, community detection, temporal evolution."""

import numpy as np


def analyze_all_seasons(model):
    """Run comprehensive analysis across all seasons."""
    results = {}

    # Summary per season
    results["season_summaries"] = [model.summary(s) for s in model.seasons]

    # Temporal centrality dynamics
    results["centrality_dynamics"] = model.temporal_centrality_dynamics()

    # Degree centrality per season
    results["degree_centrality"] = {s: model.degree_centrality(s) for s in model.seasons}

    # Eigenvector centrality per season
    results["eigenvector_centrality"] = {s: model.eigenvector_centrality(s) for s in model.seasons}

    # Betweenness centrality per season
    results["betweenness_centrality"] = {s: model.betweenness_centrality(s) for s in model.seasons}

    # Critical nodes per season
    results["critical_nodes"] = {s: model.identify_critical_nodes(s) for s in model.seasons}

    # Community detection per season
    results["communities"] = {}
    for s in model.seasons:
        print(f"  Running community detection for {s}...")
        results["communities"][s] = model.louvain_communities(s)

    # Cross-layer coupling per season
    results["cross_layer_coupling"] = {
        s: model.cross_layer_coupling_strength(s) for s in model.seasons
    }

    # Density per season
    results["density"] = {s: model.density(s) for s in model.seasons}

    return results


def compute_risk_ranking(model, season, alpha=0.5, beta=0.3, gamma=0.2):
    """Compute composite risk ranking for towers.

    Risk(tower) = alpha * degree_centrality + beta * bird_layer_coupling + 
                  gamma * vulnerability (low measure coverage)
    """
    dc = model.degree_centrality(season)
    inter = model.get_inter_layer_matrix(season, "bird", "tower")

    # Bird coupling: sum of incoming bird conflict risks
    bird_coupling = inter.sum(axis=0)  # sum over birds for each tower
    bird_coupling_norm = bird_coupling / (bird_coupling.max() + 1e-12)

    # Measure coverage per tower
    tower_measure = model.get_inter_layer_matrix(season, "tower", "measure")
    measure_coverage = tower_measure.sum(axis=1)
    vulnerability = 1.0 - measure_coverage / (measure_coverage.max() + 1e-12)

    scores = {}
    for t in model.tower_ids:
        i = model.tower_idx[t]
        scores[t] = round(alpha * dc["tower"][t] + beta * bird_coupling_norm[i] +
                          gamma * vulnerability[i], 4)

    return dict(sorted(scores.items(), key=lambda x: -x[1]))


def temporal_stability(model):
    """Assess temporal stability of the network structure.

    Measures:
      - Node degree variance across seasons
      - Jaccard similarity of top-k central nodes between consecutive seasons
    """
    dynamics = model.temporal_centrality_dynamics()

    # Variance of degree centrality across seasons
    variance = {}
    for layer in ["bird", "tower", "measure"]:
        variance[layer] = {}
        for nid, season_vals in dynamics[layer].items():
            vals = list(season_vals.values())
            variance[layer][nid] = round(float(np.var(vals)), 6)

    # Top-k Jaccard between consecutive seasons
    jaccard_topk = {}
    for k in [3, 5, 10]:
        jaccard_topk[f"top{k}"] = []
        for i in range(len(model.seasons)):
            s1 = model.seasons[i]
            s2 = model.seasons[(i + 1) % len(model.seasons)]
            dc1 = model.degree_centrality(s1)
            dc2 = model.degree_centrality(s2)
            top1 = set(dict(sorted(dc1["bird"].items(),
                                   key=lambda x: -x[1])[:k]).keys())
            top2 = set(dict(sorted(dc2["bird"].items(),
                                   key=lambda x: -x[1])[:k]).keys())
            j = len(top1 & top2) / len(top1 | top2) if (top1 | top2) else 0
            jaccard_topk[f"top{k}"].append(round(j, 4))

    return {
        "degree_variance": variance,
        "jaccard_similarity": jaccard_topk,
    }


def generate_paper_stats(data, model, results):
    """Generate key statistics suitable for paper tables."""
    stats = {}

    # Table 1: Network overview
    stats["network_overview"] = {
        "n_bird_species": model.n_birds,
        "n_towers": model.n_towers,
        "n_measures": model.n_measures,
        "seasons": model.seasons,
        "density_by_season": results["density"],
        "cross_layer_coupling_by_season": results["cross_layer_coupling"],
    }

    # Table 2: Key conflict bird species (summer peak)
    summer_dc = results["degree_centrality"]["summer"]
    stats["key_conflict_birds"] = sorted(summer_dc["bird"].items(),
                                         key=lambda x: -x[1])[:5]

    # Table 3: Most vulnerable towers
    risk = compute_risk_ranking(model, "summer")
    stats["vulnerable_towers"] = list(risk.items())[:8]

    # Temporal network metrics
    st = temporal_stability(model)
    stats["temporal_stability"] = st

    # Community structure summary
    stats["community_summary"] = {}
    for s in model.seasons:
        c = results["communities"][s]
        stats["community_summary"][s] = {
            "n_communities": c["n_communities"],
            "cross_layer_n": c["cross_layer_communities"],
            "modularity": c["modularity"],
        }

    return stats
