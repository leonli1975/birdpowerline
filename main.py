#!/usr/bin/env python3
"""Multi-layer temporal network analysis for bird-powerline-measure coupled system.

Direction 2 from rptask.md:
"基于多层时序网络的'鸟-线-措施'耦合系统建模"

Three-layer temporal network:
  L1: Bird distribution layer (seasonal)
  L2: Power line / tower layer
  L3: Mitigation measure layer

Cross-layer edges capture conflict risk, measure effectiveness, and feedback.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import generate_all_data
from src.multilayer import MultiLayerTemporalNetwork
from src.analysis import (
    analyze_all_seasons, compute_risk_ranking,
    temporal_stability, generate_paper_stats,
)
from src.viz import generate_all_plots


def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # Step 1: Generate data
    print_separator("Step 1: Generating Synthetic Data (Aba Grassland Scenario)")
    data = generate_all_data(n_towers=30, n_measures=18)
    print(f"  Bird species: {len(data['bird_layer']['nodes'])}")
    print(f"  Towers:       {len(data['tower_layer']['nodes'])}")
    print(f"  Measures:     {len(data['measure_layer']['nodes'])}")
    print(f"  Seasons:      {data['seasons']}")

    # Step 2: Build multi-layer temporal network
    print_separator("Step 2: Building Multi-Layer Temporal Network")
    model = MultiLayerTemporalNetwork(data)
    print(f"  Total nodes: {model.n_total} (birds={model.n_birds}, "
          f"towers={model.n_towers}, measures={model.n_measures})")

    # Step 3: Network overview
    print_separator("Step 3: Network Overview by Season")
    for s in model.seasons:
        sm = model.summary(s)
        print(f"  [{s.capitalize()}]  density={sm['density']:.4f}  "
              f"edges={sm['total_edges']}  coupling={sm['cross_layer_coupling']:.3f}")

    # Step 4: Full analysis
    print_separator("Step 4: Running Comprehensive Analysis")
    results = analyze_all_seasons(model)

    # Step 5: Critical conflict nodes
    print_separator("Step 5: Critical Conflict Nodes (Summer)")
    summer_critical = model.identify_critical_nodes("summer", top_k=8)
    print("  Top conflict bird species:")
    for b, score in summer_critical["top_birds"]:
        print(f"    {b:22s}  degree={score:.4f}")
    print("  Most vulnerable towers:")
    for t, score in summer_critical["top_towers"]:
        print(f"    {t}  degree={score:.4f}")

    # Step 6: Risk ranking
    print_separator("Step 6: Tower Risk Ranking (Summer)")
    risk = compute_risk_ranking(model, "summer")
    for t, score in list(risk.items())[:10]:
        print(f"    {t}: {score:.4f}")

    # Step 7: Temporal stability
    print_separator("Step 7: Temporal Stability Analysis")
    stability = temporal_stability(model)
    for k, vals in stability["jaccard_similarity"].items():
        print(f"  {k} Jaccard: {vals}  (mean={sum(vals)/len(vals):.3f})")

    # Step 8: Community detection summary
    print_separator("Step 8: Community Detection Summary")
    for s in model.seasons:
        c = results["communities"][s]
        print(f"  [{s.capitalize()}]  communities={c['n_communities']}  "
              f"cross-layer={c['cross_layer_communities']}  "
              f"modularity={c['modularity']:.4f}")

    # Step 9: Paper statistics
    print_separator("Step 9: Paper-Ready Statistics")
    stats = generate_paper_stats(data, model, results)

    # Save stats
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "analysis_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str, ensure_ascii=False)
    print(f"  Stats saved to output/analysis_stats.json")

    # Step 10: Visualizations
    print_separator("Step 10: Generating Visualizations")
    generate_all_plots(model, results, stability)

    print_separator("Done")
    print("  Output files are in output/")
    print("  - output/analysis_stats.json   : paper-ready statistics")
    print("  - output/supra_adj_*.png        : supral adjacency matrices")
    print("  - output/centrality_*.png        : centrality comparisons")
    print("  - output/cross_layer_*.png       : cross-layer coupling")
    print("  - output/risk_heatmap.png        : tower risk across seasons")
    print("  - output/community_summary.png   : community structure")
    print("  - output/temporal_jaccard.png    : temporal stability")


if __name__ == "__main__":
    main()
