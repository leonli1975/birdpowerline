#!/usr/bin/env python3
"""Temporal Knowledge Graph Reasoning for Bird-Powerline Conflict.

Direction 1 from rptask.md:
"时空异质知识图谱驱动的鸟类-线路冲突风险推理"

Uses year-level temporal granularity (2017-2024).
Temporal split: train 2017-2021, val 2022, test 2023-2024.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.kg_data import (
    build_temporal_kg, enrich_kg, build_id_mappings, build_tensor_data,
    BIRD_SPECIES, POWER_LINES, MEASURES, FAULT_TYPES, TIMESTAMPS, N_TIMESTAMPS,
    RAPTOR_ELEVATION, ELEVATION_BANDS,
)
from src.tkg_embedding import TComplEx
from src.tkg_reasoning import recommend_measures
from src.kg_viz import (
    plot_link_prediction_metrics, plot_threat_evolution,
    plot_measure_scores, plot_training_curve,
)


def print_sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def temporal_split(triples, train_years=(2017, 2021), val_year=2022, test_years=(2023, 2024)):
    """Split triples by year. train_years, val_year, test_years are calendar years."""
    triples_arr = np.array(triples, dtype=np.int64)
    train = []
    val = []
    test = []

    train_lo = train_years[0] - 2017
    train_hi = train_years[1] - 2017
    val_t = val_year - 2017
    test_lo = test_years[0] - 2017
    test_hi = test_years[1] - 2017

    for i in range(len(triples)):
        h, r, t, tau = triples_arr[i]
        triple = (int(h), int(r), int(t), int(tau))
        if train_lo <= tau <= train_hi:
            train.append(triple)
        elif tau == val_t:
            val.append(triple)
        elif test_lo <= tau <= test_hi:
            test.append(triple)

    return train, val, test


def main():
    # Step 1: Build temporal KG
    print_sep("Step 1: Building Base Temporal Knowledge Graph")
    kg = build_temporal_kg()
    print(f"  Base KG: {len(kg)} triples")

    # Step 1b: Enrich with literature data
    print_sep("Step 1b: Enriching KG from Literature (S1/S2/S3)")
    kg = enrich_kg(kg)
    print(f"  Enriched KG: {len(kg)} triples")
    n_elev = len(RAPTOR_ELEVATION)
    n_bands = len(ELEVATION_BANDS)
    print(f"  Added: {n_elev} raptor elevation profiles + "
          f"{n_bands} elevation bands + flyway context")
    e2i, r2i, i2e, i2r = build_id_mappings(kg)
    triples, _, _ = build_tensor_data(kg, e2i, r2i)

    n_entities = len(e2i)
    n_relations = len(r2i)
    print(f"  Entities: {n_entities}  Relations: {n_relations}  "
          f"Timestamps: {N_TIMESTAMPS} ({TIMESTAMPS[0]}-{TIMESTAMPS[-1]})  "
          f"Triples: {len(triples)}")

    # Step 2: KG Summary
    print_sep("Step 2: KG Summary")
    print(f"  Bird species: {len(BIRD_SPECIES)}")
    print(f"  Power lines:  {len(POWER_LINES)}")
    print(f"  Measures:     {len(MEASURES)}")
    print(f"  Fault types:  {len(FAULT_TYPES)}")

    # Step 3: Temporal split
    print_sep("Step 3: Temporal Train/Val/Test Split")
    train, val, test = temporal_split(triples)
    print(f"  Train (2017-2021): {len(train)}  "
          f"Val (2022): {len(val)}  "
          f"Test (2023-2024): {len(test)}")

    # Step 4: Train TComplEx
    print_sep("Step 4: Training TComplEx (dim=128, AdaGrad, margin loss)")
    model = TComplEx(n_entities, n_relations, N_TIMESTAMPS, dim=128,
                     reg_lambda=0.001, lr=0.1, negative_samples=10)
    history = model.fit(train, epochs=300, batch_size=512, verbose=True,
                        valid_triples=val, margin=1.0)

    # Step 5: Evaluate link prediction
    print_sep("Step 5: Link Prediction Evaluation (Test: 2023-2024)")
    train_filter = set(train)
    if len(test) > 0:
        metrics = model.evaluate_link_prediction(test, train_filter)
        print(f"  MRR:     {metrics['MRR']:.4f}")
        print(f"  Hits@1:  {metrics['Hits@1']:.4f}")
        print(f"  Hits@3:  {metrics['Hits@3']:.4f}")
        print(f"  Hits@10: {metrics['Hits@10']:.4f}")
        print(f"  MeanRank:{metrics['MeanRank']:.1f}")
    else:
        print("  No test triples for 2023-2024")
        metrics = {"MRR": 0, "Hits@1": 0, "Hits@3": 0, "Hits@10": 0, "MeanRank": 0}

    # Step 6: Predict future threats (2023-2024)
    print_sep("Step 6: Predicted Threats for 2023-2024")
    threat_r = r2i.get("threatens")
    if threat_r is not None:
        # Find all bird-tower triples NOT in train (novel predictions)
        train_threats = set()
        for h, r, t, tau in train:
            if r == threat_r:
                train_threats.add((i2e[h], i2e[t]))

        bird_ids = [e2i[k] for k in e2i if k in BIRD_SPECIES]
        tower_ids = [e2i[k] for k in e2i if k in POWER_LINES]

        predictions = []
        for bid in bird_ids:
            for twid in tower_ids:
                if (i2e[bid], i2e[twid]) in train_threats:
                    continue
                # Score for 2023 (tau=6) and 2024 (tau=7)
                s23 = model.score(bid, threat_r, twid, 6)
                s24 = model.score(bid, threat_r, twid, 7)
                avg_prob = model._sigmoid((s23 + s24) / 2)
                if avg_prob > 0.3:
                    predictions.append((i2e[bid], i2e[twid], float(avg_prob)))

        predictions.sort(key=lambda x: -x[2])
        print(f"\n  Top 10 novel threat predictions:")
        for i, (bird, tower, prob) in enumerate(predictions[:10]):
            bird_cn = BIRD_SPECIES.get(bird, {}).get("name_cn", bird)
            tower_cn = POWER_LINES.get(tower, {}).get("name_cn", tower)
            print(f"  {i+1:2d}. {bird_cn}({bird}) ⇢ {tower_cn}({tower})  "
                  f"p={prob:.4f}")

    # Step 7: Measure recommendation
    print_sep("Step 7: Measure Recommendation for 2024")
    if "mitigates" in r2i and "fault_nest_shorting" in e2i:
        fault_id = e2i["fault_nest_shorting"]
        bird_id = e2i.get("Bu_hemilasius")
        future_tau = 7  # 2024
        if bird_id is not None:
            recs = recommend_measures(
                model, bird_id, fault_id, future_tau,
                e2i, i2e, i2r, r2i, kg, top_k=5
            )
            print("\n  For 大鵟(Bu_hemilasius) → nest shorting in 2024:")
            for mname, score, deployed in recs:
                mcn = MEASURES.get(mname, {}).get("name_cn", mname)
                status = "[existing]" if deployed else "[recommend NEW]"
                print(f"    {mcn:10s} ({mname})  score={score:.4f}  {status}")

    # Step 8: Temporal threat evolution
    print_sep("Step 8: Temporal Threat Evolution")
    if "Bu_hemilasius" in e2i and "line_ruozhen" in e2i and threat_r is not None:
        b_id = e2i["Bu_hemilasius"]
        t_id = e2i["line_ruozhen"]
        print("  大鵟 → 若真线 threat probability by year:")
        for yi, year in enumerate(TIMESTAMPS):
            s = model.score(b_id, threat_r, t_id, yi)
            prob = model._sigmoid(s)
            marker = " [measures deployed 2020+; comprehensive 2023+]" if year >= 2020 else ""
            print(f"    {year}: p={prob:.4f}{marker}")

    # Step 9: Save
    print_sep("Step 9: Saving Results")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "kg_stats": {
            "n_entities": n_entities, "n_relations": n_relations,
            "n_timestamps": N_TIMESTAMPS, "n_triples": len(triples),
            "n_bird_species": len(BIRD_SPECIES),
            "n_power_lines": len(POWER_LINES), "n_measures": len(MEASURES),
        },
        "temporal_split": {"train": len(train), "val": len(val), "test": len(test)},
        "link_prediction": metrics,
        "method": "TComplEx", "dim": 128,
        "top_predicted_threats": [
            {"bird": b, "line": tw, "probability": p}
            for b, tw, p in predictions[:10]
        ],
        "final_loss": float(history["loss"][-1]),
        "best_valid_mrr": float(max(history.get("valid_mrr", [0]))),
    }

    with open(os.path.join(output_dir, "kg_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to output/kg_results.json")

    # Step 10: Visualizations
    print_sep("Step 10: Generating Visualizations")
    plot_link_prediction_metrics(metrics)
    plot_threat_evolution(model, e2i, i2e, r2i, TIMESTAMPS)
    plot_training_curve(history)
    print(f"  Plots saved to output/")

    print_sep("Done")
    print(f"  Link prediction MRR: {metrics.get('MRR', 0):.4f}")
    print(f"  Temporal split: train 2017-2021, val 2022, test 2023-2024")


if __name__ == "__main__":
    main()
