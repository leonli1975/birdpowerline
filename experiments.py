#!/usr/bin/env python3
"""Ablation studies and baseline comparisons — efficient version."""

import json, os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.kg_data import (
    build_temporal_kg, enrich_kg, build_id_mappings, build_tensor_data,
    TIMESTAMPS, N_TIMESTAMPS, BIRD_SPECIES, POWER_LINES, MEASURES,
)
from src.tkg_embedding import TComplEx
from src.baselines import ComplEx, DistMult


def temporal_split(triples, train_end=2021, val_year=2022, test_start=2023):
    triples_arr = np.array(triples, dtype=np.int64)
    train, val, test = [], [], []
    train_hi = train_end - 2017; val_t = val_year - 2017
    test_lo = test_start - 2017; test_hi = 2024 - 2017
    for i in range(len(triples)):
        h, r, t, tau = triples_arr[i]
        triple = (int(h), int(r), int(t), int(tau))
        if tau <= train_hi: train.append(triple)
        elif tau == val_t: val.append(triple)
        elif test_lo <= tau <= test_hi: test.append(triple)
    return train, val, test


def run_one(cls, train, test, n_e, n_r, n_t, train_filter, epochs=50, dim=128, **kw):
    t0 = time.time()
    m = cls(n_e, n_r, n_t, dim=dim, lr=0.1, negative_samples=kw.pop("neg", 10),
            reg_lambda=0.001)
    m.fit(train, epochs=epochs, batch_size=512, margin=kw.pop("margin", 1.0), verbose=False)
    dt = time.time() - t0
    metrics = m.evaluate_link_prediction(test, train_filter)
    return {**metrics, "time_s": round(dt, 1)}


def entity_type_mrr(model, test_triples, train_filter, i2e):
    bird_ids = {eid for eid, name in i2e.items() if name in BIRD_SPECIES}
    line_ids = {eid for eid, name in i2e.items() if name in POWER_LINES}
    measure_ids = {eid for eid, name in i2e.items() if name in MEASURES}
    triples = np.array(test_triples, dtype=np.int64)
    results = {}
    for tn, ts in [("Bird", bird_ids), ("Line", line_ids), ("Measure", measure_ids)]:
        subset = [t for t in triples if int(t[2]) in ts]
        if subset:
            m = model.evaluate_link_prediction(
                [(int(a),int(b),int(c),int(d)) for a,b,c,d in subset], train_filter)
            results[tn] = {"MRR": m["MRR"], "n": len(subset)}
    return results


def main():
    print("="*60)
    print("  Ablation Studies")
    print("="*60)

    kg = build_temporal_kg(); kg = enrich_kg(kg)
    e2i, r2i, i2e, i2r = build_id_mappings(kg)
    triples, _, _ = build_tensor_data(kg, e2i, r2i)
    n_e, n_r, n_t = len(e2i), len(r2i), N_TIMESTAMPS
    train, val, test = temporal_split(triples)
    train_filter = set(train)
    print(f"  KG: {n_e}e/{n_r}r  train={len(train)} val={len(val)} test={len(test)}")

    results = {}

    # --- 1. BASELINE COMPARISON ---
    print("\n--- 1. Baseline Comparison (temporal split) ---")
    baseline = {}
    for name, cls in [("TComplEx", TComplEx), ("ComplEx", ComplEx), ("DistMult", DistMult)]:
        r = run_one(cls, train, test, n_e, n_r, n_t, train_filter, epochs=50)
        baseline[name] = r
        print(f"  {name:12s}: MRR={r['MRR']:.4f} H@10={r['Hits@10']:.4f} t={r['time_s']}s")
    results["baselines"] = baseline

    # --- 2. ABLATION: Dimension ---
    print("\n--- 2. Ablation: Embedding Dimension ---")
    dims = {}
    for d in [32, 64, 128, 256]:
        r = run_one(TComplEx, train, test, n_e, n_r, n_t, train_filter, epochs=30, dim=d)
        dims[f"d={d}"] = r
        print(f"  d={d:3d}: MRR={r['MRR']:.4f} H@10={r['Hits@10']:.4f}")
    results["ablation_dim"] = dims

    # --- 3. ABLATION: Negative Samples ---
    print("\n--- 3. Ablation: Negative Samples ---")
    negs = {}
    for neg in [5, 10, 20]:
        r = run_one(TComplEx, train, test, n_e, n_r, n_t, train_filter, epochs=30, neg=neg)
        negs[f"neg={neg}"] = r
        print(f"  neg={neg:2d}: MRR={r['MRR']:.4f} H@10={r['Hits@10']:.4f}")
    results["ablation_neg"] = negs

    # --- 4. ABLATION: Margin ---
    print("\n--- 4. Ablation: Margin ---")
    margins = {}
    for m in [0.5, 1.0, 2.0]:
        r = run_one(TComplEx, train, test, n_e, n_r, n_t, train_filter, epochs=30, margin=m)
        margins[f"margin={m}"] = r
        print(f"  margin={m:.1f}: MRR={r['MRR']:.4f} H@10={r['Hits@10']:.4f}")
    results["ablation_margin"] = margins

    # --- 5. TEMPORAL vs STATIC (w/ more epochs for significance) ---
    print("\n--- 5. Temporal (TComplEx) vs Static (ComplEx) at 100 epochs ---")
    r_tkge = run_one(TComplEx, train, test, n_e, n_r, n_t, train_filter, epochs=100)
    r_cplx = run_one(ComplEx, train, test, n_e, n_r, n_t, train_filter, epochs=100)
    results["temporal_vs_static"] = {
        "TComplEx_100ep": r_tkge, "ComplEx_100ep": r_cplx
    }
    print(f"  TComplEx 100ep: MRR={r_tkge['MRR']:.4f} H@10={r_tkge['Hits@10']:.4f}")
    print(f"  ComplEx  100ep: MRR={r_cplx['MRR']:.4f} H@10={r_cplx['Hits@10']:.4f}")
    delta = r_tkge['MRR'] - r_cplx['MRR']
    print(f"  Delta (TComplEx - ComplEx): {delta:+.4f}")

    # --- 6. BASE KG vs ENRICHED KG ---
    print("\n--- 6. Base KG vs Enriched KG (TComplEx, 50ep) ---")
    kg_b = build_temporal_kg()
    e2i_b, r2i_b, i2e_b, i2r_b = build_id_mappings(kg_b)
    triples_b, _, _ = build_tensor_data(kg_b, e2i_b, r2i_b)
    train_b, val_b, test_b = temporal_split(triples_b)
    train_filter_b = set(train_b)
    r_base = run_one(TComplEx, train_b, test_b, len(e2i_b), len(r2i_b), n_t,
                     train_filter_b, epochs=50)
    r_enriched = baseline["TComplEx"]  # already computed
    results["base_vs_enriched"] = {"base_kg": r_base, "enriched_kg": r_enriched}
    print(f"  Base KG:     MRR={r_base['MRR']:.4f} H@10={r_base['Hits@10']:.4f}")
    print(f"  Enriched KG: MRR={r_enriched['MRR']:.4f} H@10={r_enriched['Hits@10']:.4f}")

    # --- 7. ENTITY-TYPE BREAKDOWN ---
    print("\n--- 7. Per-Entity-Type MRR ---")
    best_model = TComplEx(n_e, n_r, n_t, dim=128, lr=0.1, negative_samples=10, reg_lambda=0.001)
    best_model.fit(train, epochs=100, batch_size=512, margin=1.0, verbose=False)
    etype = entity_type_mrr(best_model, test, train_filter, i2e)
    for tn, m in etype.items():
        print(f"  {tn}: MRR={m['MRR']:.4f} (n={m['n']})")
    results["entity_type_mrr"] = etype

    # --- SAVE ---
    out = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "experiments.json"), "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  Saved to output/experiments.json")

    # --- SUMMARY ---
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  {'Model/Config':20s} {'MRR':>8s} {'Hits@1':>8s} {'Hits@10':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for name, r in baseline.items():
        print(f"  {name:20s} {r['MRR']:8.4f} {r['Hits@1']:8.4f} {r['Hits@10']:8.4f}")


if __name__ == "__main__":
    main()
