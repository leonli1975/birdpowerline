"""Temporal KG reasoning: link prediction and measure recommendation."""

import numpy as np


def predict_threat(model, bird_id, tower_id, season_idx, e2i, i2e, i2r, r2i, threshold=0.5):
    """Predict whether a bird threatens a tower in a given season.

    Returns: (probability, relation_rank, top_relations)
    """
    r_names = ["threatens", "nests_on", "causes", "present_in"]
    r_ids = [r2i[r] for r in r_names if r in r2i]
    if not r_ids:
        return None

    scores = {}
    for rid in r_ids:
        s = model.score(e2i[bird_id], rid, e2i[tower_id], season_idx)
        scores[i2r[rid]] = model._sigmoid(s)

    return scores


def recommend_measures(model, bird_id, fault_id, season_idx,
                       e2i, i2e, i2r, r2i, kg, top_k=5):
    """Given a bird-fault pair in a season, recommend the best measures.

    Reasoning chain:
      1. Which measures mitigate this fault type?
      2. Which measures are already deployed in this season?
      3. Which undeployed measure would give the highest effectiveness score?

    Returns: sorted list of (measure, score, is_deployed)
    """
    if "mitigates" not in r2i:
        return []

    mitigates_r = r2i["mitigates"]

    # Find all measures
    measure_ids = []
    for eid in range(len(i2e)):
        name = i2e[eid]
        if name.startswith(("anti_bird","artificial","smart","hd_video",
                            "dynamic","rotating","insulation","acoustic")):
            measure_ids.append(eid)

    # Score each measure for mitigating this fault
    scores = []
    deployed_on_r = r2i.get("deployed_on")

    for mid in measure_ids:
        measure_name = i2e[mid]
        s = model.score(mid, mitigates_r, fault_id, season_idx)
        score = model._sigmoid(s)

        # Check if already deployed
        is_deployed = False
        if deployed_on_r is not None and "deployed_on" in r2i:
            # Check if there exists a line where this measure is deployed at this time
            for triple in kg:
                if (triple[0] == measure_name and triple[1] == "deployed_on" and
                        triple[3] is not None and triple[3] <= season_idx and
                        (triple[4] is None or triple[4] >= season_idx)):
                    is_deployed = True
                    break

        scores.append((measure_name, float(score), is_deployed))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def predict_future_threats(model, e2i, i2e, r2i, bird_tower_map,
                           future_season, thr_threshold=0.3):
    """Predict new (unseen) bird-tower threats for a future season.

    bird_tower_map: list of (bird_id, tower_id, current_threat_bool)

    Returns: list of (bird, tower, threat_score) for new threats
    """
    if "threatens" not in r2i:
        return []

    threat_r = r2i["threatens"]
    future_idx = min(int(future_season), model.n_timestamps - 1)

    existing = set()
    for b, tw, _ in bird_tower_map:
        existing.add((b, tw))

    predictions = []
    # Get all bird and tower IDs from the entity dict
    bird_ids = [e2i[k] for k in e2i if k in _BIRD_IDS]
    tower_ids = [e2i[k] for k in e2i if k in _TOWER_IDS]

    for bid in bird_ids:
        for twid in tower_ids:
            if (bid, twid) in existing:
                continue
            s = model.score(bid, threat_r, twid, future_idx)
            score = model._sigmoid(s)
            if score > thr_threshold:
                predictions.append((i2e[bid], i2e[twid], float(score)))

    predictions.sort(key=lambda x: -x[2])
    return predictions[:20]


# Pre-filtered sets for efficient lookup
_BIRD_IDS = {
    "Gr_nigricollis","Aq_chrysaetos","Fa_cherrug","Bu_hemilasius",
    "Fa_tinnunculus","Mi_migrans","Gy_himalayensis","Co_corax",
    "Pi_pica","An_noctua","An_anser","Ta_ferruginea","Aq_nipalensis",
    "St_ciconia","Pl_leucorodia","An_formosa","Ch_dubius",
    "La_brunnicephalus","An_crecca","Gr_grus",
}
_TOWER_IDS = {
    "line_ruozhen","line_ruotang","line_anmai","line_aji",
    "line_heishang","line_heize","line_heimai","line_ruoa",
    "line_ehu","line_manqiong","line_manka",
}
