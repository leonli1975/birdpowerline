"""Synthetic data generation for Aba grassland bird-powerline scenario."""

import numpy as np

SEASONS = ["spring", "summer", "autumn", "winter"]
SEASON_MONTHS = {"spring": [3, 4, 5], "summer": [6, 7, 8],
                 "autumn": [9, 10, 11], "winter": [12, 1, 2]}

BIRD_SPECIES = {
    "black_necked_crane": {
        "name_cn": "黑颈鹤", "danger_rank": 0.95, "conservation": "VU",
        "abundance": 80,
        "seasonal_presence": {"spring": 0.3, "summer": 0.9, "autumn": 0.5, "winter": 0.1},
        "habitat_type": "wetland",
    },
    "golden_eagle": {
        "name_cn": "金雕", "danger_rank": 0.85, "conservation": "NT",
        "abundance": 40,
        "seasonal_presence": {"spring": 0.8, "summer": 0.7, "autumn": 0.9, "winter": 0.6},
        "habitat_type": "mountain",
    },
    "saker_falcon": {
        "name_cn": "猎隼", "danger_rank": 0.80, "conservation": "EN",
        "abundance": 35,
        "seasonal_presence": {"spring": 0.9, "summer": 0.6, "autumn": 0.8, "winter": 0.3},
        "habitat_type": "grassland",
    },
    "bar_headed_goose": {
        "name_cn": "斑头雁", "danger_rank": 0.60, "conservation": "LC",
        "abundance": 300,
        "seasonal_presence": {"spring": 0.4, "summer": 0.8, "autumn": 0.3, "winter": 0.0},
        "habitat_type": "wetland",
    },
    "ruddy_shelduck": {
        "name_cn": "赤麻鸭", "danger_rank": 0.40, "conservation": "LC",
        "abundance": 500,
        "seasonal_presence": {"spring": 0.5, "summer": 0.7, "autumn": 0.4, "winter": 0.1},
        "habitat_type": "wetland",
    },
    "upland_buzzard": {
        "name_cn": "大鵟", "danger_rank": 0.65, "conservation": "NT",
        "abundance": 120,
        "seasonal_presence": {"spring": 0.7, "summer": 0.9, "autumn": 0.6, "winter": 0.8},
        "habitat_type": "grassland",
    },
    "common_kestrel": {
        "name_cn": "红隼", "danger_rank": 0.50, "conservation": "LC",
        "abundance": 200,
        "seasonal_presence": {"spring": 0.8, "summer": 0.7, "autumn": 0.7, "winter": 0.5},
        "habitat_type": "grassland",
    },
    "black_kite": {
        "name_cn": "黑鸢", "danger_rank": 0.70, "conservation": "LC",
        "abundance": 150,
        "seasonal_presence": {"spring": 0.8, "summer": 0.9, "autumn": 0.5, "winter": 0.2},
        "habitat_type": "mountain",
    },
    "great_crested_grebe": {
        "name_cn": "凤头鸊鷉", "danger_rank": 0.35, "conservation": "LC",
        "abundance": 80,
        "seasonal_presence": {"spring": 0.6, "summer": 0.9, "autumn": 0.5, "winter": 0.1},
        "habitat_type": "wetland",
    },
    "white_eared_pheasant": {
        "name_cn": "白马鸡", "danger_rank": 0.75, "conservation": "VU",
        "abundance": 50,
        "seasonal_presence": {"spring": 0.7, "summer": 0.8, "autumn": 0.7, "winter": 0.6},
        "habitat_type": "mountain",
    },
    "lammergeier": {
        "name_cn": "胡兀鹫", "danger_rank": 0.90, "conservation": "NT",
        "abundance": 25,
        "seasonal_presence": {"spring": 0.6, "summer": 0.7, "autumn": 0.8, "winter": 0.7},
        "habitat_type": "mountain",
    },
    "himalayan_vulture": {
        "name_cn": "高山兀鹫", "danger_rank": 0.85, "conservation": "NT",
        "abundance": 30,
        "seasonal_presence": {"spring": 0.7, "summer": 0.8, "autumn": 0.7, "winter": 0.5},
        "habitat_type": "mountain",
    },
    "brown_headed_gull": {
        "name_cn": "棕头鸥", "danger_rank": 0.45, "conservation": "LC",
        "abundance": 400,
        "seasonal_presence": {"spring": 0.6, "summer": 0.8, "autumn": 0.4, "winter": 0.0},
        "habitat_type": "wetland",
    },
    "eurasian_spoonbill": {
        "name_cn": "白琵鹭", "danger_rank": 0.55, "conservation": "NT",
        "abundance": 60,
        "seasonal_presence": {"spring": 0.5, "summer": 0.7, "autumn": 0.4, "winter": 0.0},
        "habitat_type": "wetland",
    },
    "demoiselle_crane": {
        "name_cn": "蓑羽鹤", "danger_rank": 0.60, "conservation": "NT",
        "abundance": 100,
        "seasonal_presence": {"spring": 0.3, "summer": 0.8, "autumn": 0.4, "winter": 0.0},
        "habitat_type": "grassland",
    },
}

MEASURE_TYPES = {
    "acoustic_deterrent": {"name_cn": "声波驱鸟器", "type": "deterrent", "effect": 0.6},
    "visual_deterrent": {"name_cn": "视觉驱鸟器", "type": "deterrent", "effect": 0.5},
    "ultrasonic_deterrent": {"name_cn": "超声波驱鸟器", "type": "deterrent", "effect": 0.7},
    "laser_deterrent": {"name_cn": "激光驱鸟器", "type": "deterrent", "effect": 0.65},
    "decoy_predator": {"name_cn": "仿生天敌模型", "type": "deterrent", "effect": 0.4},
    "insulation_cover": {"name_cn": "绝缘防护罩", "type": "physical_barrier", "effect": 0.95},
    "perch_diverter": {"name_cn": "栖鸟挡板", "type": "physical_barrier", "effect": 0.85},
    "attractant_pond": {"name_cn": "诱导水域", "type": "attractant", "effect": 0.5},
    "attractant_feeding": {"name_cn": "诱导食源", "type": "attractant", "effect": 0.45},
    "attractant_nest": {"name_cn": "人工巢区", "type": "attractant", "effect": 0.4},
}

TOWER_ZONES = ["zone_A", "zone_B", "zone_C", "zone_D"]
ZONE_HABITAT = {"zone_A": "wetland", "zone_B": "grassland",
                "zone_C": "mountain", "zone_D": "wetland"}


def _co_occurrence_score(s1: str, s2: str) -> float:
    h1, h2 = BIRD_SPECIES[s1]["habitat_type"], BIRD_SPECIES[s2]["habitat_type"]
    base = 0.7 if h1 == h2 else 0.2
    d1, d2 = BIRD_SPECIES[s1]["danger_rank"], BIRD_SPECIES[s2]["danger_rank"]
    return round(base * (1 - abs(d1 - d2) * 0.3), 3)


def generate_bird_layer():
    """Generate bird distribution layer nodes and edges."""
    nodes = {bird_id: info for bird_id, info in BIRD_SPECIES.items()}
    edges = {}
    bird_ids = list(BIRD_SPECIES.keys())
    for i, b1 in enumerate(bird_ids):
        for b2 in bird_ids[i + 1:]:
            score = _co_occurrence_score(b1, b2)
            # Edge weight varies by season
            seasonal = {}
            for s in SEASONS:
                p1 = BIRD_SPECIES[b1]["seasonal_presence"][s]
                p2 = BIRD_SPECIES[b2]["seasonal_presence"][s]
                seasonal[s] = round(score * np.sqrt(p1 * p2), 3)
            edges[(b1, b2)] = seasonal
    return {"nodes": nodes, "edges": edges}


def generate_tower_layer(n_towers=30):
    """Generate tower / power line layer."""
    np.random.seed(42)
    nodes = {}
    zone_centers = {"zone_A": (0.3, 0.3), "zone_B": (0.7, 0.3),
                    "zone_C": (0.5, 0.7), "zone_D": (0.3, 0.7)}
    for i in range(n_towers):
        tid = f"T{i+1:02d}"
        # Assign to zone with noise
        z = TOWER_ZONES[i % len(TOWER_ZONES)]
        cx, cy = zone_centers[z]
        lat = cx + np.random.normal(0, 0.05)
        lon = cy + np.random.normal(0, 0.05)
        nodes[tid] = {
            "name_cn": f"杆塔{i+1:02d}", "zone": z,
            "lat": round(lat, 3), "lon": round(lon, 3),
            "voltage_kv": np.random.choice([110, 220, 500], p=[0.3, 0.5, 0.2]),
            "importance": np.random.uniform(0.3, 1.0),
        }
    # Build edges: towers in same line segment are connected
    edges = {}
    # Simple linear topology within each zone
    zone_towers = {z: [] for z in TOWER_ZONES}
    for tid, info in nodes.items():
        zone_towers[info["zone"]].append(tid)
    for z, tlist in zone_towers.items():
        for i in range(len(tlist) - 1):
            t1, t2 = tlist[i], tlist[i + 1]
            v = np.mean([nodes[t1]["voltage_kv"], nodes[t2]["voltage_kv"]])
            edges[(t1, t2)] = round(v / 500.0, 3)  # normalized weight
        # Cross-zone connections (some)
        if len(tlist) > 0:
            for other_z in TOWER_ZONES:
                if other_z != z and np.random.random() < 0.4:
                    t1 = tlist[-1]
                    t2 = zone_towers[other_z][0]
                    v = np.mean([nodes[t1]["voltage_kv"], nodes[t2]["voltage_kv"]])
                    edges[(t1, t2)] = round(v / 500.0, 3)
    return {"nodes": nodes, "edges": edges}


def generate_measure_layer(n_measures=18):
    """Generate deterrent/attractant measure layer."""
    np.random.seed(123)
    m_keys = list(MEASURE_TYPES.keys())
    nodes = {}
    for i in range(n_measures):
        mid = f"M{i+1:02d}"
        mtype = m_keys[i % len(m_keys)]
        nodes[mid] = {
            "name_cn": MEASURE_TYPES[mtype]["name_cn"],
            "type": MEASURE_TYPES[mtype]["type"],
            "effectiveness": MEASURE_TYPES[mtype]["effect"],
            "zone": np.random.choice(TOWER_ZONES),
        }
    edges = {}
    m_ids = list(nodes.keys())
    for i, m1 in enumerate(m_ids):
        for m2 in m_ids[i + 1:]:
            if nodes[m1]["zone"] == nodes[m2]["zone"]:
                edges[(m1, m2)] = round(np.random.uniform(0.3, 0.8), 3)
            elif nodes[m1]["type"] == nodes[m2]["type"]:
                edges[(m1, m2)] = round(np.random.uniform(0.2, 0.5), 3)
    return {"nodes": nodes, "edges": edges}


def generate_inter_layer_edges(bird_layer, tower_layer, measure_layer):
    """Generate cross-layer edges: L1->L2, L1->L3, L3->L1."""
    np.random.seed(99)
    bird_ids = list(bird_layer["nodes"].keys())
    tower_ids = list(tower_layer["nodes"].keys())
    measure_ids = list(measure_layer["nodes"].keys())

    # L1 -> L2 (bird -> tower conflict risk, seasonal)
    bird_tower = {}
    for b in bird_ids:
        b_habitat = BIRD_SPECIES[b]["habitat_type"]
        b_danger = BIRD_SPECIES[b]["danger_rank"]
        for t in tower_ids:
            t_zone = tower_layer["nodes"][t]["zone"]
            t_zone_hab = ZONE_HABITAT[t_zone]
            habitat_match = 0.6 if b_habitat == t_zone_hab else 0.2
            seasonal = {}
            for s in SEASONS:
                presence = BIRD_SPECIES[b]["seasonal_presence"][s]
                seasonal[s] = round(min(b_danger * presence * habitat_match +
                                        np.random.uniform(-0.1, 0.1), 1.0), 3)
            bird_tower[(b, t)] = seasonal

    # L1 -> L3 (measure effectiveness on bird species, non-seasonal)
    bird_measure = {}
    for b in bird_ids:
        b_habitat = BIRD_SPECIES[b]["habitat_type"]
        for m in measure_ids:
            m_info = measure_layer["nodes"][m]
            if m_info["type"] == "attractant":
                eff = m_info["effectiveness"] * (0.7 if b_habitat == "wetland" else 0.4) * \
                      BIRD_SPECIES[b]["seasonal_presence"]["summer"]
            else:
                eff = m_info["effectiveness"] * BIRD_SPECIES[b]["danger_rank"] * \
                      np.random.uniform(0.5, 1.0)
            bird_measure[(b, m)] = round(min(eff + np.random.uniform(-0.05, 0.05), 1.0), 3)

    # L3 -> L1 (measure impact on bird distribution, negative for deterrent)
    measure_bird = {}
    for m in measure_ids:
        m_info = measure_layer["nodes"][m]
        is_deterrent = m_info["type"] == "deterrent"
        for b in bird_ids:
            if is_deterrent:
                val = -m_info["effectiveness"] * BIRD_SPECIES[b]["danger_rank"]
            else:
                val = m_info["effectiveness"] * 0.3
            measure_bird[(m, b)] = round(val + np.random.uniform(-0.1, 0.1), 3)

    return {
        "bird_tower": bird_tower,
        "bird_measure": bird_measure,
        "measure_bird": measure_bird,
    }


def generate_all_data(n_towers=30, n_measures=18):
    """Generate the complete multi-layer network dataset."""
    bird = generate_bird_layer()
    tower = generate_tower_layer(n_towers)
    measure = generate_measure_layer(n_measures)
    inter = generate_inter_layer_edges(bird, tower, measure)
    return {
        "seasons": SEASONS,
        "bird_layer": bird,
        "tower_layer": tower,
        "measure_layer": measure,
        "inter_layer": inter,
    }
