"""Temporal Knowledge Graph data construction from Aba document data.

Uses year-level temporal granularity (2017-2024, 8 timestamps) for training.
Season-level information is captured in the relationships themselves.

Format: (head_entity, relation, tail_entity, time_begin_year, time_end_year)
"""

# ============================================================
# Entity dictionaries
# ============================================================

BIRD_SPECIES = {
    "Gr_nigricollis":        {"name_cn":"黑颈鹤","class":"I","residency":"summer_visitor",
                               "habitat":"wetland","body_size":"large","taxon":"Gruiformes"},
    "Aq_chrysaetos":         {"name_cn":"金雕","class":"I","residency":"resident",
                               "habitat":"mountain","body_size":"large","taxon":"Accipitriformes"},
    "Fa_cherrug":            {"name_cn":"猎隼","class":"I","residency":"summer_visitor",
                               "habitat":"grassland","body_size":"medium","taxon":"Falconiformes"},
    "Bu_hemilasius":         {"name_cn":"大鵟","class":"II","residency":"summer_visitor",
                               "habitat":"grassland","body_size":"medium","taxon":"Accipitriformes",
                               "note":"主要冲突物种，塔顶筑巢"},
    "Fa_tinnunculus":        {"name_cn":"红隼","class":"II","residency":"resident",
                               "habitat":"grassland","body_size":"small","taxon":"Falconiformes"},
    "Mi_migrans":            {"name_cn":"黑鸢","class":"II","residency":"summer_visitor",
                               "habitat":"mountain","body_size":"medium","taxon":"Accipitriformes",
                               "note":"集群活动,一次调查>35只"},
    "Gy_himalayensis":       {"name_cn":"高山兀鹫","class":"II","residency":"resident",
                               "habitat":"mountain","body_size":"large","taxon":"Accipitriformes"},
    "Co_corax":              {"name_cn":"渡鸦","class":"unprotected","residency":"resident",
                               "habitat":"grassland","body_size":"medium","taxon":"Passeriformes"},
    "Pi_pica":               {"name_cn":"喜鹊","class":"unprotected","residency":"resident",
                               "habitat":"grassland","body_size":"small","taxon":"Passeriformes"},
    "An_noctua":             {"name_cn":"纵纹腹小鸮","class":"II","residency":"resident",
                               "habitat":"grassland","body_size":"small","taxon":"Strigiformes"},
    "An_anser":              {"name_cn":"斑头雁","class":"unprotected","residency":"summer_visitor",
                               "habitat":"wetland","body_size":"medium","taxon":"Anseriformes"},
    "Ta_ferruginea":         {"name_cn":"赤麻鸭","class":"unprotected","residency":"summer_visitor",
                               "habitat":"wetland","body_size":"medium","taxon":"Anseriformes"},
    "Aq_nipalensis":         {"name_cn":"草原雕","class":"II","residency":"summer_visitor",
                               "habitat":"grassland","body_size":"large","taxon":"Accipitriformes"},
    "St_ciconia":            {"name_cn":"东方白鹳","class":"I","residency":"passage_migrant",
                               "habitat":"wetland","body_size":"large","taxon":"Ciconiiformes"},
    "Pl_leucorodia":         {"name_cn":"白琵鹭","class":"II","residency":"summer_visitor",
                               "habitat":"wetland","body_size":"medium","taxon":"Pelecaniformes"},
    "An_formosa":            {"name_cn":"花脸鸭","class":"II","residency":"passage_migrant",
                               "habitat":"wetland","body_size":"small","taxon":"Anseriformes"},
    "Ch_dubius":             {"name_cn":"金眶鸻","class":"unprotected","residency":"summer_visitor",
                               "habitat":"wetland","body_size":"small","taxon":"Charadriiformes"},
    "La_brunnicephalus":     {"name_cn":"棕头鸥","class":"unprotected","residency":"summer_visitor",
                               "habitat":"wetland","body_size":"small","taxon":"Charadriiformes"},
    "An_crecca":             {"name_cn":"绿翅鸭","class":"unprotected","residency":"passage_migrant",
                               "habitat":"wetland","body_size":"small","taxon":"Anseriformes"},
    "Gr_grus":               {"name_cn":"灰鹤","class":"II","residency":"passage_migrant",
                               "habitat":"wetland","body_size":"large","taxon":"Gruiformes"},
}

POWER_LINES = {
    "line_ruozhen":   {"name_cn":"若真线","voltage":110,"poles":179,"length_km":68.9,
                        "nest_poles":92,"nest_rate":0.514,"n_trippings_2020_2022":18},
    "line_ruotang":   {"name_cn":"若唐线","voltage":110,"n_trippings_2021":5,
                        "n_trippings_2022":1},
    "line_anmai":     {"name_cn":"安麦线","voltage":110},
    "line_aji":       {"name_cn":"阿吉线","voltage":35,"poles":260,"length_km":77.618,
                        "altitude_range":"2900-4200m"},
    "line_heishang":  {"name_cn":"黑上线","voltage":10,"artificial_nests":5,"activity":3},
    "line_heize":     {"name_cn":"黑泽线","voltage":10,"artificial_nests":5,"activity":1},
    "line_heimai":    {"name_cn":"黑麦线","voltage":10,"artificial_nests":5,"activity":5},
    "line_ruoa":      {"name_cn":"若阿线","voltage":10,"artificial_nests":9,"activity":5},
    "line_ehu":       {"name_cn":"俄湖线","voltage":10,"artificial_nests":3,"activity":1},
    "line_manqiong":  {"name_cn":"曼穷线","voltage":10},
    "line_manka":     {"name_cn":"曼卡线","voltage":10},
}

MEASURES = {
    "anti_bird_baffle":     {"name_cn":"防鸟绝缘挡板","type":"physical_barrier",
                              "target":"droppings_flashover","n_deployed":325},
    "artificial_nest":      {"name_cn":"仿生人工鸟巢","type":"attractant",
                              "target":"nest_shorting","n_deployed_110kV":33,"n_deployed_10kV":27},
    "smart_deterrent":      {"name_cn":"智能驱鸟装置","type":"deterrent",
                              "target":"perching","n_deployed":20},
    "hd_video":             {"name_cn":"高清视频监测","type":"monitoring",
                              "target":"behavior_tracking","n_deployed":20},
    "dynamic_spikes":       {"name_cn":"动态防鸟刺","type":"physical_barrier",
                              "target":"perching","n_deployed":107},
    "rotating_deterrent":   {"name_cn":"旋转驱鸟器","type":"deterrent",
                              "target":"perching","n_deployed":2},
    "insulation_sleeve":    {"name_cn":"绝缘护套","type":"physical_barrier",
                              "target":"body_shorting"},
    "acoustic_deterrent":   {"name_cn":"声波驱鸟器","type":"deterrent",
                              "target":"flocking"},
}

FAULT_TYPES = {
    "droppings_flashover":  {"name_cn":"鸟粪闪络","national_pct":87},
    "nest_shorting":        {"name_cn":"筑巢短接","national_pct":10},
    "body_shorting":        {"name_cn":"鸟体短接","national_pct":2},
    "pecking_damage":       {"name_cn":"鸟啄损伤","national_pct":1},
}

HABITATS = {
    "alpine_meadow":    {"name_cn":"高寒草甸","species_count":165},
    "alpine_shrub":     {"name_cn":"高山灌丛","species_count":127},
    "wetland_marsh":    {"name_cn":"沼泽湿地"},
    "river_lake":       {"name_cn":"河流湖泊"},
    "bare_rock":        {"name_cn":"裸岩"},
    "settlement":       {"name_cn":"居民区"},
    "farmland":         {"name_cn":"农田"},
}

# Year-level timestamps: 2017-2024 (8 years)
TIMESTAMPS = list(range(2017, 2025))
N_TIMESTAMPS = len(TIMESTAMPS)

# Legacy: season-level timestamps
SEASONS = ["2017-Spring","2017-Summer","2017-Autumn","2017-Winter",
           "2018-Spring","2018-Summer","2018-Autumn","2018-Winter",
           "2019-Spring","2019-Summer","2019-Autumn","2019-Winter",
           "2020-Spring","2020-Summer","2020-Autumn","2020-Winter",
           "2021-Spring","2021-Summer","2021-Autumn","2021-Winter",
           "2022-Spring","2022-Summer","2022-Autumn","2022-Winter",
           "2023-Spring","2023-Summer","2023-Autumn","2023-Winter",
           "2024-Spring"]
TIME_TO_IDX = {t: i for i, t in enumerate(SEASONS)}


def build_temporal_kg():
    """Build the full temporal knowledge graph from extracted document data.

    Temporal triples use YEAR-level granularity (0=2017, ..., 7=2024).
    Returns: list of (head, relation, tail, time_begin, time_end) tuples.
    """
    kg = []

    def add(h, r, t, tb=None, te=None):
        kg.append((h, r, t, tb, te))

    # --- Static attributes ---
    for bid, info in BIRD_SPECIES.items():
        add(bid, "has_residency", f"res_{info['residency']}")
        add(bid, "inhabits", f"hab_{info.get('habitat','unknown')}")
        if info.get("body_size"):
            add(bid, "has_body_size", f"size_{info['body_size']}")
        if info.get("class") in ("I","II"):
            add(bid, "has_protection", f"prot_{info['class']}")

    for lid, info in POWER_LINES.items():
        add(lid, "has_voltage", f"volt_{info['voltage']}")
        add(lid, "located_in", "reg_zoige")

    for mid, info in MEASURES.items():
        add(mid, "targets_fault", f"fault_{info['target']}")
        add(mid, "has_type", f"meas_type_{info['type']}")

    # ============================================================
    # Temporal triples — YEAR-level granularity
    # ============================================================

    # --- Bird presence per year (resident = all years, summer_visitor = spring-autumn, etc.) ---
    for bid, info in BIRD_SPECIES.items():
        res = info["residency"]
        for year in TIMESTAMPS:
            yi = year - 2017
            if res == "resident":
                add(bid, "present_in", "reg_zoige", yi, yi)
            elif res == "summer_visitor":
                add(bid, "present_in", "reg_zoige", yi, yi)
            elif res == "passage_migrant":
                add(bid, "migrates_through", "reg_zoige", yi, yi)
            elif res == "winter_visitor":
                add(bid, "present_in", "reg_zoige", yi, yi)

    # --- Breeding threats (spring-summer birds threaten key lines) ---
    breeding_birds = ["Bu_hemilasius","Fa_tinnunculus","Mi_migrans",
                      "Co_corax","Pi_pica","Fa_cherrug"]
    target_lines = ["line_ruozhen","line_ruotang","line_heishang","line_heimai","line_ruoa"]
    for bid in breeding_birds:
        for year in TIMESTAMPS:
            yi = year - 2017
            for line in target_lines:
                add(bid, "nests_on", line, yi, yi)
                add(bid, "threatens", line, yi, yi)
            add(bid, "causes", "fault_nest_shorting", yi, yi)

    # --- Large bird body shorting ---
    large_birds = ["Gr_nigricollis","Aq_chrysaetos","Gy_himalayensis",
                   "St_ciconia","Gr_grus"]
    for bid in large_birds:
        info = BIRD_SPECIES[bid]
        if info["residency"] in ("resident","summer_visitor"):
            for year in TIMESTAMPS:
                yi = year - 2017
                add(bid, "causes", "fault_body_shorting", yi, yi)

    # --- Droppings flashover — common perching birds ---
    dropping_birds = ["Bu_hemilasius","Mi_migrans","Co_corax","Pi_pica",
                      "Fa_tinnunculus","Aq_nipalensis"]
    for bid in dropping_birds:
        for year in TIMESTAMPS:
            yi = year - 2017
            add(bid, "causes", "fault_droppings_flashover", yi, yi)

    # --- Migration threats (spring) ---
    migrant_birds = ["Gr_nigricollis","An_anser","Gr_grus","St_ciconia",
                     "An_formosa","An_crecca","Ch_dubius"]
    for bid in migrant_birds:
        for year in TIMESTAMPS:
            yi = year - 2017
            add(bid, "threatens", "line_ruozhen", yi, yi)
            add(bid, "threatens", "line_aji", yi, yi)

    # --- Competition (observed 2024) ---
    add("Fa_cherrug","competes_with","Bu_hemilasius", 7, 7)  # 2024
    add("Bu_hemilasius","competes_with","Fa_tinnunculus")
    add("Mi_migrans","competes_with","Bu_hemilasius")

    # --- Measure deployment ---
    # 2020: first dynamic spikes + smart deterrents
    add("dynamic_spikes","deployed_on","line_ruozhen", 3, None)    # 2020+
    add("dynamic_spikes","deployed_on","line_ruotang", 3, None)
    add("dynamic_spikes","deployed_on","line_anmai", 3, None)
    add("smart_deterrent","deployed_on","line_ruozhen", 3, 6)      # 2020-2023
    add("hd_video","deployed_on","line_ruozhen", 3, None)           # 2020+

    # 2023 April: Comprehensive deployment on 110kV
    for line in ["line_ruozhen","line_ruotang","line_anmai"]:
        for measure in ["anti_bird_baffle","artificial_nest","smart_deterrent",
                         "hd_video","dynamic_spikes"]:
            add(measure, "deployed_on", line, 6, None)  # 2023+

    # 2023: 10kV artificial nests + insulation
    for line in ["line_heishang","line_ruoa","line_heimai","line_ehu",
                 "line_heize","line_manqiong","line_manka"]:
        add("artificial_nest","deployed_on", line, 6, None)
    for line in ["line_heishang","line_ruoa","line_heimai"]:
        add("insulation_sleeve","deployed_on", line, 6, None)

    # --- Measure effectiveness (static) ---
    add("anti_bird_baffle","mitigates","fault_droppings_flashover")
    add("artificial_nest","mitigates","fault_nest_shorting")
    add("insulation_sleeve","mitigates","fault_body_shorting")
    add("dynamic_spikes","mitigates","fault_nest_shorting")
    add("smart_deterrent","mitigates","fault_droppings_flashover")
    add("dynamic_spikes","mitigates","fault_droppings_flashover")

    # --- Nest occupancy (2023, 2024) ---
    add("Bu_hemilasius","occupies_nest","line_ruozhen", 6, 7)
    add("Fa_cherrug","occupies_nest","line_ruozhen", 7, 7)

    # --- Fault events ---
    # 2017-2019: high fault period
    for year in range(2017, 2020):
        yi = year - 2017
        add("line_ruozhen","has_fault","fault_droppings_flashover", yi, yi)
        add("line_ruotang","has_fault","fault_droppings_flashover", yi, yi)

    # 2020-2022: reduced faults
    for year in range(2020, 2023):
        yi = year - 2017
        add("line_ruozhen","has_fault","fault_droppings_flashover", yi, yi)
        add("line_ruozhen","has_fault","fault_nest_shorting", yi, yi)

    # 10kV body shorting 2020-2021
    add("line_heishang","has_fault","fault_body_shorting", 3, 4)
    add("line_heimai","has_fault","fault_body_shorting", 3, 4)

    # 2023+: zero faults (no fault triples needed — this is the outcome)

    # --- Habitat associations ---
    add("line_aji","near_habitat","hab_wetland_marsh")
    add("line_ruozhen","near_habitat","hab_wetland_marsh")
    add("line_ruozhen","near_habitat","hab_alpine_meadow")
    add("line_heize","near_habitat","hab_alpine_meadow")
    add("line_ruoa","near_habitat","hab_alpine_meadow")

    for hab_bird in ["Bu_hemilasius","Fa_tinnunculus","Mi_migrans"]:
        add("hab_alpine_meadow","supports", hab_bird)
    for hab_bird in ["Gr_nigricollis","An_anser","Gr_grus"]:
        add("hab_wetland_marsh","supports", hab_bird)

    return kg


# ============================================================
# Literature enrichment (from S1, S2, S3 papers in data2/)
# ============================================================

# Raptor elevation ranges from S1 (Xiayong NR) + S2 (Shaluli Mtns)
RAPTOR_ELEVATION = {
    "Gr_nigricollis":      (2800, 4500),  # 黑颈鹤: 若尔盖湿地繁殖
    "Aq_chrysaetos":       (3200, 5000),  # 金雕: S1~3980m, widely distributed
    "Fa_cherrug":          (3400, 4500),  # 猎隼: S1~3420m, IUCN EN
    "Bu_hemilasius":       (3500, 4800),  # 大鵟: S1~4240m, 主要冲突种
    "Fa_tinnunculus":      (2800, 4300),  # 红隼: S1~3200m
    "Mi_migrans":          (2100, 4300),  # 黑鸢: S1 2100-4300m
    "Gy_himalayensis":     (2200, 5100),  # 高山兀鹫: S1 2200-5100m
    "Aq_nipalensis":       (3000, 4800),  # 草原雕: IUCN EN
    "St_ciconia":          (2000, 3500),  # 东方白鹳: 湿地候鸟
    "An_anser":            (3000, 4500),  # 斑头雁: 高原湖泊
    "Gr_grus":             (2500, 4000),  # 灰鹤: 迁徙湿地
}

# Seasonal abundance patterns from S3 (Chengdu, 2020 survey)
# Format: (spring, summer, autumn, winter) — relative abundance tiers
SEASONAL_ABUNDANCE = {
    # 1=low, 2=medium, 3=high, 4=very high
    "Gr_nigricollis":      (3, 2, 2, 1),  # 夏候鸟,春迁高峰
    "Bu_hemilasius":       (3, 4, 3, 1),  # 夏候鸟,夏季繁殖高峰
    "Fa_tinnunculus":      (2, 2, 2, 2),  # 留鸟,全年稳定
    "Mi_migrans":          (3, 4, 2, 1),  # 夏候鸟,夏季集群
    "Gy_himalayensis":     (2, 3, 3, 2),  # 留鸟,秋冬活跃
    "Aq_chrysaetos":       (2, 2, 2, 2),  # 留鸟
    "Fa_cherrug":          (3, 3, 2, 1),  # 夏候鸟
    "An_anser":            (2, 3, 2, 1),  # 夏候鸟
    "Ta_ferruginea":       (2, 3, 2, 1),  # 夏候鸟
    "An_crecca":           (1, 1, 2, 3),  # 冬候鸟
    "An_formosa":          (1, 1, 2, 3),  # 冬候鸟
    "Co_corax":            (2, 2, 2, 2),  # 留鸟
    "Pi_pica":             (2, 2, 2, 2),  # 留鸟
}

# Species richness by elevation band (from S2 mid-peak model)
ELEVATION_BANDS = {
    "elev_low":    (1500, 2800, "低海拔"),
    "elev_mid":    (2800, 3600, "中海拔-物种峰值区"),
    "elev_high":   (3600, 4500, "高海拔-猛禽活动区"),
    "elev_alpine": (4500, 5600, "极高海拔"),
}

# Lines mapped to elevation zones (based on document data)
LINE_ELEVATION = {
    "line_aji":       "elev_mid",      # 阿吉线: 2900-4200m
    "line_ruozhen":   "elev_mid",      # 若真线: 若尔盖~3500m
    "line_ruotang":   "elev_mid",
    "line_anmai":     "elev_mid",
    "line_heishang":  "elev_mid",
    "line_heize":     "elev_mid",
    "line_heimai":    "elev_mid",
    "line_ruoa":      "elev_mid",
    "line_ehu":       "elev_mid",
    "line_manqiong":  "elev_mid",
    "line_manka":     "elev_mid",
}

# Flyway context (S2: two international flyways)
FLYWAYS = [
    "flyway_central_asian_indian",
    "flyway_east_asian_australasian",
]


def enrich_kg(kg):
    """Enrich the base KG with literature-derived data from data2/ papers.

    Adds: elevation mappings, seasonal abundance, flyway context, wetland risk.
    """
    def add(h, r, t, tb=None, te=None):
        kg.append((h, r, t, tb, te))

    # --- Elevation entities ---
    for elev_id, (lo, hi, name) in ELEVATION_BANDS.items():
        add(elev_id, "elev_range_lo", f"val_{lo}")
        add(elev_id, "elev_range_hi", f"val_{hi}")

    # --- Line → elevation zone ---
    for line, elev in LINE_ELEVATION.items():
        add(line, "at_elevation", elev)

    # --- Bird → elevation range ---
    for bird, (lo, hi) in RAPTOR_ELEVATION.items():
        add(bird, "has_elev_min", f"val_{lo}")
        add(bird, "has_elev_max", f"val_{hi}")
        # Link to elevation zones
        for elev_name, (elo, ehi, _) in ELEVATION_BANDS.items():
            if lo <= ehi and hi >= elo:  # overlap
                add(bird, "occupies_elevation", elev_name)

    # --- Bird → elevation zone for non-raptors (wetland birds) ---
    waterbirds = ["An_anser","Ta_ferruginea","An_formosa","An_crecca",
                  "Ch_dubius","La_brunnicephalus","Pl_leucorodia"]
    for bird in waterbirds:
        add(bird, "occupies_elevation", "elev_mid")
        add(bird, "occupies_elevation", "elev_low")

    # --- Seasonal abundance (year-independent static knowledge) ---
    seasons = ["Spring","Summer","Autumn","Winter"]
    for bird, abund in SEASONAL_ABUNDANCE.items():
        for i, s in enumerate(seasons):
            if abund[i] >= 3:
                add(bird, "peak_season", f"season_{s}")

    # --- Flyway context ---
    for fw in FLYWAYS:
        add("reg_zoige", "on_flyway", fw)
    add("reg_zoige", "on_flyway", "flyway_central_asian_indian")
    add("reg_zoige", "on_flyway", "flyway_east_asian_australasian")

    # --- Migration corridor ---
    add("reg_zoige", "is_migration_corridor", "corridor_sichuan_tibet")

    # --- Wetland congregation risk (S2: 35.59% of individuals in wetlands) ---
    add("hab_wetland_marsh", "has_congregation_risk", "risk_high")
    for line in ["line_aji","line_ruozhen","line_ruotang"]:
        add(line, "has_wetland_congestion_risk", "risk_high")

    # --- Raptor soaring behavior risk factor ---
    large_soaring = ["Aq_chrysaetos","Gy_himalayensis","Fa_cherrug",
                     "Bu_hemilasius","Mi_migrans","Aq_nipalensis"]
    for bird in large_soaring:
        add(bird, "has_flight_behavior", "behavior_soaring")
        add(bird, "has_collision_risk_factor", "risk_high_wingspan")

    # --- Black-necked Crane specific (若尔盖核心保护物种) ---
    add("Gr_nigricollis","has_collision_risk_factor","risk_high_wingspan")
    add("Gr_nigricollis","peak_season","season_Spring")
    add("Gr_nigricollis","is_flagship_species","reg_zoige")

    # --- Line-voltage collision risk ---
    for line in ["line_ruozhen","line_ruotang","line_anmai"]:
        add(line, "has_collision_risk", "risk_110kV_high")
    for line in ["line_aji"]:
        add(line, "has_collision_risk", "risk_35kV_moderate")
    for line in ["line_heishang","line_heize","line_heimai",
                 "line_ruoa","line_ehu","line_manqiong","line_manka"]:
        add(line, "has_collision_risk", "risk_10kV_low")

    return kg


def build_id_mappings(kg):
    """Build entity and relation ID mappings from KG triples."""
    all_entities = set()
    all_relations = set()
    for h, r, t, tb, te in kg:
        all_entities.add(h)
        all_entities.add(t)
        all_relations.add(r)

    e2i = {e: i for i, e in enumerate(sorted(all_entities))}
    r2i = {r: i for i, r in enumerate(sorted(all_relations))}
    i2e = {i: e for e, i in e2i.items()}
    i2r = {i: r for r, i in r2i.items()}

    return e2i, r2i, i2e, i2r


def build_tensor_data(kg, e2i, r2i):
    """Convert KG triples to (h, r, t, tau) format for embedding.

    Uses year-level timestamps (0-7 for 2017-2024).
    Static triples (tb=te=None) are replicated across all timestamps.
    """
    n_time = N_TIMESTAMPS  # 8
    triples = []

    for h, r, t, tb, te in kg:
        h_id = e2i.get(h)
        t_id = e2i.get(t)
        r_id = r2i.get(r)
        if h_id is None or t_id is None or r_id is None:
            continue

        if tb is None and te is None:
            for tau in range(n_time):
                triples.append((h_id, r_id, t_id, tau))
        else:
            te = te if te is not None else n_time - 1
            te = min(te, n_time - 1)
            tb = max(0, tb)
            for tau in range(tb, te + 1):
                triples.append((h_id, r_id, t_id, tau))

    return triples, e2i, r2i
