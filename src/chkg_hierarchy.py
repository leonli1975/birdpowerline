"""Causal-Hierarchical Knowledge Graph — hierarchy construction.

Builds three causal hierarchy trees:
  H_sp: Spatial containment (Landscape > Ecoregion > Habitat > Grid > Tower)
  H_tax: Taxonomic classification (Order > Family > Genus > Species)
  H_tro: Trophic (food-web) dynamics

Each tree is encoded as adjacency lists for efficient partial order enforcement.
"""

import numpy as np

# ============================================================
# Spatial Containment Hierarchy
# ============================================================

# Root → Landscape → Ecoregion → Habitat → Grid → Tower
SPATIAL_NODES = [
    "root_spatial",
    "landscape_qinghai_tibet",
    "ecoregion_zoige_wetland",
    "hab_alpine_meadow", "hab_wetland_marsh", "hab_alpine_shrub",
    "hab_river_lake", "hab_bare_rock", "hab_settlement",
]

SPATIAL_EDGES = [
    # Root -> Landscape
    ("root_spatial", "landscape_qinghai_tibet"),
    # Landscape -> Ecoregion
    ("landscape_qinghai_tibet", "ecoregion_zoige_wetland"),
    # Ecoregion -> Habitats (all habitats within Zoige)
    ("ecoregion_zoige_wetland", "hab_alpine_meadow"),
    ("ecoregion_zoige_wetland", "hab_wetland_marsh"),
    ("ecoregion_zoige_wetland", "hab_alpine_shrub"),
    ("ecoregion_zoige_wetland", "hab_river_lake"),
    ("ecoregion_zoige_wetland", "hab_bare_rock"),
    ("ecoregion_zoige_wetland", "hab_settlement"),
]

# Map power lines to habitats (from KG data)
LINE_TO_HABITAT = {
    "line_aji":       "hab_wetland_marsh",
    "line_ruozhen":   "hab_wetland_marsh",
    "line_ruotang":   "hab_wetland_marsh",
    "line_anmai":     "hab_alpine_meadow",
    "line_heishang":  "hab_alpine_meadow",
    "line_heize":     "hab_alpine_meadow",
    "line_heimai":    "hab_alpine_meadow",
    "line_ruoa":      "hab_alpine_meadow",
    "line_ehu":       "hab_alpine_meadow",
    "line_manqiong":  "hab_alpine_meadow",
    "line_manka":     "hab_alpine_meadow",
}

# Map bird species to primary habitats (from KG data)
BIRD_TO_HABITAT = {
    "Gr_nigricollis":    "hab_wetland_marsh",
    "An_anser":          "hab_wetland_marsh",
    "Ta_ferruginea":     "hab_wetland_marsh",
    "St_ciconia":        "hab_wetland_marsh",
    "Gr_grus":           "hab_wetland_marsh",
    "Pl_leucorodia":     "hab_wetland_marsh",
    "Ch_dubius":         "hab_wetland_marsh",
    "La_brunnicephalus": "hab_wetland_marsh",
    "An_formosa":        "hab_wetland_marsh",
    "An_crecca":         "hab_wetland_marsh",
    "Bu_hemilasius":     "hab_alpine_meadow",
    "Fa_tinnunculus":    "hab_alpine_meadow",
    "Aq_nipalensis":     "hab_alpine_meadow",
    "Co_corax":          "hab_alpine_meadow",
    "Pi_pica":           "hab_alpine_meadow",
    "An_noctua":         "hab_alpine_meadow",
    "Fa_cherrug":        "hab_alpine_meadow",
    "Aq_chrysaetos":     "hab_alpine_shrub",
    "Gy_himalayensis":   "hab_alpine_shrub",
    "Mi_migrans":        "hab_alpine_shrub",
}

# ============================================================
# Taxonomic Classification Hierarchy
# ============================================================

TAXONOMIC_NODES = [
    "root_taxonomic",
    "order_Accipitriformes", "order_Falconiformes", "order_Gruiformes",
    "order_Anseriformes", "order_Charadriiformes", "order_Pelecaniformes",
    "order_Ciconiiformes", "order_Strigiformes", "order_Passeriformes",
    # Accipitriformes
    "fam_Accipitridae",
    # Falconiformes
    "fam_Falconidae",
    # Gruiformes
    "fam_Gruidae",
    # Anseriformes
    "fam_Anatidae",
    # Charadriiformes
    "fam_Scolopacidae", "fam_Laridae", "fam_Charadriidae",
    # Pelecaniformes
    "fam_Threskiornithidae",
    # Ciconiiformes
    "fam_Ciconiidae",
    # Strigiformes
    "fam_Strigidae",
    # Passeriformes
    "fam_Corvidae",
]

TAXONOMIC_EDGES = [
    ("root_taxonomic", "order_Accipitriformes"),
    ("root_taxonomic", "order_Falconiformes"),
    ("root_taxonomic", "order_Gruiformes"),
    ("root_taxonomic", "order_Anseriformes"),
    ("root_taxonomic", "order_Charadriiformes"),
    ("root_taxonomic", "order_Pelecaniformes"),
    ("root_taxonomic", "order_Ciconiiformes"),
    ("root_taxonomic", "order_Strigiformes"),
    ("root_taxonomic", "order_Passeriformes"),
    # Families
    ("order_Accipitriformes", "fam_Accipitridae"),
    ("order_Falconiformes", "fam_Falconidae"),
    ("order_Gruiformes", "fam_Gruidae"),
    ("order_Anseriformes", "fam_Anatidae"),
    ("order_Charadriiformes", "fam_Scolopacidae"),
    ("order_Charadriiformes", "fam_Laridae"),
    ("order_Charadriiformes", "fam_Charadriidae"),
    ("order_Pelecaniformes", "fam_Threskiornithidae"),
    ("order_Ciconiiformes", "fam_Ciconiidae"),
    ("order_Strigiformes", "fam_Strigidae"),
    ("order_Passeriformes", "fam_Corvidae"),
]

# Map bird species to families
BIRD_TO_FAMILY = {
    "Bu_hemilasius":     "fam_Accipitridae",
    "Aq_chrysaetos":     "fam_Accipitridae",
    "Aq_nipalensis":     "fam_Accipitridae",
    "Gy_himalayensis":   "fam_Accipitridae",
    "Mi_migrans":        "fam_Accipitridae",
    "Fa_cherrug":        "fam_Falconidae",
    "Fa_tinnunculus":    "fam_Falconidae",
    "Gr_nigricollis":    "fam_Gruidae",
    "Gr_grus":           "fam_Gruidae",
    "An_anser":          "fam_Anatidae",
    "Ta_ferruginea":     "fam_Anatidae",
    "An_formosa":        "fam_Anatidae",
    "An_crecca":         "fam_Anatidae",
    "Ch_dubius":         "fam_Charadriidae",
    "La_brunnicephalus": "fam_Laridae",
    "Pl_leucorodia":     "fam_Threskiornithidae",
    "St_ciconia":        "fam_Ciconiidae",
    "An_noctua":         "fam_Strigidae",
    "Co_corax":          "fam_Corvidae",
    "Pi_pica":           "fam_Corvidae",
}

# ============================================================
# Trophic (Food-Web) Hierarchy — Expanded with species-level edges
# Based on Yi et al. (2004): stable carbon isotope analysis of
# alpine meadow food chains at Haibei Station, Qinghai-Tibet Plateau.
#
# Five food chains identified (δ13C enrichment factor = 1.05‰):
#   1. Plants → Livestock / Herbivorous passerines (2-node)
#   2. Plants → Small mammals → Raptors/Carnivores (3-node)
#   3. Plants → Insects → Passerine birds → Raptors (4-node) ★
#   4. Plants → Insects → Amphibians → Raptors/Carnivores (4-node)
#
# Key finding: 大鵟 (Buteo hemilasius) δ13C = -22.80‰,
# enrichment from small mammals = 2.60‰ (too large → NOT primary prey).
# Primary diet = passerine birds (insect-eating/omnivorous).
# This dietary shift followed large-scale rodent control (Yi et al. 2003).
# ============================================================

TROPHIC_NODES = [
    "root_trophic",
    "level_producer",       # Plants (禾本科/菊科/莎草科)
    "level_primary_cons",   # Herbivores
    "level_insect",         # Insects (草原毛虫/蝶/蝇/蝽/步甲)
    "level_secondary_cons", # Insectivores / small carnivores
    "level_apex_predator",  # Top predators
]

TROPHIC_EDGES = [
    # Root → Producer
    ("root_trophic", "level_producer"),
    # Producer → Primary consumers (herbivores)
    ("level_producer", "level_primary_cons"),
    # Producer → Insects
    ("level_producer", "level_insect"),
    # Primary consumers → Apex predators (chain 2)
    ("level_primary_cons", "level_apex_predator"),
    # Insects → Secondary consumers (passerine birds)
    ("level_insect", "level_secondary_cons"),
    # Secondary consumers → Apex predators (chain 3 main path)
    ("level_secondary_cons", "level_apex_predator"),
    # Insects → Alternative path (amphibians, chain 4)
    ("level_insect", "trophic_amphibians"),
    # Amphibians → Apex predators (chain 4)
    ("trophic_amphibians", "level_apex_predator"),
]

# Add amphibian trophic node
TROPHIC_NODES.append("trophic_amphibians")

# Map birds and mammals to trophic levels (updated with isotope evidence)
BIRD_TO_TROPHIC = {
    # Apex predators (confirmed by δ13C)
    "Bu_hemilasius":     "level_apex_predator",     # 大鵟 — mainly eats passerines
    "Fa_cherrug":        "level_apex_predator",     # 猎隼
    "Fa_tinnunculus":    "level_apex_predator",     # 红隼
    "Aq_chrysaetos":     "level_apex_predator",     # 金雕
    "Gy_himalayensis":   "level_apex_predator",     # 高山兀鹫
    "Mi_migrans":        "level_apex_predator",     # 黑鸢
    "Aq_nipalensis":     "level_apex_predator",     # 草原雕
    "An_noctua":         "level_apex_predator",     # 纵纹腹小鸮
    # Secondary consumers — insect-eating passerines
    "Co_corax":          "level_secondary_cons",    # 渡鸦 (omnivorous)
    "Pi_pica":           "level_secondary_cons",    # 喜鹊 (omnivorous)
    # Primary consumers — herbivorous waterbirds
    "Gr_nigricollis":    "level_primary_cons",      # 黑颈鹤 (plants/small animals)
    "Gr_grus":           "level_primary_cons",      # 灰鹤
    "St_ciconia":        "level_primary_cons",      # 白鹳
    "Pl_leucorodia":     "level_primary_cons",      # 白琵鹭
    "An_anser":          "level_primary_cons",      # 灰雁
    "Ta_ferruginea":     "level_primary_cons",      # 赤麻鸭
    "An_formosa":        "level_primary_cons",      # 花脸鸭
    "An_crecca":         "level_primary_cons",      # 绿翅鸭
    "Ch_dubius":         "level_primary_cons",      # 金眶鸻
    "La_brunnicephalus": "level_primary_cons",      # 棕头鸥
}


def build_all_hierarchies():
    """Build all three hierarchy trees as (node_list, edge_list) pairs."""
    hierarchies = {}

    # Spatial
    sp_edges = SPATIAL_EDGES.copy()
    for line, hab in LINE_TO_HABITAT.items():
        sp_edges.append((hab, line))
    for bird, hab in BIRD_TO_HABITAT.items():
        sp_edges.append((hab, bird))
    sp_nodes = list(set([u for u, v in sp_edges] + [v for u, v in sp_edges]))
    hierarchies["spatial"] = (sp_nodes, sp_edges)

    # Taxonomic
    tax_edges = TAXONOMIC_EDGES.copy()
    for bird, fam in BIRD_TO_FAMILY.items():
        tax_edges.append((fam, bird))
    tax_nodes = list(set([u for u, v in tax_edges] + [v for u, v in tax_edges]))
    hierarchies["taxonomic"] = (tax_nodes, tax_edges)

    # Trophic
    tro_edges = TROPHIC_EDGES.copy()
    for bird, level in BIRD_TO_TROPHIC.items():
        tro_edges.append((level, bird))
    tro_nodes = list(set([u for u, v in tro_edges] + [v for u, v in tro_edges]))
    hierarchies["trophic"] = (tro_nodes, tro_edges)

    return hierarchies


def extend_entity_set(e2i, i2e, hierarchies):
    """Extend entity dictionary with hierarchy nodes.

    Returns (e2i_new, i2e_new) with hierarchy nodes appended.
    """
    all_nodes = set()
    for (nodes, edges) in hierarchies.values():
        all_nodes.update(nodes)
        all_nodes.update([u for u, v in edges])
        all_nodes.update([v for u, v in edges])

    new_nodes = sorted([n for n in all_nodes if n not in e2i])
    e2i_new = dict(e2i)
    i2e_new = dict(i2e)
    n_e = len(e2i_new)
    for node in new_nodes:
        e2i_new[node] = n_e
        i2e_new[n_e] = node
        n_e += 1
    return e2i_new, i2e_new


def build_hierarchy_masks(hierarchies, e2i):
    """Build adjacency and ancestor masks for each hierarchy.

    Returns dict with keys: spatial, taxonomic, trophic.
    Each value is {'adj': dict, 'ancestor_mask': np.array, 'n_edges': int}
    """
    n_e = len(e2i)
    hierarchy_data = {}

    for hname, (nodes, edges) in hierarchies.items():
        adj = {i: [] for i in range(n_e)}
        for parent, child in edges:
            if parent in e2i and child in e2i:
                adj[e2i[parent]].append(e2i[child])

        ancestor = np.zeros((n_e, n_e), dtype=bool)
        for parent_name, parent_id in {p: e2i[p] for p, c in edges if p in e2i}.items():
            queue = [parent_id]
            visited = {parent_id}
            while queue:
                cur = queue.pop(0)
                for ch in adj.get(cur, []):
                    if ch not in visited:
                        visited.add(ch)
                        queue.append(ch)
                        ancestor[parent_id, ch] = True

        hierarchy_data[hname] = {
            "adj": adj,
            "ancestor_mask": ancestor,
            "n_nodes": len([n for n in nodes if n in e2i]),
            "n_edges": len(edges),
        }

    return hierarchy_data


if __name__ == "__main__":
    from src.kg_data import build_temporal_kg, enrich_kg, build_id_mappings

    kg = build_temporal_kg()
    kg = enrich_kg(kg)
    e2i, r2i, i2e, i2r = build_id_mappings(kg)

    hierarchies = build_all_hierarchies()
    e2i_ext, i2e_ext = extend_entity_set(e2i, i2e, hierarchies)
    masks = build_hierarchy_masks(hierarchies, e2i_ext)

    for name, info in masks.items():
        n_anc = info["ancestor_mask"].sum()
        print(f"{name}: {info['n_edges']} edges, {info['n_nodes']} nodes, "
              f"{n_anc} ancestor pairs, density={n_anc/(len(e2i_ext)**2):.4f}")
