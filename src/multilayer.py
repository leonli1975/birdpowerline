"""Multi-layer temporal network model for bird-powerline-measure coupled system."""

import itertools
import numpy as np
from collections import defaultdict


class MultiLayerTemporalNetwork:
    """三层时序网络模型: L1鸟类层, L2输电线层, L3措施层."""

    def __init__(self, data):
        self.data = data
        self.seasons = data["seasons"]
        self.bird_ids = sorted(data["bird_layer"]["nodes"].keys())
        self.tower_ids = sorted(data["tower_layer"]["nodes"].keys())
        self.measure_ids = sorted(data["measure_layer"]["nodes"].keys())
        self.n_birds = len(self.bird_ids)
        self.n_towers = len(self.tower_ids)
        self.n_measures = len(self.measure_ids)
        self.n_total = self.n_birds + self.n_towers + self.n_measures

        self.bird_idx = {b: i for i, b in enumerate(self.bird_ids)}
        self.tower_idx = {t: i for i, t in enumerate(self.tower_ids)}
        self.measure_idx = {m: i for i, m in enumerate(self.measure_ids)}

        # Precompute all season snapshots
        self._snapshots = self._build_all_snapshots()

    def _build_all_snapshots(self):
        """Build adjacency matrices for each season."""
        snapshots = {}
        for s in self.seasons:
            snapshots[s] = self._build_supra_adjacency(s)
        return snapshots

    def _fill_intra_layer(self, adj, base_offset, ids, node_edges, is_seasonal=False, season=None):
        """Fill intra-layer edges into the supral adjacency matrix."""
        idx_map = {nid: base_offset + i for i, nid in enumerate(ids)}
        for (n1, n2), val in node_edges.items():
            if isinstance(val, dict):
                w = val[season] if is_seasonal else sum(val.values()) / len(val)
            else:
                w = val
            if w > 0:
                i, j = idx_map[n1], idx_map[n2]
                adj[i, j] = w
                adj[j, i] = w

    def _build_supra_adjacency(self, season, inter_weight=0.5):
        """Build a supral adjacency matrix for a given season.

        Block structure:
        [ L1    L1↔L2  L1↔L3 ]
        [ L2↔L1  L2    L2↔L3 ]
        [ L3↔L1  L3↔L2  L3   ]
        """
        n = self.n_total
        A = np.zeros((n, n))

        off_b = 0
        off_t = self.n_birds
        off_m = self.n_birds + self.n_towers

        # L1 intra-layer (seasonal)
        self._fill_intra_layer(A, off_b, self.bird_ids,
                               self.data["bird_layer"]["edges"],
                               is_seasonal=True, season=season)

        # L2 intra-layer
        self._fill_intra_layer(A, off_t, self.tower_ids,
                               self.data["tower_layer"]["edges"])

        # L3 intra-layer
        self._fill_intra_layer(A, off_m, self.measure_ids,
                               self.data["measure_layer"]["edges"])

        # Inter-layer L1 <-> L2 (seasonal)
        inter = self.data["inter_layer"]
        for (b, t), val in inter["bird_tower"].items():
            w = val[season] * inter_weight
            if w > 0:
                i = off_b + self.bird_idx[b]
                j = off_t + self.tower_idx[t]
                A[i, j] = w
                A[j, i] = w

        # Inter-layer L1 <-> L3
        for (b, m), val in inter["bird_measure"].items():
            w = val * inter_weight
            if w > 0:
                i = off_b + self.bird_idx[b]
                j = off_m + self.measure_idx[m]
                A[i, j] = w
                A[j, i] = w

        # Inter-layer L2 <-> L3 (measure deployed on towers)
        for t in self.tower_ids:
            t_zone = self.data["tower_layer"]["nodes"][t]["zone"]
            for m in self.measure_ids:
                m_zone = self.data["measure_layer"]["nodes"][m]["zone"]
                if t_zone == m_zone:
                    w = 0.4 * inter_weight
                    i = off_t + self.tower_idx[t]
                    j = off_m + self.measure_idx[m]
                    A[i, j] = w
                    A[j, i] = w

        # Measure -> Bird impact (negative edge for deterrent, could use signed)
        for (m, b), val in inter["measure_bird"].items():
            abs_w = abs(val) * inter_weight
            if abs_w > 0:
                i = off_m + self.measure_idx[m]
                j = off_b + self.bird_idx[b]
                A[i, j] = abs_w  # Use absolute weight for undirected analysis
                A[j, i] = abs_w

        return A

    def get_snapshot(self, season):
        return self._snapshots[season]

    def get_intra_layer_subgraph(self, season, layer="bird"):
        """Extract intra-layer adjacency from the supral matrix."""
        A = self._snapshots[season]
        if layer == "bird":
            return A[:self.n_birds, :self.n_birds]
        elif layer == "tower":
            r = range(self.n_birds, self.n_birds + self.n_towers)
            return A[self.n_birds:self.n_birds + self.n_towers,
                     self.n_birds:self.n_birds + self.n_towers]
        elif layer == "measure":
            off = self.n_birds + self.n_towers
            return A[off:, off:]
        else:
            raise ValueError(f"Unknown layer: {layer}")

    def get_inter_layer_matrix(self, season, from_layer, to_layer):
        """Extract cross-layer submatrix."""
        A = self._snapshots[season]
        offsets = {"bird": 0, "tower": self.n_birds,
                   "measure": self.n_birds + self.n_towers}
        sizes = {"bird": self.n_birds, "tower": self.n_towers,
                 "measure": self.n_measures}
        r0, c0 = offsets[from_layer], offsets[to_layer]
        rn, cn = sizes[from_layer], sizes[to_layer]
        return A[r0:r0 + rn, c0:c0 + cn]

    def cross_layer_coupling_strength(self, season):
        """Compute total cross-layer coupling strength for a season."""
        A = self._snapshots[season]
        off_b, off_t, off_m = 0, self.n_birds, self.n_birds + self.n_towers
        total = 0.0
        # L1-L2
        total += A[off_b:off_b + self.n_birds,
                   off_t:off_t + self.n_towers].sum()
        # L1-L3
        total += A[off_b:off_b + self.n_birds,
                   off_m:off_m + self.n_measures].sum()
        # L2-L3
        total += A[off_t:off_t + self.n_towers,
                   off_m:off_m + self.n_measures].sum()
        return total

    def intral_layer_total_weight(self, season, layer="bird"):
        return self.get_intra_layer_subgraph(season, layer).sum()

    def density(self, season):
        A = self._snapshots[season]
        n = self.n_total
        possible = n * (n - 1) / 2
        return (A.sum() / 2) / possible if possible > 0 else 0

    def degree_centrality(self, season):
        """Compute degree centrality for all nodes in a season snapshot."""
        A = self._snapshots[season]
        n = self.n_total
        degrees = A.sum(axis=1)
        return {
            "bird": {b: round(degrees[self.bird_idx[b]] / (n - 1), 4)
                     for b in self.bird_ids},
            "tower": {t: round(degrees[self.n_birds + self.tower_idx[t]] / (n - 1), 4)
                      for t in self.tower_ids},
            "measure": {m: round(degrees[self.n_birds + self.n_towers + self.measure_idx[m]] /
                                 (n - 1), 4) for m in self.measure_ids},
        }

    def eigenvector_centrality(self, season, max_iter=200, tol=1e-8):
        """Compute eigenvector centrality via power iteration on the supral matrix."""
        A = self._snapshots[season]
        n = self.n_total
        x = np.random.random(n)
        x = x / x.sum()
        for _ in range(max_iter):
            x_new = A @ x
            x_new = x_new / (x_new.sum() + 1e-12)
            if np.abs(x_new - x).max() < tol:
                x = x_new
                break
            x = x_new
        return {
            "bird": {b: round(x[self.bird_idx[b]], 6) for b in self.bird_ids},
            "tower": {t: round(x[self.n_birds + self.tower_idx[t]], 6) for t in self.tower_ids},
            "measure": {m: round(x[self.n_birds + self.n_towers + self.measure_idx[m]], 6)
                        for m in self.measure_ids},
        }

    def betweenness_centrality(self, season):
        """Compute betweenness centrality via Floyd-Warshall shortest paths.

        Uses Brandes-like approach for weighted networks: each path is weighted
        by inverse edge weight so higher weight = shorter distance.
        """
        A = self._snapshots[season]
        n = self.n_total
        # Convert to distance matrix: 1/w for w>0, inf otherwise
        D = np.full((n, n), np.inf)
        np.fill_diagonal(D, 0)
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] > 0:
                    D[i, j] = 1.0 / A[i, j]
                    D[j, i] = 1.0 / A[i, j]

        # Floyd-Warshall
        for k in range(n):
            dk = D[k, :]
            for i in range(n):
                dik = D[i, k]
                if dik == np.inf:
                    continue
                d_new = dik + dk
                D[i, :] = np.minimum(D[i, :], d_new)

        # Approximate betweenness: count nodes on shortest paths
        bt = np.zeros(n)
        for s in range(n):
            for t in range(n):
                if s == t:
                    continue
                for v in range(n):
                    if v == s or v == t:
                        continue
                    d_st = D[s, t]
                    d_sv, d_vt = D[s, v], D[v, t]
                    if d_st < np.inf and d_sv < np.inf and d_vt < np.inf:
                        if abs(d_sv + d_vt - d_st) < 1e-6:
                            bt[v] += 1

        norm = (n - 1) * (n - 2)
        bt /= max(norm, 1)
        return {
            "bird": {b: round(bt[self.bird_idx[b]], 6) for b in self.bird_ids},
            "tower": {t: round(bt[self.n_birds + self.tower_idx[t]], 6) for t in self.tower_ids},
            "measure": {m: round(bt[self.n_birds + self.n_towers + self.measure_idx[m]], 6)
                        for m in self.measure_ids},
        }

    def louvain_communities(self, season, gamma=1.0, max_iter=50):
        """Multilayer community detection using Louvain on the supral adjacency matrix.

        Returns communities as a dict mapping node_id -> community_label.
        """
        A = self._snapshots[season]
        n = self.n_total
        m = A.sum() / 2  # total edge weight (undirected)

        # Use dict for communities to avoid index-shift bugs
        communities = {i: i for i in range(n)}  # node -> community id
        # Reverse map: community id -> set of nodes
        comm_sets = {i: {i} for i in range(n)}

        k = A.sum(axis=1)

        for _iter in range(max_iter):
            improved = False
            nodes = np.random.permutation(n)
            for node in nodes:
                old_comm = communities[node]
                comm_sets[old_comm].discard(node)

                # Compute neighbor community weights
                neighbor_weights = defaultdict(float)
                for j in np.nonzero(A[node])[0]:
                    neighbor_weights[communities[j]] += A[node, j]

                best_dq = 0
                best_comm = old_comm
                k_node = k[node]
                for comm, w_to_comm in neighbor_weights.items():
                    if comm == old_comm:
                        continue
                    comm_total = sum(k[i] for i in comm_sets[comm])
                    dq = w_to_comm / m - gamma * comm_total * k_node / (2 * m * m)
                    if dq > best_dq:
                        best_dq = dq
                        best_comm = comm

                communities[node] = best_comm
                comm_sets[best_comm].add(node)
                if best_comm != old_comm:
                    improved = True
                if not comm_sets[old_comm]:
                    del comm_sets[old_comm]

            if not improved:
                break

        # Re-index community ids contiguously
        comm_id_map = {cid: i for i, cid in enumerate(comm_sets.keys())}
        node_to_comm = [comm_id_map[communities[i]] for i in range(n)]

        # Map to node IDs
        result = {}
        for i in range(n):
            cid = node_to_comm[i]
            if i < self.n_birds:
                result[self.bird_ids[i]] = cid
            elif i < self.n_birds + self.n_towers:
                result[self.tower_ids[i - self.n_birds]] = cid
            else:
                result[self.measure_ids[i - self.n_birds - self.n_towers]] = cid

        # Cross-layer mixing
        layer_map = {}
        for nid, comm in result.items():
            if nid in self.bird_ids:
                layer_map.setdefault(comm, set()).add("bird")
            elif nid in self.tower_ids:
                layer_map.setdefault(comm, set()).add("tower")
            else:
                layer_map.setdefault(comm, set()).add("measure")

        cross_layer_communities = sum(1 for layers in layer_map.values() if len(layers) >= 2)
        n_comms = len(comm_sets)
        modularity = _compute_modularity(A, node_to_comm, m)

        return {
            "partition": result,
            "n_communities": n_comms,
            "cross_layer_communities": cross_layer_communities,
            "modularity": round(modularity, 6),
            "community_layers": {str(k): list(v) for k, v in layer_map.items()},
        }

    def temporal_centrality_dynamics(self):
        """Compute degree centrality for all seasons, tracking temporal change."""
        dynamics = {layer: {} for layer in ["bird", "tower", "measure"]}
        for s in self.seasons:
            dc = self.degree_centrality(s)
            for layer in ["bird", "tower", "measure"]:
                for nid, val in dc[layer].items():
                    dynamics[layer].setdefault(nid, {})[s] = val
        return dynamics

    def identify_critical_nodes(self, season, top_k=5):
        """Identify critical conflict nodes: top birds by degree + towers vulnerable."""
        dc = self.degree_centrality(season)
        bird_rank = sorted(dc["bird"].items(), key=lambda x: -x[1])
        tower_rank = sorted(dc["tower"].items(), key=lambda x: -x[1])
        return {
            "season": season,
            "top_birds": bird_rank[:top_k],
            "top_towers": tower_rank[:top_k],
        }

    def summary(self, season):
        """Print a comprehensive summary for a given season."""
        s = self.get_snapshot(season)
        dc = self.degree_centrality(season)
        return {
            "season": season,
            "n_nodes": self.n_total,
            "n_birds": self.n_birds,
            "n_towers": self.n_towers,
            "n_measures": self.n_measures,
            "total_edges": int((s > 0).sum() / 2),
            "density": round(self.density(season), 6),
            "cross_layer_coupling": round(self.cross_layer_coupling_strength(season), 4),
            "top_bird_degree": sorted(dc["bird"].items(), key=lambda x: -x[1])[:5],
            "top_tower_degree": sorted(dc["tower"].items(), key=lambda x: -x[1])[:5],
        }


def _compute_modularity(A, node_to_comm, m):
    """Compute Newman-Girvan modularity."""
    if m == 0:
        return 0.0
    n = len(node_to_comm)
    k = A.sum(axis=1)
    Q = 0.0
    for i in range(n):
        for j in range(n):
            if node_to_comm[i] == node_to_comm[j]:
                Q += A[i, j] - k[i] * k[j] / (2 * m)
    return Q / (2 * m)
