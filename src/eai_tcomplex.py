"""Enrichment-Aware Inductive TComplEx (EAI-TComplEx).

Extension A: Inductive entity initialization from features.
Extension B: Enrichment-aware scoring with attribute similarity.

EAI-TComplEx(h,r,t,τ) = Re(⟨e_h ⊙ (w_r + v_τ), conj(e_t)⟩) + λ · cos(attr_h, attr_t)
"""

import numpy as np
import time


def build_entity_features(e2i, i2e, triples):
    """Extract feature vectors for power line entities from KG structure.

    Each line entity gets a feature vector encoding:
      - voltage level (one-hot: 10, 35, 110)
      - elevation zone (one-hot: low, mid, high, alpine) 
      - near_habitat indicators (wetland_marsh, alpine_meadow)
      - normalized pole count (0-1 from entity attributes)
      - collision risk indicators
    """
    n_e = len(e2i)
    # Fixed feature dimensions
    feat_dim = 16
    features = np.zeros((n_e, feat_dim))

    # Line entity IDs to process
    line_ids = [eid for eid, name in i2e.items() if name.startswith('line_')]

    # Build attribute lookup from static triples
    # We scan all static triples to build per-entity attribute maps
    # Format: (h, r, t, tb, te) — static if tb=te=None
    # Since we have tensor-format triples here, we need raw KG triples
    # Let's build features directly from entity names in i2e

    # Voltage encoding (offset 0-2)
    voltage_map = {'volt_10': 0, 'volt_35': 1, 'volt_110': 2}
    # Elevation encoding (offset 3-6)
    elev_map = {'elev_low': 3, 'elev_mid': 4, 'elev_high': 5, 'elev_alpine': 6}
    # Habitat encoding (offset 7-8)
    hab_map = {'hab_wetland_marsh': 7, 'hab_alpine_meadow': 8}
    # Other features (9-15): collision risk, pole count, etc.

    # Scan existing KG — use the mapping we know from kg_data
    from src.kg_data import POWER_LINES, LINE_ELEVATION

    for eid, name in i2e.items():
        if not name.startswith('line_'):
            continue

        info = POWER_LINES.get(name, {})

        # Voltage
        volt_key = f"volt_{info.get('voltage', 10)}"
        if volt_key in voltage_map:
            features[eid, voltage_map[volt_key]] = 1.0

        # Elevation
        elev_key = LINE_ELEVATION.get(name, 'elev_mid')
        if elev_key in elev_map:
            features[eid, elev_map[elev_key]] = 1.0

        # Near wetland marsh
        if name in ('line_aji', 'line_ruozhen', 'line_ruotang'):
            features[eid, 7] = 1.0  # wetland
        if name in ('line_ruozhen', 'line_heize', 'line_ruoa'):
            features[eid, 8] = 1.0  # alpine_meadow

        # Collision risk (based on voltage)
        if info.get('voltage', 10) == 110:
            features[eid, 9] = 1.0  # high risk
        elif info.get('voltage', 10) == 35:
            features[eid, 10] = 1.0  # moderate
        else:
            features[eid, 11] = 1.0  # low

        # Normalized pole count
        poles = info.get('poles', 0)
        if poles > 0:
            features[eid, 12] = min(poles / 300.0, 1.0)

        # Length in km
        length = info.get('length_km', 0)
        if length > 0:
            features[eid, 13] = min(length / 100.0, 1.0)

        # Nesting rate
        nest_rate = info.get('nest_rate', 0)
        features[eid, 14] = nest_rate

        # Has measures deployed (indicator: yes from 2020+)
        # We don't have per-line deployment info in features, just set a default
        features[eid, 15] = 0.5  # uncertain

    # For non-line entities, set a small random baseline
    for eid in range(n_e):
        if np.sum(features[eid]) == 0:
            features[eid, -1] = 0.01  # minimal signal

    return features


class EAITComplEx:
    """Enrichment-Aware Inductive TComplEx.

    f(h,r,t,τ) = Re(⟨e_h ⊙ (w_r + v_τ), conj(e_t)⟩) + λ · cos(attr_h, attr_t)
    """

    def __init__(self, n_entities, n_relations, n_timestamps, entity_features,
                 dim=128, reg_lambda=0.001, lr=0.1, negative_samples=10,
                 lambda_attr=0.1):
        self.n_e = n_entities; self.n_r = n_relations; self.n_t = n_timestamps
        self.dim = dim; self.reg_lambda = reg_lambda; self.lr = lr
        self.neg = negative_samples; self.lambda_attr = lambda_attr

        rng = np.random.RandomState(42)
        scale = 0.5 / np.sqrt(dim)
        self.E_re = rng.randn(n_entities, dim) * scale
        self.E_im = rng.randn(n_entities, dim) * scale
        self.R_re = rng.randn(n_relations, dim) * scale
        self.R_im = rng.randn(n_relations, dim) * scale
        self.T_re = rng.randn(n_timestamps, dim) * scale * 0.5
        self.T_im = rng.randn(n_timestamps, dim) * scale * 0.5

        # Inductive feature-to-embedding MLP (single linear layer for simplicity)
        feat_dim = entity_features.shape[1]
        self.attr = entity_features.copy()
        # Normalize attribute vectors for cosine similarity
        self.attr_norm = self.attr / (np.linalg.norm(self.attr, axis=1, keepdims=True) + 1e-12)
        # Feature projection matrix: feat_dim x 2*dim (re+im)
        self.W_feat_re = rng.randn(feat_dim, dim) * 0.01
        self.W_feat_im = rng.randn(feat_dim, dim) * 0.01
        self.b_feat_re = np.zeros(dim)
        self.b_feat_im = np.zeros(dim)

        # AdaGrad
        self._g2 = {}
        for k in ['Er','Ei','Rr','Ri','Tr','Ti','Wfr','Wfi','bfr','bfi']:
            self._g2[k] = 0.0

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def _induce_embedding(self, eid):
        """Compute inductive embedding from entity features (for unseen entities)."""
        feat = self.attr[eid]  # [F]
        e_re = feat @ self.W_feat_re + self.b_feat_re  # [D]
        e_im = feat @ self.W_feat_im + self.b_feat_im  # [D]
        return e_re, e_im

    def get_entity_embedding(self, eid, seen_mask=None):
        """Get entity embedding: learned if seen, induced from features if unseen."""
        if seen_mask is not None and seen_mask[eid]:
            return self.E_re[eid], self.E_im[eid]
        return self._induce_embedding(eid)

    def score(self, h_idx, r_idx, t_idx, tau, seen_h=True, seen_t=True):
        h_re, h_im = self.get_entity_embedding(h_idx, np.array([seen_h]))
        t_re, t_im = self.get_entity_embedding(t_idx, np.array([seen_t]))
        h_re, h_im = h_re.reshape(-1), h_im.reshape(-1)
        t_re, t_im = t_re.reshape(-1), t_re.reshape(-1)

        r_re = self.R_re[r_idx] + self.T_re[tau]
        r_im = self.R_im[r_idx] + self.T_im[tau]

        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        tkg_score = float(np.sum(hr_re * t_re + hr_im * t_im))

        # Enrichment-aware bonus
        attr_score = float(np.dot(self.attr_norm[h_idx], self.attr_norm[t_idx]))

        return tkg_score + self.lambda_attr * attr_score

    def _scores_batch(self, h_vec, r_vec, t_vec, tau_vec):
        """Vectorized scoring. Uses learned embeddings."""
        Eh_re = self.E_re[h_vec]; Eh_im = self.E_im[h_vec]
        Et_re = self.E_re[t_vec]; Et_im = self.E_im[t_vec]
        Rr = self.R_re[r_vec] + self.T_re[tau_vec]
        Ri = self.R_im[r_vec] + self.T_im[tau_vec]
        HR_re = Eh_re * Rr - Eh_im * Ri
        HR_im = Eh_re * Ri + Eh_im * Rr
        tkg_scores = np.sum(HR_re * Et_re + HR_im * Et_im, axis=1)

        # Attribute similarity bonus
        attr_scores = np.sum(self.attr_norm[h_vec] * self.attr_norm[t_vec], axis=1)

        return tkg_scores + self.lambda_attr * attr_scores

    def _adagrad_update(self, name, grad, param):
        key = f'{name}_{id(param)}'
        if key not in self._g2 or isinstance(self._g2[key], float):
            self._g2[key] = np.zeros_like(grad)
        self._g2[key] += grad ** 2
        return self.lr * grad / (np.sqrt(self._g2[key]) + 1e-8)

    def fit(self, triples, epochs=100, batch_size=512, margin=1.0, verbose=False):
        triples = np.array(triples, dtype=np.int64)
        n_train = len(triples)
        tail_counts = np.bincount(triples[:, 2], minlength=self.n_e)
        tail_prob = (tail_counts.astype(float) + 1) ** 0.75
        tail_prob /= tail_prob.sum()

        for epoch in range(epochs):
            perm = np.random.permutation(n_train)
            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                B = len(idx)
                h, r, t, tau = triples[idx].T
                pos = self._scores_batch(h, r, t, tau)

                neg_t_all = np.random.choice(self.n_e, size=B * self.neg, p=tail_prob)
                neg_scores = self._scores_batch(
                    np.tile(h, self.neg), np.tile(r, self.neg),
                    neg_t_all, np.tile(tau, self.neg))
                pos_rep = np.repeat(pos, self.neg)
                hinge = np.maximum(0, margin + neg_scores - pos_rep)
                active = hinge > 0
                if active.any():
                    act_idx = np.where(active)[0]
                    pos_indices = act_idx // self.neg
                    neg_t_act = neg_t_all[act_idx]
                    h_p, r_p, t_p = h[pos_indices], r[pos_indices], t[pos_indices]
                    tau_p = tau[pos_indices]
                    h_n = np.tile(h, self.neg)[act_idx]
                    r_n = np.tile(r, self.neg)[act_idx]
                    tau_n = np.tile(tau, self.neg)[act_idx]

                    dE_re = np.zeros_like(self.E_re); dE_im = np.zeros_like(self.E_im)
                    dR_re = np.zeros_like(self.R_re); dR_im = np.zeros_like(self.R_im)
                    dT_re = np.zeros_like(self.T_re); dT_im = np.zeros_like(self.T_im)

                    # Positive gradients (same as TComplEx)
                    self._acc_grad(h_p, r_p, t_p, tau_p, -1.0,
                                   dE_re, dE_im, dR_re, dR_im, dT_re, dT_im)
                    # Negative gradients
                    self._acc_grad(h_n, r_n, neg_t_act, tau_n, +1.0,
                                   dE_re, dE_im, dR_re, dR_im, dT_re, dT_im)

                    Bf = B
                    self.E_re -= self._adagrad_update("Er", dE_re/Bf + self.reg_lambda*self.E_re, self.E_re)
                    self.E_im -= self._adagrad_update("Ei", dE_im/Bf + self.reg_lambda*self.E_im, self.E_im)
                    self.R_re -= self._adagrad_update("Rr", dR_re/Bf + self.reg_lambda*self.R_re, self.R_re)
                    self.R_im -= self._adagrad_update("Ri", dR_im/Bf + self.reg_lambda*self.R_im, self.R_im)
                    self.T_re -= self._adagrad_update("Tr", dT_re/Bf + self.reg_lambda*self.T_re, self.T_re)
                    self.T_im -= self._adagrad_update("Ti", dT_im/Bf + self.reg_lambda*self.T_im, self.T_im)
        return {"loss": []}

    def _acc_grad(self, h_vec, r_vec, t_vec, tau_vec, g,
                   dE_re, dE_im, dR_re, dR_im, dT_re, dT_im):
        Eh_re = self.E_re[h_vec]; Eh_im = self.E_im[h_vec]
        Et_re = self.E_re[t_vec]; Et_im = self.E_im[t_vec]
        Rr = self.R_re[r_vec] + self.T_re[tau_vec]
        Ri = self.R_im[r_vec] + self.T_im[tau_vec]
        HR_re = Eh_re * Rr - Eh_im * Ri
        HR_im = Eh_re * Ri + Eh_im * Rr
        dEh_re = g * (Rr * Et_re + Ri * Et_im)
        dEh_im = g * (-Ri * Et_re + Rr * Et_im)
        dEt_re = g * HR_re; dEt_im = g * HR_im
        dr = g * (Eh_re * Et_re + Eh_im * Et_im)
        di = g * (Eh_re * Et_im - Eh_im * Et_re)
        np.add.at(dE_re, h_vec, dEh_re); np.add.at(dE_im, h_vec, dEh_im)
        np.add.at(dE_re, t_vec, dEt_re); np.add.at(dE_im, t_vec, dEt_im)
        np.add.at(dR_re, r_vec, dr); np.add.at(dR_im, r_vec, di)
        np.add.at(dT_re, tau_vec, dr); np.add.at(dT_im, tau_vec, di)

    def evaluate_link_prediction(self, test_triples, train_filter):
        triples = np.array(test_triples, dtype=np.int64)
        n_test = len(triples)
        if n_test == 0:
            return {"MRR": 0, "Hits@1": 0, "Hits@3": 0, "Hits@10": 0, "MeanRank": 0}

        ranks = []
        for i in range(n_test):
            h, r, t, tau = triples[i]
            hi, ri, ti, ti_i = int(h), int(r), int(t), int(tau)
            h_vec = np.full(self.n_e, hi, dtype=np.int64)
            t_vec = np.arange(self.n_e, dtype=np.int64)
            r_vec = np.full(self.n_e, ri, dtype=np.int64)
            tau_vec = np.full(self.n_e, ti_i, dtype=np.int64)
            scores = self._scores_batch(h_vec, r_vec, t_vec, tau_vec)
            if train_filter is not None:
                for cand in range(self.n_e):
                    if cand != ti and (hi, ri, cand, ti_i) in train_filter:
                        scores[cand] = -1e10
            rank = 1 + np.sum(scores > scores[ti])
            ranks.append(rank)

        ranks = np.array(ranks)
        return {
            "MRR": float(np.mean(1.0/ranks)), "Hits@1": float(np.mean(ranks<=1)),
            "Hits@3": float(np.mean(ranks<=3)), "Hits@10": float(np.mean(ranks<=10)),
            "MeanRank": float(np.mean(ranks)),
        }
