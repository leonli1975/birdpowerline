"""Inductive Temporal Knowledge Graph Embedding (Inductive TComplEx).

Enables link prediction for entities unseen during training by learning a
feature-to-embedding projection matrix W_feat ∈ R^(F×D) with auxiliary 
feature alignment loss.

Key equation:
  e_unseen = f_unseen · W_feat + b_feat

The projection is trained via an auxiliary MSE loss between learned embeddings
and feature-induced embeddings for seen entities:
  L_feat = ||e_learned - (f · W_feat + b_feat)||²
"""

import numpy as np
from collections import defaultdict


class InductiveTComplEx:
    """TComplEx variant with inductive entity initialization from features.

    For seen entities: uses learned embeddings E_re, E_im.
    For unseen entities: computes embeddings as f @ W_feat + b.

    The feature projection is trained via an auxiliary alignment loss
    that matches feature-induced embeddings to learned embeddings for
    all seen entities at each epoch.
    """

    def __init__(self, n_entities, n_relations, n_timestamps, entity_features,
                 unseen_mask, dim=128, lr=0.1, reg_lambda=0.001,
                 negative_samples=10, feat_align_weight=0.5):
        self.n_e = n_entities
        self.n_r = n_relations
        self.n_t = n_timestamps
        self.dim = dim
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.neg = negative_samples
        self.feat_align_weight = feat_align_weight
        self.unseen = unseen_mask.astype(bool)
        self.seen = ~self.unseen

        rng = np.random.RandomState(42)
        scale = 0.5 / np.sqrt(dim)
        self.E_re = rng.randn(n_entities, dim) * scale
        self.E_im = rng.randn(n_entities, dim) * scale
        self.R_re = rng.randn(n_relations, dim) * scale
        self.R_im = rng.randn(n_relations, dim) * scale
        self.T_re = rng.randn(n_timestamps, dim) * scale * 0.5
        self.T_im = rng.randn(n_timestamps, dim) * scale * 0.5

        self.attr = entity_features.astype(np.float32)
        F = entity_features.shape[1]
        self.W_re = rng.randn(F, dim) * 0.05
        self.W_im = rng.randn(F, dim) * 0.05
        self.b_re = np.zeros(dim)
        self.b_im = np.zeros(dim)

        self._g2 = defaultdict(lambda: np.zeros(1))

    def _ensure_g2(self, name, shape):
        if self._g2[name].shape != shape:
            self._g2[name] = np.zeros(shape)
        return self._g2[name]

    def _adagrad(self, name, grad, param):
        g2 = self._ensure_g2(name, grad.shape)
        g2 += grad ** 2
        return self.lr * grad / (np.sqrt(g2) + 1e-8)

    def get_embedding_batch(self, eids):
        """Return (re[N,D], im[N,D]) for entity batch.
        Unseen → feature projection. Seen → learned embedding.
        """
        re_out = self.E_re[eids].copy()
        im_out = self.E_im[eids].copy()
        unseen_ids = eids[self.unseen[eids]]
        if len(unseen_ids) > 0:
            feat = self.attr[unseen_ids]
            re_out[self.unseen[eids]] = feat @ self.W_re + self.b_re
            im_out[self.unseen[eids]] = feat @ self.W_im + self.b_im
        return re_out, im_out

    def _scores_batch(self, h_vec, r_vec, t_vec, tau_vec):
        Eh_re, Eh_im = self.get_embedding_batch(h_vec)
        Et_re, Et_im = self.get_embedding_batch(t_vec)
        Rr = self.R_re[r_vec] + self.T_re[tau_vec]
        Ri = self.R_im[r_vec] + self.T_im[tau_vec]
        HR_re = Eh_re * Rr - Eh_im * Ri
        HR_im = Eh_re * Ri + Eh_im * Rr
        return np.sum(HR_re * Et_re + HR_im * Et_im, axis=1)

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
                neg_t = np.random.choice(self.n_e, size=B * self.neg, p=tail_prob)
                neg_h = np.tile(h, self.neg); neg_r = np.tile(r, self.neg)
                neg_tau = np.tile(tau, self.neg)
                neg_scores = self._scores_batch(neg_h, neg_r, neg_t, neg_tau)
                pos_rep = np.repeat(pos, self.neg)
                hinge = np.maximum(0, margin + neg_scores - pos_rep)
                active = hinge > 0
                if active.any():
                    act_idx = np.where(active)[0]
                    pos_ids = act_idx // self.neg
                    neg_t_act = neg_t[act_idx]
                    h_p, r_p, t_p = h[pos_ids], r[pos_ids], t[pos_ids]
                    tau_p = tau[pos_ids]
                    h_n = neg_h[act_idx]; r_n = neg_r[act_idx]
                    tau_n = neg_tau[act_idx]

                    dE_re = np.zeros_like(self.E_re)
                    dE_im = np.zeros_like(self.E_im)
                    dR_re = np.zeros_like(self.R_re); dR_im = np.zeros_like(self.R_im)
                    dT_re = np.zeros_like(self.T_re); dT_im = np.zeros_like(self.T_im)
                    dW_re = np.zeros_like(self.W_re); dW_im = np.zeros_like(self.W_im)
                    db_re = np.zeros_like(self.b_re); db_im = np.zeros_like(self.b_im)

                    self._acc_grad(h_p, r_p, t_p, tau_p, -1.0,
                                   dE_re, dE_im, dR_re, dR_im, dT_re, dT_im,
                                   dW_re, dW_im, db_re, db_im)
                    self._acc_grad(h_n, r_n, neg_t_act, tau_n, +1.0,
                                   dE_re, dE_im, dR_re, dR_im, dT_re, dT_im,
                                   dW_re, dW_im, db_re, db_im)

                    self.E_re -= self._adagrad("Er", dE_re/B + self.reg_lambda*self.E_re, self.E_re)
                    self.E_im -= self._adagrad("Ei", dE_im/B + self.reg_lambda*self.E_im, self.E_im)
                    self.R_re -= self._adagrad("Rr", dR_re/B + self.reg_lambda*self.R_re, self.R_re)
                    self.R_im -= self._adagrad("Ri", dR_im/B + self.reg_lambda*self.R_im, self.R_im)
                    self.T_re -= self._adagrad("Tr", dT_re/B + self.reg_lambda*self.T_re, self.T_re)
                    self.T_im -= self._adagrad("Ti", dT_im/B + self.reg_lambda*self.T_im, self.T_im)
                    self.W_re -= self._adagrad("Wr", dW_re/B + self.reg_lambda*self.W_re, self.W_re)
                    self.W_im -= self._adagrad("Wi", dW_im/B + self.reg_lambda*self.W_im, self.W_im)
                    self.b_re -= self._adagrad("br", db_re/B + self.reg_lambda*self.b_re, self.b_re)
                    self.b_im -= self._adagrad("bi", db_im/B + self.reg_lambda*self.b_im, self.b_im)

            # --- Auxiliary feature alignment loss ---
            if self.feat_align_weight > 0:
                seen_e = np.where(self.seen)[0]
                if len(seen_e) > 0:
                    Fs = self.attr[seen_e]
                    pred_re = Fs @ self.W_re + self.b_re
                    pred_im = Fs @ self.W_im + self.b_im
                    diff_re = pred_re - self.E_re[seen_e]
                    diff_im = pred_im - self.E_im[seen_e]
                    Ns = len(seen_e)
                    dWr = 2 * Fs.T @ diff_re / Ns
                    dWi = 2 * Fs.T @ diff_im / Ns
                    dbr = 2 * diff_re.mean(axis=0)
                    dbi = 2 * diff_im.mean(axis=0)
                    self.W_re -= self._adagrad("Wr", self.feat_align_weight*dWr + self.reg_lambda*self.W_re, self.W_re)
                    self.W_im -= self._adagrad("Wi", self.feat_align_weight*dWi + self.reg_lambda*self.W_im, self.W_im)
                    self.b_re -= self._adagrad("br", self.feat_align_weight*dbr + self.reg_lambda*self.b_re, self.b_re)
                    self.b_im -= self._adagrad("bi", self.feat_align_weight*dbi + self.reg_lambda*self.b_im, self.b_im)
        return {"loss": []}

    def _acc_grad(self, h_vec, r_vec, t_vec, tau_vec, g,
                   dE_re, dE_im, dR_re, dR_im, dT_re, dT_im,
                   dW_re, dW_im, db_re, db_im):
        Eh_re, Eh_im = self.get_embedding_batch(h_vec)
        Et_re, Et_im = self.get_embedding_batch(t_vec)
        Rr = self.R_re[r_vec] + self.T_re[tau_vec]
        Ri = self.R_im[r_vec] + self.T_im[tau_vec]
        HR_re = Eh_re * Rr - Eh_im * Ri
        HR_im = Eh_re * Ri + Eh_im * Rr

        dEh_re = g * (Rr * Et_re + Ri * Et_im)
        dEh_im = g * (-Ri * Et_re + Rr * Et_im)
        dEt_re = g * HR_re; dEt_im = g * HR_im
        dr = g * (Eh_re * Et_re + Eh_im * Et_im)
        di = g * (Eh_re * Et_im - Eh_im * Et_re)

        h_unseen = self.unseen[h_vec]; t_unseen = self.unseen[t_vec]

        # Seen: accumulate to E_re/E_im
        if (~h_unseen).any():
            np.add.at(dE_re, h_vec[~h_unseen], dEh_re[~h_unseen])
            np.add.at(dE_im, h_vec[~h_unseen], dEh_im[~h_unseen])
        if (~t_unseen).any():
            np.add.at(dE_re, t_vec[~t_unseen], dEt_re[~t_unseen])
            np.add.at(dE_im, t_vec[~t_unseen], dEt_im[~t_unseen])

        # Unseen: route through feature projection
        if h_unseen.any():
            fh = self.attr[h_vec[h_unseen]]
            dW_re += fh.T @ dEh_re[h_unseen]; dW_im += fh.T @ dEh_im[h_unseen]
            db_re += dEh_re[h_unseen].sum(axis=0); db_im += dEh_im[h_unseen].sum(axis=0)
        if t_unseen.any():
            ft = self.attr[t_vec[t_unseen]]
            dW_re += ft.T @ dEt_re[t_unseen]; dW_im += ft.T @ dEt_im[t_unseen]
            db_re += dEt_re[t_unseen].sum(axis=0); db_im += dEt_im[t_unseen].sum(axis=0)

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
        return {"MRR": float(np.mean(1.0/ranks)),
                "Hits@1": float(np.mean(ranks <= 1)),
                "Hits@3": float(np.mean(ranks <= 3)),
                "Hits@10": float(np.mean(ranks <= 10)),
                "MeanRank": float(np.mean(ranks))}


def build_entity_features(e2i, i2e):
    """Extract 12-dim feature vectors for all entity types.

    Power lines: voltage(3), poles, length, nesting_rate, near_wetland,
                 near_meadow, elev_zone(2), has_nests, activity.
    Bird species: protection(2), body_size(2), habitat(3), residency(2), baseline.
    Measures: type(3), baseline.
    All vectors are L2-normalized.
    """
    from src.kg_data import POWER_LINES, LINE_ELEVATION, BIRD_SPECIES, MEASURES

    n_e = len(e2i)
    F = 12
    feat = np.zeros((n_e, F))
    line_names = {name for name in i2e.values() if name.startswith('line_')}

    for eid, name in i2e.items():
        if name in line_names:
            info = POWER_LINES.get(name, {})
            v = info.get('voltage', 10)
            if v == 110: feat[eid, 0] = 1
            elif v == 35: feat[eid, 1] = 1
            else: feat[eid, 2] = 1
            feat[eid, 3] = min(info.get('poles', 0) / 300.0, 1.0)
            feat[eid, 4] = min(info.get('length_km', 0) / 100.0, 1.0)
            feat[eid, 5] = info.get('nest_rate', 0.0)
            if name in ('line_aji', 'line_ruozhen', 'line_ruotang'):
                feat[eid, 6] = 1
            if name in ('line_ruozhen', 'line_heize', 'line_ruoa'):
                feat[eid, 7] = 1
            elev = LINE_ELEVATION.get(name, 'elev_mid')
            if 'mid' in elev: feat[eid, 8] = 1
            elif 'high' in elev: feat[eid, 9] = 1
            if info.get('artificial_nests', 0) > 0:
                feat[eid, 10] = 1
            if info.get('activity', 0) > 0:
                feat[eid, 11] = info['activity'] / 5.0

        elif name in BIRD_SPECIES:
            info = BIRD_SPECIES[name]
            if info.get('class') == 'I': feat[eid, 0] = 1
            elif info.get('class') == 'II': feat[eid, 1] = 1
            bs = info.get('body_size', '')
            if bs == 'large': feat[eid, 2] = 1
            elif bs == 'medium': feat[eid, 3] = 1
            hab = info.get('habitat', '')
            if hab == 'wetland': feat[eid, 4] = 1
            elif hab == 'grassland': feat[eid, 5] = 1
            elif hab == 'mountain': feat[eid, 6] = 1
            res = info.get('residency', '')
            if res == 'resident': feat[eid, 7] = 1
            elif res == 'summer_visitor': feat[eid, 8] = 1
            feat[eid, 9] = 0.5

        elif name in MEASURES:
            info = MEASURES[name]
            tp = info.get('type', '')
            if tp == 'physical_barrier': feat[eid, 0] = 1
            elif tp == 'deterrent': feat[eid, 1] = 1
            elif tp == 'attractant': feat[eid, 2] = 1
            feat[eid, 3] = 0.5

    # L2-normalize non-zero rows
    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return feat / norms
