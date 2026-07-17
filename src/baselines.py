"""Baseline models and ablation experiments for temporal KG.

Implements:
  - ComplEx (static, no time modulation)
  - TransE (additive translation)
  - DistMult (simple bilinear)
  - Ablation variants of TComplEx

All use the same AdaGrad + margin loss training for fair comparison.
"""

import numpy as np


class ComplEx:
    """Static ComplEx: f(h,r,t) = Re(⟨e_h ⊙ w_r, conj(e_t)⟩). No time modulation."""

    def __init__(self, n_entities, n_relations, n_timestamps, dim=128,
                 reg_lambda=0.001, lr=0.1, negative_samples=10):
        self.n_e = n_entities
        self.n_r = n_relations
        self.n_t = n_timestamps
        self.dim = dim
        self.reg_lambda = reg_lambda
        self.lr = lr
        self.neg = negative_samples
        rng = np.random.RandomState(42)
        scale = 0.5 / np.sqrt(dim)
        self.E_re = rng.randn(n_entities, dim) * scale
        self.E_im = rng.randn(n_entities, dim) * scale
        self.R_re = rng.randn(n_relations, dim) * scale
        self.R_im = rng.randn(n_relations, dim) * scale
        self._g2 = {k: np.zeros_like(v) for k, v in
                    [("Er", self.E_re), ("Ei", self.E_im),
                     ("Rr", self.R_re), ("Ri", self.R_im)]}

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def score(self, h_idx, r_idx, t_idx, tau=None):
        h_re, h_im = self.E_re[h_idx], self.E_im[h_idx]
        t_re, t_im = self.E_re[t_idx], self.E_im[t_idx]
        r_re, r_im = self.R_re[r_idx], self.R_im[r_idx]
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        return float(np.sum(hr_re * t_re + hr_im * t_im))

    def _scores_batch(self, h_vec, r_vec, t_vec, tau_vec):
        Eh_re = self.E_re[h_vec]
        Eh_im = self.E_im[h_vec]
        Et_re = self.E_re[t_vec]
        Et_im = self.E_im[t_vec]
        Rr = self.R_re[r_vec]
        Ri = self.R_im[r_vec]
        HR_re = Eh_re * Rr - Eh_im * Ri
        HR_im = Eh_re * Ri + Eh_im * Rr
        return np.sum(HR_re * Et_re + HR_im * Et_im, axis=1)

    def _adagrad_update(self, name, grad):
        self._g2[name] += grad ** 2
        return self.lr * grad / (np.sqrt(self._g2[name]) + 1e-8)

    def _acc_grad(self, h_vec, r_vec, t_vec, tau_vec, g, dE_re, dE_im, dR_re, dR_im):
        Eh_re = self.E_re[h_vec]
        Eh_im = self.E_im[h_vec]
        Et_re = self.E_re[t_vec]
        Et_im = self.E_im[t_vec]
        Rr = self.R_re[r_vec]
        Ri = self.R_im[r_vec]
        HR_re = Eh_re * Rr - Eh_im * Ri
        HR_im = Eh_re * Ri + Eh_im * Rr
        dEh_re = g * (Rr * Et_re + Ri * Et_im)
        dEh_im = g * (-Ri * Et_re + Rr * Et_im)
        dEt_re = g * HR_re
        dEt_im = g * HR_im
        dr = g * (Eh_re * Et_re + Eh_im * Et_im)
        di = g * (Eh_re * Et_im - Eh_im * Et_re)
        np.add.at(dE_re, h_vec, dEh_re)
        np.add.at(dE_im, h_vec, dEh_im)
        np.add.at(dE_re, t_vec, dEt_re)
        np.add.at(dE_im, t_vec, dEt_im)
        np.add.at(dR_re, r_vec, dr)
        np.add.at(dR_im, r_vec, di)

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
                neg_scores = self._scores_batch(np.tile(h, self.neg),
                                                np.tile(r, self.neg),
                                                neg_t_all,
                                                np.tile(tau, self.neg))
                pos_rep = np.repeat(pos, self.neg)
                hinge = np.maximum(0, margin + neg_scores - pos_rep)
                active = hinge > 0
                if active.any():
                    act_idx = np.where(active)[0]
                    pos_indices = act_idx // self.neg
                    dE_re = np.zeros_like(self.E_re)
                    dE_im = np.zeros_like(self.E_im)
                    dR_re = np.zeros_like(self.R_re)
                    dR_im = np.zeros_like(self.R_im)
                    self._acc_grad(h[pos_indices], r[pos_indices],
                                   t[pos_indices], tau[pos_indices],
                                   -1.0, dE_re, dE_im, dR_re, dR_im)
                    self._acc_grad(np.tile(h, self.neg)[act_idx],
                                   np.tile(r, self.neg)[act_idx],
                                   neg_t_all[act_idx],
                                   np.tile(tau, self.neg)[act_idx],
                                   +1.0, dE_re, dE_im, dR_re, dR_im)
                    Bf = B
                    self.E_re -= self._adagrad_update("Er", dE_re / Bf + self.reg_lambda * self.E_re)
                    self.E_im -= self._adagrad_update("Ei", dE_im / Bf + self.reg_lambda * self.E_im)
                    self.R_re -= self._adagrad_update("Rr", dR_re / Bf + self.reg_lambda * self.R_re)
                    self.R_im -= self._adagrad_update("Ri", dR_im / Bf + self.reg_lambda * self.R_im)
        return {"loss": []}

    def evaluate_link_prediction(self, test_triples, train_filter):
        return _evaluate_generic(self, test_triples, train_filter)


class TransE:
    """TransE: f(h,r,t) = -||e_h + w_r - e_t||_L1. Additive model."""

    def __init__(self, n_entities, n_relations, n_timestamps, dim=128,
                 reg_lambda=0.001, lr=0.1, negative_samples=10):
        self.n_e = n_entities
        self.n_r = n_relations
        self.n_t = n_timestamps
        self.dim = dim
        self.reg_lambda = reg_lambda
        self.lr = lr
        self.neg = negative_samples
        rng = np.random.RandomState(42)
        scale = 0.5 / np.sqrt(dim)
        self.E = rng.randn(n_entities, dim) * scale
        self.R = rng.randn(n_relations, dim) * scale
        # Normalize entity embeddings
        self.E = self.E / (np.linalg.norm(self.E, axis=1, keepdims=True) + 1e-12)
        self._g2 = {"E": np.zeros_like(self.E), "R": np.zeros_like(self.R)}

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def score(self, h_idx, r_idx, t_idx, tau=None):
        return -np.sum(np.abs(self.E[h_idx] + self.R[r_idx] - self.E[t_idx]))

    def _scores_batch(self, h_vec, r_vec, t_vec, tau_vec):
        scores = -np.sum(np.abs(self.E[h_vec] + self.R[r_vec] - self.E[t_vec]), axis=1)
        return scores

    def _adagrad_update(self, name, grad):
        self._g2[name] += grad ** 2
        return self.lr * grad / (np.sqrt(self._g2[name]) + 1e-8)

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
                neg_scores = self._scores_batch(np.tile(h, self.neg),
                                                np.tile(r, self.neg),
                                                neg_t_all,
                                                np.tile(tau, self.neg))
                pos_rep = np.repeat(pos, self.neg)
                hinge = np.maximum(0, margin + neg_scores - pos_rep)
                active = hinge > 0
                if active.any():
                    act_idx = np.where(active)[0]
                    pos_indices = np.repeat(np.arange(B), self.neg)[act_idx]
                    neg_t_act = neg_t_all[act_idx]
                    h_act_p = h[pos_indices]
                    r_act_p = r[pos_indices]
                    t_act_p = t[pos_indices]
                    h_act_n = np.tile(h, self.neg)[act_idx]
                    r_act_n = np.tile(r, self.neg)[act_idx]

                    # Gradients for L1 norm: sign(x) for each component
                    d_pos = self.E[h_act_p] + self.R[r_act_p] - self.E[t_act_p]  # [N,D]
                    d_neg = self.E[h_act_n] + self.R[r_act_n] - self.E[neg_t_act]  # [N,D]
                    g_pos = np.sign(d_pos)  # [N,D]
                    g_neg = np.sign(d_neg)  # [N,D]

                    dE = np.zeros_like(self.E)
                    dR = np.zeros_like(self.R)
                    np.add.at(dE, h_act_p, -g_pos)
                    np.add.at(dE, t_act_p, g_pos)
                    np.add.at(dR, r_act_p, -g_pos)
                    np.add.at(dE, h_act_n, g_neg)
                    np.add.at(dE, neg_t_act, -g_neg)
                    np.add.at(dR, r_act_n, g_neg)

                    Bf = B
                    self.E -= self._adagrad_update("E", dE / Bf + self.reg_lambda * self.E)
                    self.R -= self._adagrad_update("R", dR / Bf + self.reg_lambda * self.R)
                    # Renormalize
                    self.E = self.E / (np.linalg.norm(self.E, axis=1, keepdims=True) + 1e-12)
        return {"loss": []}

    def evaluate_link_prediction(self, test_triples, train_filter):
        return _evaluate_generic(self, test_triples, train_filter)


class DistMult:
    """DistMult: f(h,r,t) = ⟨e_h, w_r, e_t⟩. Simple trilinear dot product."""

    def __init__(self, n_entities, n_relations, n_timestamps, dim=128,
                 reg_lambda=0.001, lr=0.1, negative_samples=10):
        self.n_e = n_entities
        self.n_r = n_relations
        self.n_t = n_timestamps
        self.dim = dim
        self.reg_lambda = reg_lambda
        self.lr = lr
        self.neg = negative_samples
        rng = np.random.RandomState(42)
        scale = 0.5 / np.sqrt(dim)
        self.E = rng.randn(n_entities, dim) * scale
        self.R = rng.randn(n_relations, dim) * scale
        self._g2 = {"E": np.zeros_like(self.E), "R": np.zeros_like(self.R)}

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def score(self, h_idx, r_idx, t_idx, tau=None):
        return float(np.sum(self.E[h_idx] * self.R[r_idx] * self.E[t_idx]))

    def _scores_batch(self, h_vec, r_vec, t_vec, tau_vec):
        return np.sum(self.E[h_vec] * self.R[r_vec] * self.E[t_vec], axis=1)

    def _adagrad_update(self, name, grad):
        self._g2[name] += grad ** 2
        return self.lr * grad / (np.sqrt(self._g2[name]) + 1e-8)

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
                neg_h = np.tile(h, self.neg)
                neg_r = np.tile(r, self.neg)
                neg_scores = self._scores_batch(neg_h, neg_r, neg_t_all, np.tile(tau, self.neg))
                pos_rep = np.repeat(pos, self.neg)
                hinge = np.maximum(0, margin + neg_scores - pos_rep)
                active = hinge > 0
                if active.any():
                    act_idx = np.where(active)[0]
                    pos_indices = np.repeat(np.arange(B), self.neg)[act_idx]
                    neg_t_act = neg_t_all[act_idx]
                    h_p = h[pos_indices]
                    r_p = r[pos_indices]
                    t_p = t[pos_indices]
                    h_n = neg_h[act_idx]
                    r_n = neg_r[act_idx]

                    dE = np.zeros_like(self.E)
                    dR = np.zeros_like(self.R)
                    # d/d(e_h): w_r * e_t,  d/d(e_t): e_h * w_r,  d/d(w_r): e_h * e_t
                    dEh_pos = -self.R[r_p] * self.E[t_p]
                    dEt_pos = -self.E[h_p] * self.R[r_p]
                    dR_pos = -self.E[h_p] * self.E[t_p]
                    dEh_neg = self.R[r_n] * self.E[neg_t_act]
                    dEt_neg = self.E[h_n] * self.R[r_n]
                    dR_neg = self.E[h_n] * self.E[neg_t_act]
                    np.add.at(dE, h_p, dEh_pos)
                    np.add.at(dE, t_p, dEt_pos)
                    np.add.at(dR, r_p, dR_pos)
                    np.add.at(dE, h_n, dEh_neg)
                    np.add.at(dE, neg_t_act, dEt_neg)
                    np.add.at(dR, r_n, dR_neg)

                    Bf = B
                    self.E -= self._adagrad_update("E", dE / Bf + self.reg_lambda * self.E)
                    self.R -= self._adagrad_update("R", dR / Bf + self.reg_lambda * self.R)
        return {"loss": []}

    def evaluate_link_prediction(self, test_triples, train_filter):
        return _evaluate_generic(self, test_triples, train_filter)


def _evaluate_generic(model, test_triples, train_filter):
    """Generic filtered evaluation for any model with score() and _scores_batch()."""
    triples = np.array(test_triples, dtype=np.int64)
    n_test = len(triples)
    if n_test == 0:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "MeanRank": 0.0}

    ranks = []
    for i in range(n_test):
        h, r, t, tau = triples[i]
        hi, ri, ti, ti_i = int(h), int(r), int(t), int(tau)

        h_vec = np.full(model.n_e, hi, dtype=np.int64)
        t_vec = np.arange(model.n_e, dtype=np.int64)
        r_vec = np.full(model.n_e, ri, dtype=np.int64)
        tau_vec = np.full(model.n_e, ti_i, dtype=np.int64)

        scores = model._scores_batch(h_vec, r_vec, t_vec, tau_vec)

        if train_filter is not None:
            for cand in range(model.n_e):
                if cand != ti and (hi, ri, cand, ti_i) in train_filter:
                    scores[cand] = -1e10

        rank = 1 + np.sum(scores > scores[ti])
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "MRR": float(np.mean(1.0 / ranks)),
        "Hits@1": float(np.mean(ranks <= 1)),
        "Hits@3": float(np.mean(ranks <= 3)),
        "Hits@10": float(np.mean(ranks <= 10)),
        "MeanRank": float(np.mean(ranks)),
    }


# ============================================================
# SOTA Baselines: RotatE, TuckER-T, ATiSE
# ============================================================

class RotatE:
    """RotatE: f(h,r,t) = -||e_h ∘ w_r - e_t|| where ∘ is rotation in complex plane.

    Key property: can model symmetry (r = -r), inversion, and composition.
    """

    def __init__(self, n_entities, n_relations, n_timestamps, dim=128,
                 reg_lambda=0.001, lr=0.1, negative_samples=10):
        self.n_e = n_entities; self.n_r = n_relations; self.n_t = n_timestamps
        self.dim = dim; self.reg_lambda = reg_lambda; self.lr = lr; self.neg = negative_samples
        rng = np.random.RandomState(42)
        scale = 0.5 / np.sqrt(dim)
        # Entity: complex (2*dim real = dim complex)
        self.E_re = rng.randn(n_entities, dim) * scale
        self.E_im = rng.randn(n_entities, dim) * scale
        # Relation: phase angles (constrained to unit circle)
        phase = rng.uniform(-np.pi, np.pi, (n_relations, dim))
        self.R_re = np.cos(phase)  # r_re[d] = cos(theta_d)
        self.R_im = np.sin(phase)  # r_im[d] = sin(theta_d)
        self._g2 = {"Er": np.zeros_like(self.E_re), "Ei": np.zeros_like(self.E_im),
                    "Rr": np.zeros_like(self.R_re), "Ri": np.zeros_like(self.R_im)}

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def score(self, h_idx, r_idx, t_idx, tau=None):
        h_re, h_im = self.E_re[h_idx], self.E_im[h_idx]
        t_re, t_im = self.E_re[t_idx], self.E_im[t_idx]
        r_re, r_im = self.R_re[r_idx], self.R_im[r_idx]
        # Rotation: h_re*r_re - h_im*r_im, h_re*r_im + h_im*r_re
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        diff = np.sum((hr_re - t_re)**2 + (hr_im - t_im)**2)
        return float(-diff)

    def _scores_batch(self, h_vec, r_vec, t_vec, tau_vec):
        Eh_re = self.E_re[h_vec]; Eh_im = self.E_im[h_vec]
        Et_re = self.E_re[t_vec]; Et_im = self.E_im[t_vec]
        Rr = self.R_re[r_vec]; Ri = self.R_im[r_vec]
        HR_re = Eh_re * Rr - Eh_im * Ri
        HR_im = Eh_re * Ri + Eh_im * Rr
        diff = np.sum((HR_re - Et_re)**2 + (HR_im - Et_im)**2, axis=1)
        return -diff

    def _adagrad_update(self, name, grad):
        self._g2[name] += grad ** 2
        step = self.lr * grad / (np.sqrt(self._g2[name]) + 1e-8)
        return step

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
                neg_scores = self._scores_batch(np.tile(h, self.neg),
                                                np.tile(r, self.neg),
                                                neg_t_all, np.tile(tau, self.neg))
                pos_rep = np.repeat(pos, self.neg)
                hinge = np.maximum(0, margin + neg_scores - pos_rep)
                active = hinge > 0
                if active.any():
                    act_idx = np.where(active)[0]
                    pos_indices = np.repeat(np.arange(B), self.neg)[act_idx]
                    neg_t_act = neg_t_all[act_idx]
                    h_p, r_p, t_p = h[pos_indices], r[pos_indices], t[pos_indices]
                    h_n, r_n = np.tile(h, self.neg)[act_idx], np.tile(r, self.neg)[act_idx]

                    dE_re = np.zeros_like(self.E_re); dE_im = np.zeros_like(self.E_im)
                    dR_re = np.zeros_like(self.R_re); dR_im = np.zeros_like(self.R_im)

                    # Positive: push h∘r closer to t
                    Eh_p_re, Eh_p_im = self.E_re[h_p], self.E_im[h_p]
                    Et_p_re, Et_p_im = self.E_re[t_p], self.E_im[t_p]
                    Rr_p, Ri_p = self.R_re[r_p], self.R_im[r_p]
                    HRp_re = Eh_p_re * Rr_p - Eh_p_im * Ri_p
                    HRp_im = Eh_p_re * Ri_p + Eh_p_im * Rr_p
                    diff_p_re = HRp_re - Et_p_re; diff_p_im = HRp_im - Et_p_im
                    # d/d(h_re): 2*(rot_t-re - t_re)*r_re + 2*(rot_im - t_im)*r_im = 2*diff_re*r_re + 2*diff_im*r_im
                    g = -2.0  # negative gradient for positive sample
                    dEh_re_p = g * (diff_p_re * Rr_p + diff_p_im * Ri_p)
                    dEh_im_p = g * (-diff_p_re * Ri_p + diff_p_im * Rr_p)
                    dEt_re_p = g * (-diff_p_re)
                    dEt_im_p = g * (-diff_p_im)
                    dRr_p = g * (diff_p_re * Eh_p_re + diff_p_im * Eh_p_im)
                    dRi_p = g * (-diff_p_re * Eh_p_im + diff_p_im * Eh_p_re)

                    # Negative: push h∘r away from t'
                    Eh_n_re, Eh_n_im = self.E_re[h_n], self.E_im[h_n]
                    Et_n_re, Et_n_im = self.E_re[neg_t_act], self.E_im[neg_t_act]
                    Rr_n, Ri_n = self.R_re[r_n], self.R_im[r_n]
                    HRn_re = Eh_n_re * Rr_n - Eh_n_im * Ri_n
                    HRn_im = Eh_n_re * Ri_n + Eh_n_im * Rr_n
                    diff_n_re = HRn_re - Et_n_re; diff_n_im = HRn_im - Et_n_im
                    g = +2.0
                    dEh_re_n = g * (diff_n_re * Rr_n + diff_n_im * Ri_n)
                    dEh_im_n = g * (-diff_n_re * Ri_n + diff_n_im * Rr_n)
                    dEt_re_n = g * (-diff_n_re)
                    dEt_im_n = g * (-diff_n_im)
                    dRr_n = g * (diff_n_re * Eh_n_re + diff_n_im * Eh_n_im)
                    dRi_n = g * (-diff_n_re * Eh_n_im + diff_n_im * Eh_n_re)

                    np.add.at(dE_re, h_p, dEh_re_p); np.add.at(dE_im, h_p, dEh_im_p)
                    np.add.at(dE_re, t_p, dEt_re_p); np.add.at(dE_im, t_p, dEt_im_p)
                    np.add.at(dR_re, r_p, dRr_p); np.add.at(dR_im, r_p, dRi_p)
                    np.add.at(dE_re, h_n, dEh_re_n); np.add.at(dE_im, h_n, dEh_im_n)
                    np.add.at(dE_re, neg_t_act, dEt_re_n); np.add.at(dE_im, neg_t_act, dEt_im_n)
                    np.add.at(dR_re, r_n, dRr_n); np.add.at(dR_im, r_n, dRi_n)

                    Bf = B
                    self.E_re -= self._adagrad_update("Er", dE_re / Bf + self.reg_lambda * self.E_re)
                    self.E_im -= self._adagrad_update("Ei", dE_im / Bf + self.reg_lambda * self.E_im)
                    # Normalize relation phases to unit circle
                    norm = np.sqrt(self.R_re**2 + self.R_im**2) + 1e-12
                    self.R_re = self.R_re / norm; self.R_im = self.R_im / norm
        return {"loss": []}

    def evaluate_link_prediction(self, test_triples, train_filter):
        return _evaluate_generic(self, test_triples, train_filter)


class TuckERT:
    """Simplified TuckER with temporal core tensor.

    f(h,r,t,τ) = W ×_1 e_h ×_2 w_r ×_3 e_t where w_r includes time modulation.
    Uses a diagonal core tensor for efficiency (like ComplEx but trilinear).
    """

    def __init__(self, n_entities, n_relations, n_timestamps, dim=128,
                 reg_lambda=0.001, lr=0.1, negative_samples=10):
        self.n_e = n_entities; self.n_r = n_relations; self.n_t = n_timestamps
        self.dim = dim; self.reg_lambda = reg_lambda; self.lr = lr; self.neg = negative_samples
        rng = np.random.RandomState(42)
        scale = 0.5 / np.sqrt(dim)
        self.E = rng.randn(n_entities, dim) * scale
        self.R = rng.randn(n_relations, dim) * scale
        # Temporal modulation (additive)
        self.T = rng.randn(n_timestamps, dim) * scale * 0.2
        self._g2 = {"E": np.zeros_like(self.E), "R": np.zeros_like(self.R),
                    "T": np.zeros_like(self.T)}

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def score(self, h_idx, r_idx, t_idx, tau=None):
        w = self.R[r_idx] + self.T[tau]
        return float(np.sum(self.E[h_idx] * w * self.E[t_idx]))

    def _scores_batch(self, h_vec, r_vec, t_vec, tau_vec):
        w = self.R[r_vec] + self.T[tau_vec]
        return np.sum(self.E[h_vec] * w * self.E[t_vec], axis=1)

    def _adagrad_update(self, name, grad):
        self._g2[name] += grad ** 2
        return self.lr * grad / (np.sqrt(self._g2[name]) + 1e-8)

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
                neg_h = np.tile(h, self.neg); neg_r = np.tile(r, self.neg)
                neg_tau = np.tile(tau, self.neg)
                neg_scores = self._scores_batch(neg_h, neg_r, neg_t_all, neg_tau)
                pos_rep = np.repeat(pos, self.neg)
                hinge = np.maximum(0, margin + neg_scores - pos_rep)
                active = hinge > 0
                if active.any():
                    act_idx = np.where(active)[0]
                    pos_indices = np.repeat(np.arange(B), self.neg)[act_idx]
                    neg_t_act = neg_t_all[act_idx]
                    h_p, r_p, t_p = h[pos_indices], r[pos_indices], t[pos_indices]
                    tau_p = tau[pos_indices]
                    h_n, r_n = neg_h[act_idx], neg_r[act_idx]
                    tau_n = neg_tau[act_idx]

                    dE = np.zeros_like(self.E); dR = np.zeros_like(self.R)
                    dT = np.zeros_like(self.T)

                    # Positive gradients
                    wp = self.R[r_p] + self.T[tau_p]
                    dE_pos = -self.E[t_p] * wp  # d/d(e_h)
                    dEt_pos = -self.E[h_p] * wp  # d/d(e_t)
                    dR_pos = -self.E[h_p] * self.E[t_p]
                    np.add.at(dE, h_p, dE_pos); np.add.at(dE, t_p, dEt_pos)
                    np.add.at(dR, r_p, dR_pos); np.add.at(dT, tau_p, dR_pos)

                    # Negative gradients
                    wn = self.R[r_n] + self.T[tau_n]
                    dE_neg_h = self.E[neg_t_act] * wn
                    dE_neg_t = self.E[h_n] * wn
                    dR_neg = self.E[h_n] * self.E[neg_t_act]
                    np.add.at(dE, h_n, dE_neg_h); np.add.at(dE, neg_t_act, dE_neg_t)
                    np.add.at(dR, r_n, dR_neg); np.add.at(dT, tau_n, dR_neg)

                    Bf = B
                    self.E -= self._adagrad_update("E", dE / Bf + self.reg_lambda * self.E)
                    self.R -= self._adagrad_update("R", dR / Bf + self.reg_lambda * self.R)
                    self.T -= self._adagrad_update("T", dT / Bf + self.reg_lambda * self.T)
        return {"loss": []}

    def evaluate_link_prediction(self, test_triples, train_filter):
        return _evaluate_generic(self, test_triples, train_filter)
