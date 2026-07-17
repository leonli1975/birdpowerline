#!/usr/bin/env python3
"""EAI-TComplEx experiments v2 — with proper inductive training.

Key fix: for unseen entities, embeddings come from feature projection, 
and the projection matrices are trained via gradient flow.
"""

import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.kg_data import (
    build_temporal_kg, enrich_kg, build_id_mappings, build_tensor_data, N_TIMESTAMPS,
    POWER_LINES, LINE_ELEVATION, BIRD_SPECIES, MEASURES, FAULT_TYPES,
)


class InductiveTComplEx:
    """TComplEx variant where unseen entities get embedding from feature projection.

    Training: for seen entities, use learned embeddings.
              for unseen entities, use W_feat @ features.
    """

    def __init__(self, n_entities, n_relations, n_timestamps, entity_features,
                 unseen_mask, dim=128, lr=0.1, reg_lambda=0.001, negative_samples=10,
                 feat_align_weight=0.1):
        self.n_e = n_entities; self.n_r = n_relations; self.n_t = n_timestamps
        self.dim = dim; self.lr = lr; self.reg_lambda = reg_lambda; self.neg = negative_samples
        self.unseen = unseen_mask.astype(bool)
        self.feat_align_weight = feat_align_weight

        rng = np.random.RandomState(42)
        scale = 0.5 / np.sqrt(dim)
        self.E_re = rng.randn(n_entities, dim) * scale
        self.E_im = rng.randn(n_entities, dim) * scale
        self.R_re = rng.randn(n_relations, dim) * scale
        self.R_im = rng.randn(n_relations, dim) * scale
        self.T_re = rng.randn(n_timestamps, dim) * scale * 0.5
        self.T_im = rng.randn(n_timestamps, dim) * scale * 0.5

        # Feature projection
        self.attr = entity_features.astype(np.float32)
        feat_dim = entity_features.shape[1]
        self.W_re = rng.randn(feat_dim, dim) * 0.05
        self.W_im = rng.randn(feat_dim, dim) * 0.05
        self.b_re = np.zeros(dim); self.b_im = np.zeros(dim)

        # AdaGrad
        self._g2 = {}
        for name in ['Er','Ei','Rr','Ri','Tr','Ti','Wr','Wi','br','bi']:
            self._g2[name] = np.zeros(1)  # placeholder, dynamically sized

    def _get_g2(self, name, shape):
        key = name
        if key not in self._g2 or self._g2[key].shape != shape:
            self._g2[key] = np.zeros(shape)
        return self._g2[key]

    def _adagrad(self, name, grad, param):
        g2 = self._get_g2(name, grad.shape)
        g2 += grad ** 2
        return self.lr * grad / (np.sqrt(g2) + 1e-8)

    def get_embedding(self, eid):
        """Return (re, im) for entity. Unseen → feature projection."""
        if self.unseen[eid]:
            feat = self.attr[eid]
            re = feat @ self.W_re + self.b_re
            im = feat @ self.W_im + self.b_im
            return re, im
        else:
            return self.E_re[eid], self.E_im[eid]

    def get_embedding_batch(self, eids):
        """Batch version: returns (re, im) arrays [B, D]."""
        seen_mask = ~self.unseen[eids]
        unseen_mask = self.unseen[eids]

        re_out = self.E_re[eids].copy()
        im_out = self.E_im[eids].copy()

        if unseen_mask.any():
            unseen_ids = eids[unseen_mask]
            feat = self.attr[unseen_ids]  # [N_u, F]
            re_out[unseen_mask] = feat @ self.W_re + self.b_re
            im_out[unseen_mask] = feat @ self.W_im + self.b_im

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
        tail_prob = (tail_counts.astype(float) + 1) ** 0.75; tail_prob /= tail_prob.sum()

        for epoch in range(epochs):
            perm = np.random.permutation(n_train)
            for start in range(0, n_train, batch_size):
                idx = perm[start:start+ batch_size]; B = len(idx)
                h, r, t, tau = triples[idx].T
                pos = self._scores_batch(h, r, t, tau)
                neg_t_all = np.random.choice(self.n_e, size=B*self.neg, p=tail_prob)
                neg_scores = self._scores_batch(np.tile(h,self.neg), np.tile(r,self.neg),
                                                neg_t_all, np.tile(tau,self.neg))
                pos_rep = np.repeat(pos, self.neg)
                hinge = np.maximum(0, margin + neg_scores - pos_rep)
                active = hinge > 0
                if active.any():
                    act_idx = np.where(active)[0]
                    pos_indices = act_idx // self.neg
                    neg_t_act = neg_t_all[act_idx]
                    h_p, r_p, t_p = h[pos_indices], r[pos_indices], t[pos_indices]
                    tau_p = tau[pos_indices]
                    h_n, r_n = np.tile(h,self.neg)[act_idx], np.tile(r,self.neg)[act_idx]
                    tau_n = np.tile(tau,self.neg)[act_idx]

                    # Accumulate gradients
                    dE_re = np.zeros_like(self.E_re); dE_im = np.zeros_like(self.E_im)
                    dR_re = np.zeros_like(self.R_re); dR_im = np.zeros_like(self.R_im)
                    dT_re = np.zeros_like(self.T_re); dT_im = np.zeros_like(self.T_im)
                    dW_re = np.zeros_like(self.W_re); dW_im = np.zeros_like(self.W_im)
                    db_re = np.zeros_like(self.b_re); db_im = np.zeros_like(self.b_im)

                    self._acc_grad_inductive(h_p, r_p, t_p, tau_p, -1.0,
                        dE_re, dE_im, dR_re, dR_im, dT_re, dT_im, dW_re, dW_im, db_re, db_im)
                    self._acc_grad_inductive(h_n, r_n, neg_t_act, tau_n, +1.0,
                        dE_re, dE_im, dR_re, dR_im, dT_re, dT_im, dW_re, dW_im, db_re, db_im)

                    Bf = B
                    self.E_re -= self._adagrad("Er", dE_re/Bf + self.reg_lambda*self.E_re, self.E_re)
                    self.E_im -= self._adagrad("Ei", dE_im/Bf + self.reg_lambda*self.E_im, self.E_im)
                    self.R_re -= self._adagrad("Rr", dR_re/Bf + self.reg_lambda*self.R_re, self.R_re)
                    self.R_im -= self._adagrad("Ri", dR_im/Bf + self.reg_lambda*self.R_im, self.R_im)
                    self.T_re -= self._adagrad("Tr", dT_re/Bf + self.reg_lambda*self.T_re, self.T_re)
                    self.T_im -= self._adagrad("Ti", dT_im/Bf + self.reg_lambda*self.T_im, self.T_im)
                    self.W_re -= self._adagrad("Wr", dW_re/Bf + self.reg_lambda*self.W_re, self.W_re)
                    self.W_im -= self._adagrad("Wi", dW_im/Bf + self.reg_lambda*self.W_im, self.W_im)
                    self.b_re -= self._adagrad("br", db_re/Bf + self.reg_lambda*self.b_re, self.b_re)
                    self.b_im -= self._adagrad("bi", db_im/Bf + self.reg_lambda*self.b_im, self.b_im)

            # Feature alignment loss: force W_feat to map features to learned embeddings
            if self.feat_align_weight > 0:
                seen_entities = np.where(~self.unseen)[0]
                if len(seen_entities) > 0:
                    feat_seen = self.attr[seen_entities]  # [N_s, F]
                    pred_re = feat_seen @ self.W_re + self.b_re  # [N_s, D]
                    pred_im = feat_seen @ self.W_im + self.b_im
                    # MSE to learned embeddings
                    diff_re = pred_re - self.E_re[seen_entities]
                    diff_im = pred_im - self.E_im[seen_entities]
                    # Gradient: d/dW_re = 2*feat^T @ diff_re / N_s
                    N_s = len(seen_entities)
                    dWr_align = 2 * feat_seen.T @ diff_re / N_s
                    dWi_align = 2 * feat_seen.T @ diff_im / N_s
                    dbr_align = 2 * diff_re.mean(axis=0)
                    dbi_align = 2 * diff_im.mean(axis=0)
                    self.W_re -= self._adagrad("Wr", self.feat_align_weight*dWr_align + self.reg_lambda*self.W_re, self.W_re)
                    self.W_im -= self._adagrad("Wi", self.feat_align_weight*dWi_align + self.reg_lambda*self.W_im, self.W_im)
                    self.b_re -= self._adagrad("br", self.feat_align_weight*dbr_align + self.reg_lambda*self.b_re, self.b_re)
                    self.b_im -= self._adagrad("bi", self.feat_align_weight*dbi_align + self.reg_lambda*self.b_im, self.b_im)
        return {"loss": []}

    def _acc_grad_inductive(self, h_vec, r_vec, t_vec, tau_vec, g,
                             dE_re, dE_im, dR_re, dR_im, dT_re, dT_im,
                             dW_re, dW_im, db_re, db_im):
        """Gradient accumulation that routes gradients to W_re/W_im for unseen entities."""
        Eh_re, Eh_im = self.get_embedding_batch(h_vec)
        Et_re, Et_im = self.get_embedding_batch(t_vec)
        Rr = self.R_re[r_vec] + self.T_re[tau_vec]
        Ri = self.R_im[r_vec] + self.T_im[tau_vec]
        HR_re = Eh_re * Rr - Eh_im * Ri
        HR_im = Eh_re * Ri + Eh_im * Rr

        # Gradients w.r.t. entity embeddings
        dEh_re = g * (Rr * Et_re + Ri * Et_im)
        dEh_im = g * (-Ri * Et_re + Rr * Et_im)
        dEt_re = g * HR_re; dEt_im = g * HR_im
        dr_g = g * (Eh_re * Et_re + Eh_im * Et_im)
        di_g = g * (Eh_re * Et_im - Eh_im * Et_re)

        # Route: for unseen entities, gradients go through feature projection
        h_unseen = self.unseen[h_vec]
        t_unseen = self.unseen[t_vec]

        # Seen entities: accumulate to E_re/E_im
        if (~h_unseen).any():
            np.add.at(dE_re, h_vec[~h_unseen], dEh_re[~h_unseen])
            np.add.at(dE_im, h_vec[~h_unseen], dEh_im[~h_unseen])
        if (~t_unseen).any():
            np.add.at(dE_re, t_vec[~t_unseen], dEt_re[~t_unseen])
            np.add.at(dE_im, t_vec[~t_unseen], dEt_im[~t_unseen])

        # Unseen head entities: d/d(W_re) = feat^T @ dEh_re  [F, D]
        if h_unseen.any():
            feat_h = self.attr[h_vec[h_unseen]]  # [N_hu, F]
            dW_re += feat_h.T @ dEh_re[h_unseen]  # [F, N_hu] @ [N_hu, D] = [F, D]
            dW_im += feat_h.T @ dEh_im[h_unseen]
            db_re += dEh_re[h_unseen].sum(axis=0)
            db_im += dEh_im[h_unseen].sum(axis=0)

        # Unseen tail entities
        if t_unseen.any():
            feat_t = self.attr[t_vec[t_unseen]]
            dW_re += feat_t.T @ dEt_re[t_unseen]
            dW_im += feat_t.T @ dEt_im[t_unseen]
            db_re += dEt_re[t_unseen].sum(axis=0)
            db_im += dEt_im[t_unseen].sum(axis=0)

        # Relation/time gradients (shared)
        np.add.at(dR_re, r_vec, dr_g); np.add.at(dR_im, r_vec, di_g)
        np.add.at(dT_re, tau_vec, dr_g); np.add.at(dT_im, tau_vec, di_g)

    def score(self, h_idx, r_idx, t_idx, tau):
        re_h, im_h = self.get_embedding(h_idx)
        re_t, im_t = self.get_embedding(t_idx)
        rr = self.R_re[r_idx] + self.T_re[tau]
        ri = self.R_im[r_idx] + self.T_im[tau]
        hr_re = re_h * rr - im_h * ri
        hr_im = re_h * ri + im_h * rr
        return float(np.sum(hr_re * re_t + hr_im * im_t))

    def evaluate_link_prediction(self, test_triples, train_filter):
        triples = np.array(test_triples, dtype=np.int64)
        n_test = len(triples)
        if n_test == 0:
            return {"MRR":0,"Hits@1":0,"Hits@3":0,"Hits@10":0,"MeanRank":0}
        ranks = []
        for i in range(n_test):
            h,r,t,tau = triples[i]; hi,ri,ti,ti_i = int(h),int(r),int(t),int(tau)
            h_vec = np.full(self.n_e, hi, dtype=np.int64)
            t_vec = np.arange(self.n_e, dtype=np.int64)
            r_vec = np.full(self.n_e, ri, dtype=np.int64)
            tau_vec = np.full(self.n_e, ti_i, dtype=np.int64)
            scores = self._scores_batch(h_vec, r_vec, t_vec, tau_vec)
            if train_filter is not None:
                for cand in range(self.n_e):
                    if cand != ti and (hi,ri,cand,ti_i) in train_filter:
                        scores[cand] = -1e10
            rank = 1 + np.sum(scores > scores[ti])
            ranks.append(rank)
        ranks = np.array(ranks)
        return {"MRR":float(np.mean(1.0/ranks)),"Hits@1":float(np.mean(ranks<=1)),
                "Hits@3":float(np.mean(ranks<=3)),"Hits@10":float(np.mean(ranks<=10)),
                "MeanRank":float(np.mean(ranks))}


def build_features(e2i, i2e):
    """Simpler feature extraction."""
    n_e = len(e2i); F = 12
    feat = np.zeros((n_e, F))
    line_names = {name for name in i2e.values() if name.startswith('line_')}

    for eid, name in i2e.items():
        if name in line_names:
            info = POWER_LINES.get(name, {})
            v = info.get('voltage', 10)
            # Voltage one-hot
            if v == 110: feat[eid,0]=1
            elif v == 35: feat[eid,1]=1
            else: feat[eid,2]=1
            # Poles normalized
            feat[eid,3] = min(info.get('poles',0)/300.0, 1)
            # Length normalized
            feat[eid,4] = min(info.get('length_km',0)/100.0, 1)
            # Nest rate
            feat[eid,5] = info.get('nest_rate', 0)
            # Near wetland
            if name in ('line_aji','line_ruozhen','line_ruotang'):
                feat[eid,6]=1
            # Near alpine meadow
            if name in ('line_ruozhen','line_heize','line_ruoa'):
                feat[eid,7]=1
            # Elevation zone
            elev = LINE_ELEVATION.get(name, 'elev_mid')
            if 'mid' in elev: feat[eid,8]=1
            elif 'high' in elev: feat[eid,9]=1
            # Has artificial nests
            if info.get('artificial_nests', 0) > 0:
                feat[eid,10]=1
            if info.get('activity', 0) > 0:
                feat[eid,11]=info['activity']/5.0
        elif name in BIRD_SPECIES:
            info = BIRD_SPECIES[name]
            if info.get('class')=='I': feat[eid,0]=1
            elif info.get('class')=='II': feat[eid,1]=1
            if info.get('body_size')=='large': feat[eid,2]=1
            elif info.get('body_size')=='medium': feat[eid,3]=1
            if info.get('habitat')=='wetland': feat[eid,4]=1
            elif info.get('habitat')=='grassland': feat[eid,5]=1
            elif info.get('habitat')=='mountain': feat[eid,6]=1
            if info.get('residency')=='resident': feat[eid,7]=1
            elif info.get('residency')=='summer_visitor': feat[eid,8]=1
            feat[eid,9] = 0.5  # baseline
        elif name in MEASURES:
            info = MEASURES[name]
            if info.get('type')=='physical_barrier': feat[eid,0]=1
            elif info.get('type')=='deterrent': feat[eid,1]=1
            elif info.get('type')=='attractant': feat[eid,2]=1
            feat[eid,3]=0.5
        # Normalize
        norm = np.linalg.norm(feat[eid])
        if norm > 0:
            feat[eid] /= norm

    return feat


def main():
    kg = build_temporal_kg(); kg = enrich_kg(kg)
    e2i, r2i, i2e, i2r = build_id_mappings(kg)
    triples, _, _ = build_tensor_data(kg, e2i, r2i)
    arr = np.array(triples, dtype=np.int64)
    n_e, n_r, n_t = len(e2i), len(r2i), N_TIMESTAMPS
    features = build_features(e2i, i2e)
    print(f"KG: {n_e} entities, {n_r} relations, features dim={features.shape[1]}")
    print()

    # LOO setup: Ruozhen line as unseen
    unseen_name = 'line_ruozhen'
    unseen_id = e2i[unseen_name]
    unseen_mask = np.zeros(n_e, dtype=bool)
    unseen_mask[unseen_id] = True

    # Train without Ruozhen
    train = []; test = []
    for h,r,t,tau in arr:
        hi,ri,ti,ti_i = int(h),int(r),int(t),int(tau)
        if hi == unseen_id or ti == unseen_id:
            test.append((hi,ri,ti,ti_i))
        elif ti_i <= 4:
            train.append((hi,ri,ti,ti_i))
    tf = set(train)
    print(f"Leave-{unseen_name}-Out: train={len(train)} test={len(test)}")
    print()

    # Baseline: standard TComplEx (Ruozhen = random init, never trained)
    print("="*60)
    print("Experiment: Leave-One-Line-Out OOD")
    print("="*60)
    from src.tkg_embedding import TComplEx
    t0 = time.time()
    m1 = TComplEx(n_e, n_r, n_t, dim=128, lr=0.1, negative_samples=10, reg_lambda=0.001)
    m1.fit(train, epochs=100, batch_size=512, margin=1.0, verbose=False)
    r1 = m1.evaluate_link_prediction(test, tf)
    print(f"  Baseline TComplEx:       MRR={r1['MRR']:.4f} H@10={r1['Hits@10']:.4f} t={time.time()-t0:.1f}s")

    # Inductive TComplEx (unseen → feature projection)
    t0 = time.time()
    m2 = InductiveTComplEx(n_e, n_r, n_t, features, unseen_mask,
                           dim=128, lr=0.1, negative_samples=10, reg_lambda=0.001)
    m2.fit(train, epochs=100, batch_size=512, margin=1.0, verbose=False)
    r2 = m2.evaluate_link_prediction(test, tf)
    print(f"  Inductive TComplEx:      MRR={r2['MRR']:.4f} H@10={r2['Hits@10']:.4f} t={time.time()-t0:.1f}s")

    print(f"\n  OOD Delta: {r2['MRR']-r1['MRR']:+.4f}")

    # Also test in-distribution with inductive
    print()
    print("="*60)
    print("Experiment: In-Distribution (all entities seen)")
    print("="*60)
    train_all, test_all = [], []
    for h,r,t,tau in arr:
        hi,ri,ti,ti_i = int(h),int(r),int(t),int(tau)
        if ti_i >= 6: test_all.append((hi,ri,ti,ti_i))
        elif ti_i <= 4: train_all.append((hi,ri,ti,ti_i))
    tf_all = set(train_all)
    unseen_mask_all = np.zeros(n_e, dtype=bool)  # all seen

    t0 = time.time()
    m3 = InductiveTComplEx(n_e, n_r, n_t, features, unseen_mask_all,
                           dim=128, lr=0.1, negative_samples=10, reg_lambda=0.001)
    m3.fit(train_all, epochs=100, batch_size=512, margin=1.0, verbose=False)
    r3 = m3.evaluate_link_prediction(test_all, tf_all)
    print(f"  Inductive (all seen):    MRR={r3['MRR']:.4f} H@10={r3['Hits@10']:.4f} t={time.time()-t0:.1f}s")

    t0 = time.time()
    m4 = TComplEx(n_e, n_r, n_t, dim=128, lr=0.1, negative_samples=10, reg_lambda=0.001)
    m4.fit(train_all, epochs=100, batch_size=512, margin=1.0, verbose=False)
    r4 = m4.evaluate_link_prediction(test_all, tf_all)
    print(f"  Baseline TComplEx:       MRR={r4['MRR']:.4f} H@10={r4['Hits@10']:.4f} t={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
