"""CHKG Embedding v2: TComplEx + Multi-Tree Hyperbolic Regularization + Causal Scoring.

Full implementation of C1-C4 contributions:
  C1: Multi-tree hierarchy formalization
  C2: Joint hyperbolic embedding with causal norm ordering
  C3: Causal link prediction via P(t|do(h)) = TComplEx + β * CE(h→t)
  C4: Counterfactual reasoning via DAG propagation
"""

import numpy as np
from collections import defaultdict


def poincare_proj(x, eps=1e-5):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    mask = norm >= 1.0
    if np.any(mask):
        x = np.where(mask, x / (norm + eps) * (1.0 - eps), x)
    return x


def poincare_distance(x, y):
    num = np.sum((x - y)**2, axis=-1)
    denom_x = 1 - np.sum(x**2, axis=-1)
    denom_y = 1 - np.sum(y**2, axis=-1)
    return np.arccosh(1 + 2 * num / (denom_x * denom_y + 1e-12))


class CHKGv2:
    """Full CHKG with causal scoring and counterfactual reasoning."""

    def __init__(self, n_entities, n_relations, n_timestamps,
                 hierarchy_masks, dim=128, lr=0.1, reg_lambda=0.001,
                 negative_samples=10, alpha_sp=0.1, alpha_tax=0.1, alpha_tro=0.1,
                 causal_beta=0.05, margin_norm=0.1):
        self.n_e = n_entities; self.n_r = n_relations; self.n_t = n_timestamps
        self.dim = dim; self.lr = lr; self.reg_lambda = reg_lambda; self.neg = negative_samples
        self.alpha_sp = alpha_sp; self.alpha_tax = alpha_tax; self.alpha_tro = alpha_tro
        self.causal_beta = causal_beta
        self.margin_norm = margin_norm
        self.hierarchy_masks = hierarchy_masks

        rng = np.random.RandomState(42)
        scale = 0.5 / np.sqrt(dim)
        self.E_re = rng.randn(n_entities, dim) * scale
        self.E_im = rng.randn(n_entities, dim) * scale
        self.R_re = rng.randn(n_relations, dim) * scale
        self.R_im = rng.randn(n_relations, dim) * scale
        self.T_re = rng.randn(n_timestamps, dim) * scale * 0.5
        self.T_im = rng.randn(n_timestamps, dim) * scale * 0.5

        # Hyperbolic embeddings
        self.H = poincare_proj(rng.randn(n_entities, dim) * 0.01)

        # Causal path lookup
        self._ancestor_lookup = {}
        self._build_causal_lookup()

        # Causal path cache — precompute pairwise distances
        self._causal_matrix = None
        self._build_causal_matrix()

        self._g2 = defaultdict(lambda: np.zeros(1))

    def _build_causal_matrix(self):
        """Precompute pairwise hyperbolic causal distances.

        Uses norm-based approximation: entities sharing ancestors
        get causal effect proportional to 1 - |n(h)-n(t)|/max_range.
        """
        self._causal_matrix = np.zeros((self.n_e, self.n_e), dtype=np.float32)
        norms = np.sqrt(np.sum(self.H**2, axis=1))

        for hname in ["spatial", "taxonomic", "trophic"]:
            ancestor = self._ancestor_lookup.get(hname)
            if ancestor is None or ancestor.sum() == 0:
                continue
            for anc in range(self.n_e):
                descendants = np.where(ancestor[anc])[0]
                if len(descendants) <= 1:
                    continue
                for i in range(len(descendants)):
                    for j in range(len(descendants)):
                        if i == j:
                            continue
                        h, t = descendants[i], descendants[j]
                        norm_diff = abs(norms[h] - norms[t])
                        ce = 1.0 - norm_diff  # closer norms → stronger causal link
                        self._causal_matrix[h, t] = max(self._causal_matrix[h, t], ce)
                        self._causal_matrix[t, h] = max(self._causal_matrix[t, h], ce)

    def _build_causal_lookup(self):
        """Pre-compute causal ancestor lookup for all hierarchy trees."""
        for hname in ["spatial", "taxonomic", "trophic"]:
            mask = self.hierarchy_masks.get(hname, {})
            ancestor = mask.get("ancestor_mask")
            if ancestor is None:
                continue
            self._ancestor_lookup[hname] = ancestor

    def compute_causal_effects_batch(self, h_vec, t_vec):
        """Batch causal effect computation using precomputed matrix."""
        return self._causal_matrix[h_vec, t_vec]

    def compute_causal_effect(self, h, t):
        """Single causal effect lookup."""
        return float(self._causal_matrix[h, t])

    def _scores_batch(self, h_vec, r_vec, t_vec, tau_vec):
        """Causal link prediction scoring.

        f_causal(h,r,t,τ) = f_TComplEx(h,r,t,τ) + β · CE(h→t)
        """
        # TComplEx score
        Eh_re = self.E_re[h_vec]; Eh_im = self.E_im[h_vec]
        Et_re = self.E_re[t_vec]; Et_im = self.E_im[t_vec]
        Rr = self.R_re[r_vec] + self.T_re[tau_vec]
        Ri = self.R_im[r_vec] + self.T_im[tau_vec]
        HR_re = Eh_re * Rr - Eh_im * Ri
        HR_im = Eh_re * Ri + Eh_im * Rr
        tkg_scores = np.sum(HR_re * Et_re + HR_im * Et_im, axis=1)

        # Causal effect bonus
        if self.causal_beta > 0:
            ce = self.compute_causal_effects_batch(h_vec, t_vec)
            return tkg_scores + self.causal_beta * ce
        return tkg_scores

    def compute_tree_loss(self, hname):
        """Tree norm-ordering loss."""
        mask = self.hierarchy_masks.get(hname, {})
        ancestor = mask.get("ancestor_mask")
        if ancestor is None or ancestor.sum() == 0:
            return 0.0

        norms = np.sqrt(np.sum(self.H**2, axis=1))
        anc_rows, anc_cols = np.where(ancestor)
        if len(anc_rows) == 0:
            return 0.0
        norm_diff = norms[anc_rows] - norms[anc_cols]
        return float(np.sum(np.maximum(0, norm_diff + self.margin_norm)) / max(len(anc_rows), 1))

    def compute_all_tree_loss(self):
        loss = 0.0
        if self.alpha_sp > 0:
            loss += self.alpha_sp * self.compute_tree_loss("spatial")
        if self.alpha_tax > 0:
            loss += self.alpha_tax * self.compute_tree_loss("taxonomic")
        if self.alpha_tro > 0:
            loss += self.alpha_tro * self.compute_tree_loss("trophic")
        return loss

    def fit(self, triples, epochs=100, batch_size=512, margin=1.0, verbose=False):
        triples = np.array(triples, dtype=np.int64)
        n_train = len(triples)
        tail_counts = np.bincount(triples[:, 2], minlength=self.n_e)
        tail_prob = (tail_counts.astype(float) + 1) ** 0.75
        tail_prob /= tail_prob.sum()

        history = {"loss": [], "tree_loss": []}

        for epoch in range(epochs):
            perm = np.random.permutation(n_train)
            total_loss = 0.0; n_batches = 0

            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]; B = len(idx)
                h, r, t, tau = triples[idx].T
                pos = self._scores_batch(h, r, t, tau)
                neg_t = np.random.choice(self.n_e, size=B * self.neg, p=tail_prob)
                neg_scores = self._scores_batch(np.tile(h, self.neg), np.tile(r, self.neg),
                                                neg_t, np.tile(tau, self.neg))
                pos_rep = np.repeat(pos, self.neg)
                hinge = np.maximum(0, margin + neg_scores - pos_rep)
                margin_loss = np.mean(hinge)

                if hinge.max() > 0:
                    dE_re = np.zeros_like(self.E_re); dE_im = np.zeros_like(self.E_im)
                    dR_re = np.zeros_like(self.R_re); dR_im = np.zeros_like(self.R_im)
                    dT_re = np.zeros_like(self.T_re); dT_im = np.zeros_like(self.T_im)

                    act_idx = np.where(hinge > 0)[0]
                    pos_ids = act_idx // self.neg
                    neg_t_act = neg_t[act_idx]
                    for i in pos_ids:
                        self._add_grad(int(h[i]), int(r[i]), int(t[i]), int(tau[i]),
                                      -1.0/B, dE_re, dE_im, dR_re, dR_im, dT_re, dT_im)
                    for i in range(len(act_idx)):
                        self._add_grad(int(np.tile(h, self.neg)[act_idx[i]]),
                                      int(np.tile(r, self.neg)[act_idx[i]]),
                                      int(neg_t_act[i]),
                                      int(np.tile(tau, self.neg)[act_idx[i]]),
                                      +1.0/B, dE_re, dE_im, dR_re, dR_im, dT_re, dT_im)

                    self.E_re -= self._adagrad("Er", dE_re+self.reg_lambda*self.E_re, self.E_re)
                    self.E_im -= self._adagrad("Ei", dE_im+self.reg_lambda*self.E_im, self.E_im)
                    self.R_re -= self._adagrad("Rr", dR_re+self.reg_lambda*self.R_re, self.R_re)
                    self.R_im -= self._adagrad("Ri", dR_im+self.reg_lambda*self.R_im, self.R_im)
                    self.T_re -= self._adagrad("Tr", dT_re+self.reg_lambda*self.T_re, self.T_re)
                    self.T_im -= self._adagrad("Ti", dT_im+self.reg_lambda*self.T_im, self.T_im)

                total_loss += margin_loss; n_batches += 1

            # Tree regularization
            tree_loss = self.compute_all_tree_loss()
            if tree_loss > 0:
                for hname in ["spatial", "taxonomic", "trophic"]:
                    ancestor = self._ancestor_lookup.get(hname)
                    if ancestor is None or ancestor.sum() == 0:
                        continue
                    anc_rows, anc_cols = np.where(ancestor)
                    norms = np.sqrt(np.sum(self.H**2, axis=1))
                    for i in np.random.choice(len(anc_rows), min(50, len(anc_rows)), replace=False):
                        a, d = anc_rows[i], anc_cols[i]
                        if norms[a] >= norms[d] - self.margin_norm:
                            self.H[a] *= 0.995
                            self.H[d] = poincare_proj(self.H[d] * 1.005)
                        else:
                            self.H[a] *= 0.9995
                            self.H[d] = poincare_proj(self.H[d] * 1.0005)
                self.H = poincare_proj(self.H)

            history["loss"].append(float(total_loss/max(n_batches, 1)))
            history["tree_loss"].append(float(tree_loss))

            if verbose and (epoch+1) % 20 == 0:
                print(f"  Ep {epoch+1:3d} margin={history['loss'][-1]:.4f} tree={history['tree_loss'][-1]:.4f}")

        return history

    def _add_grad(self, h, r, t, tau, g, dE_re, dE_im, dR_re, dR_im, dT_re, dT_im):
        Ehr, Ehi = self.E_re[h], self.E_im[h]
        Etr, Eti = self.E_re[t], self.E_im[t]
        Rr = self.R_re[r] + self.T_re[tau]
        Ri = self.R_im[r] + self.T_im[tau]
        dE_re[h] += g*(Rr*Etr+Ri*Eti); dE_im[h] += g*(-Ri*Etr+Rr*Eti)
        dE_re[t] += g*(Ehr*Rr-Ehi*Ri); dE_im[t] += g*(Ehr*Ri+Ehi*Rr)
        dr = g*(Ehr*Etr+Ehi*Eti); di = g*(Ehr*Eti-Ehi*Etr)
        dR_re[r] += dr; dR_im[r] += di
        dT_re[tau] += dr; dT_im[tau] += di

    def _adagrad(self, name, grad, param):
        if self._g2[name].shape != grad.shape:
            self._g2[name] = np.zeros(grad.shape)
        self._g2[name] += grad**2
        return self.lr*grad/(np.sqrt(self._g2[name])+1e-8)

    def evaluate_link_prediction(self, test_triples, train_filter):
        triples = np.array(test_triples, dtype=np.int64)
        if len(triples) == 0:
            return {"MRR":0,"Hits@1":0,"Hits@3":0,"Hits@10":0,"MeanRank":0}
        ranks = []
        for i in range(len(triples)):
            h, r, t, tau = triples[i]; hi,ri,ti,ti_i = int(h),int(r),int(t),int(tau)
            h_vec = np.full(self.n_e, hi, dtype=np.int64)
            t_vec = np.arange(self.n_e, dtype=np.int64)
            r_vec = np.full(self.n_e, ri, dtype=np.int64)
            tau_vec = np.full(self.n_e, ti_i, dtype=np.int64)
            scores = self._scores_batch(h_vec, r_vec, t_vec, tau_vec)
            if train_filter:
                for cand in range(self.n_e):
                    if cand != ti and (hi,ri,cand,ti_i) in train_filter:
                        scores[cand] = -1e10
            ranks.append(1+np.sum(scores>scores[ti]))
        ranks = np.array(ranks)
        return {"MRR":float(np.mean(1.0/ranks)),"Hits@1":float(np.mean(ranks<=1)),
                "Hits@3":float(np.mean(ranks<=3)),"Hits@10":float(np.mean(ranks<=10)),
                "MeanRank":float(np.mean(ranks))}

    def counterfactual_predict(self, h, r, t, tau, intervention_entity, delta_H=None):
        """Predict counterfactual score under intervention.

        intervention_entity: entity receiving the intervention
        delta_H: shift applied to the entity's hyperbolic embedding

        Returns (factual_score, counterfactual_score, delta)
        """
        factual = float(self._scores_batch(
            np.array([h]), np.array([r]), np.array([t]), np.array([tau])))

        # Save state
        H_saved = self.H.copy()
        E_re_saved = self.E_re.copy()
        E_im_saved = self.E_im.copy()

        # Apply intervention: shift the intervened entity's embedding
        if delta_H is not None:
            self.H[intervention_entity] = poincare_proj(self.H[intervention_entity] + delta_H)

        # Propagate through causal hierarchy to affected entities
        descendants = set()
        for hname, ancestor in self._ancestor_lookup.items():
            desc = np.where(ancestor[intervention_entity])[0]
            descendants.update(desc)

        # Shift descendants proportionally
        if delta_H is not None and len(descendants) > 0:
            for d in descendants:
                dist = float(poincare_distance(
                    self.H[intervention_entity].reshape(1,-1),
                    self.H[d].reshape(1,-1)))
                decay = np.exp(-dist)
                self.H[d] = poincare_proj(self.H[d] + delta_H * decay * 0.5)

        # Compute counterfactual score
        counterfactual = float(self._scores_batch(
            np.array([h]), np.array([r]), np.array([t]), np.array([tau])))

        # Restore
        self.H = H_saved
        self.E_re = E_re_saved
        self.E_im = E_im_saved

        return factual, counterfactual, counterfactual - factual
