"""Temporal Knowledge Graph Embedding (TComplEx) — NumPy implementation.

TComplEx scoring: f(h,r,t,τ) = Re(⟨e_h ⊙ (w_r + v_τ), conj(e_t)⟩)

Uses margin-based ranking loss with AdaGrad optimization.
"""

import numpy as np


class TComplEx:
    def __init__(self, n_entities, n_relations, n_timestamps, dim=128,
                 reg_lambda=0.01, lr=0.1, negative_samples=10):
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
        self.T_re = rng.randn(n_timestamps, dim) * scale * 0.5
        self.T_im = rng.randn(n_timestamps, dim) * scale * 0.5

        # AdaGrad accumulators
        self._g2 = {k: np.zeros_like(v) for k, v in
                    [("Er", self.E_re), ("Ei", self.E_im),
                     ("Rr", self.R_re), ("Ri", self.R_im),
                     ("Tr", self.T_re), ("Ti", self.T_im)]}

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def score(self, h_idx, r_idx, t_idx, tau):
        hr_re = self.E_re[h_idx] * (self.R_re[r_idx] + self.T_re[tau]) - \
                self.E_im[h_idx] * (self.R_im[r_idx] + self.T_im[tau])
        hr_im = self.E_re[h_idx] * (self.R_im[r_idx] + self.T_im[tau]) + \
                self.E_im[h_idx] * (self.R_re[r_idx] + self.T_re[tau])
        return float(np.sum(hr_re * self.E_re[t_idx] + hr_im * self.E_im[t_idx]))

    def _scores_batch(self, h_vec, r_vec, t_vec, tau_vec):
        """Vectorized scoring. All inputs are [B] integer arrays."""
        Eh_re = self.E_re[h_vec]       # [B, D]
        Eh_im = self.E_im[h_vec]
        Et_re = self.E_re[t_vec]
        Et_im = self.E_im[t_vec]
        Rr = self.R_re[r_vec] + self.T_re[tau_vec]  # [B, D]
        Ri = self.R_im[r_vec] + self.T_im[tau_vec]

        HR_re = Eh_re * Rr - Eh_im * Ri
        HR_im = Eh_re * Ri + Eh_im * Rr
        return np.sum(HR_re * Et_re + HR_im * Et_im, axis=1)  # [B]

    def _adagrad_update(self, name, grad):
        """Apply AdaGrad update: param -= lr * grad / sqrt(accum + 1e-8)."""
        accum = self._g2[name]
        accum += grad ** 2
        step = self.lr * grad / (np.sqrt(accum) + 1e-8)
        return step

    def fit(self, triples, epochs=500, batch_size=512, verbose=True,
            valid_triples=None, margin=1.0):
        """Training with margin-based ranking loss + AdaGrad."""
        triples = np.array(triples, dtype=np.int64)
        n_train = len(triples)

        tail_counts = np.bincount(triples[:, 2], minlength=self.n_e)
        tail_prob = (tail_counts.astype(float) + 1) ** 0.75
        tail_prob /= tail_prob.sum()

        train_filter = None
        if valid_triples is not None:
            train_filter = {(int(h), int(r), int(t), int(tau))
                           for h, r, t, tau in triples}

        best_mrr = 0.0
        history = {"loss": [], "valid_mrr": []}

        for epoch in range(epochs):
            perm = np.random.permutation(n_train)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                B = len(idx)
                h, r, t, tau = triples[idx].T  # [B]

                # Positive scores
                pos = self._scores_batch(h, r, t, tau)  # [B]

                # Negative tails: B*neg corrupted tails
                neg_t_all = np.random.choice(self.n_e, size=B * self.neg, p=tail_prob)
                neg_h = np.tile(h, self.neg)
                neg_r = np.tile(r, self.neg)
                neg_tau = np.tile(tau, self.neg)
                neg_scores = self._scores_batch(neg_h, neg_r, neg_t_all, neg_tau)  # [B*neg]

                # Margin ranking loss: max(0, margin + neg_score - pos_score_repeated)
                pos_rep = np.repeat(pos, self.neg)
                hinge = np.maximum(0, margin + neg_scores - pos_rep)
                loss = np.mean(hinge)

                total_loss += loss
                n_batches += 1

                # Gradient: only for active negatives (hinge > 0)
                active_mask = hinge > 0
                if not active_mask.any():
                    continue

                # Active indices (relative to B*neg array)
                act_idx = np.where(active_mask)[0]
                # Corresponding positive triple indices
                pos_indices = act_idx // self.neg  # [n_active]

                # Compute gradients for positive triples and negative triples
                # dL/d(pos_score) = -1, dL/d(neg_score) = +1 (per active sample, normalized by B)
                n_active = len(act_idx)
                scale_factor = 1.0 / B

                # --- Positive gradient accumulation ---
                pos_h = h[pos_indices]
                pos_r = r[pos_indices]
                pos_t = t[pos_indices]
                pos_tau = tau[pos_indices]
                pos_grad = -1.0  # gradient of loss w.r.t positive score

                # --- Negative gradient accumulation ---
                neg_h_act = neg_h[act_idx]
                neg_r_act = neg_r[act_idx]
                neg_t_act = neg_t_all[act_idx]
                neg_tau_act = neg_tau[act_idx]
                neg_grad = +1.0  # gradient of loss w.r.t negative score

                self._apply_gradients(
                    pos_h, pos_r, pos_t, pos_tau, pos_grad,
                    neg_h_act, neg_r_act, neg_t_act, neg_tau_act, neg_grad,
                    scale_factor
                )

            avg_loss = total_loss / max(n_batches, 1)
            history["loss"].append(float(avg_loss))

            valid_mrr = 0.0
            if valid_triples is not None and (epoch + 1) % 20 == 0:
                metrics = self.evaluate_link_prediction(valid_triples, train_filter)
                valid_mrr = metrics["MRR"]
                if valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                history["valid_mrr"].append(float(valid_mrr))

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  "
                      f"valid_MRR={valid_mrr:.4f}"
                      + (" *" if valid_mrr >= best_mrr and valid_mrr > 0 else ""))

        return history

    def _apply_gradients(self, pos_h, pos_r, pos_t, pos_tau, pos_g,
                          neg_h, neg_r, neg_t, neg_tau, neg_g, scale):
        """Vectorized gradient computation and AdaGrad update for margin loss.

        pos: triple with gradient pos_g (e.g., -1 to push score up)
        neg: triple with gradient neg_g (e.g., +1 to push score down)
        """
        # Accumulators
        dE_re = np.zeros_like(self.E_re)
        dE_im = np.zeros_like(self.E_im)
        dR_re = np.zeros_like(self.R_re)
        dR_im = np.zeros_like(self.R_im)
        dT_re = np.zeros_like(self.T_re)
        dT_im = np.zeros_like(self.T_im)

        # -- Positive --
        if len(pos_h) > 0:
            self._acc_grad(pos_h, pos_r, pos_t, pos_tau, pos_g,
                           dE_re, dE_im, dR_re, dR_im, dT_re, dT_im)

        # -- Negative --
        if len(neg_h) > 0:
            self._acc_grad(neg_h, neg_r, neg_t, neg_tau, neg_g,
                           dE_re, dE_im, dR_re, dR_im, dT_re, dT_im)

        # Apply AdaGrad updates with regularization
        self.E_re -= self._adagrad_update("Er",
                                          scale * dE_re + self.reg_lambda * self.E_re)
        self.E_im -= self._adagrad_update("Ei",
                                          scale * dE_im + self.reg_lambda * self.E_im)
        self.R_re -= self._adagrad_update("Rr",
                                          scale * dR_re + self.reg_lambda * self.R_re)
        self.R_im -= self._adagrad_update("Ri",
                                          scale * dR_im + self.reg_lambda * self.R_im)
        self.T_re -= self._adagrad_update("Tr",
                                          scale * dT_re + self.reg_lambda * self.T_re)
        self.T_im -= self._adagrad_update("Ti",
                                          scale * dT_im + self.reg_lambda * self.T_im)

    def _acc_grad(self, h_vec, r_vec, t_vec, tau_vec, grad_sign,
                   dE_re, dE_im, dR_re, dR_im, dT_re, dT_im):
        """Accumulate gradients for a set of triples with a given sign."""
        # Look up
        Eh_re = self.E_re[h_vec]       # [N, D]
        Eh_im = self.E_im[h_vec]
        Et_re = self.E_re[t_vec]
        Et_im = self.E_im[t_vec]
        Rr = self.R_re[r_vec] + self.T_re[tau_vec]  # [N, D]
        Ri = self.R_im[r_vec] + self.T_im[tau_vec]

        # Forward caches
        HR_re = Eh_re * Rr - Eh_im * Ri  # [N, D]
        HR_im = Eh_re * Ri + Eh_im * Rr

        g = grad_sign  # scalar

        # d/d(Eh_re): g * (Rr * Et_re + Ri * Et_im)
        dEh_re = g * (Rr * Et_re + Ri * Et_im)
        # d/d(Eh_im): g * (-Ri * Et_re + Rr * Et_im)
        dEh_im = g * (-Ri * Et_re + Rr * Et_im)
        # d/d(Et_re): g * HR_re
        dEt_re = g * HR_re
        # d/d(Et_im): g * HR_im
        dEt_im = g * HR_im
        # d/d(Rr) = d/d(Tr): g * (Eh_re * Et_re + Eh_im * Et_im)
        dr = g * (Eh_re * Et_re + Eh_im * Et_im)
        # d/d(Ri) = d/d(Ti): g * (Eh_re * Et_im - Eh_im * Et_re)
        di = g * (Eh_re * Et_im - Eh_im * Et_re)

        np.add.at(dE_re, h_vec, dEh_re)
        np.add.at(dE_im, h_vec, dEh_im)
        np.add.at(dE_re, t_vec, dEt_re)
        np.add.at(dE_im, t_vec, dEt_im)
        np.add.at(dR_re, r_vec, dr)
        np.add.at(dR_im, r_vec, di)
        np.add.at(dT_re, tau_vec, dr)
        np.add.at(dT_im, tau_vec, di)

    def evaluate_link_prediction(self, test_triples, train_filter):
        """Evaluate MRR, Hits@1,3,10 (filtered setting)."""
        triples = np.array(test_triples, dtype=np.int64)
        n_test = len(triples)
        if n_test == 0:
            return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0,
                    "Hits@10": 0.0, "MeanRank": 0.0}

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

            target_score = scores[ti]
            rank = 1 + np.sum(scores > target_score)
            ranks.append(rank)

        ranks = np.array(ranks)
        return {
            "MRR": float(np.mean(1.0 / ranks)),
            "Hits@1": float(np.mean(ranks <= 1)),
            "Hits@3": float(np.mean(ranks <= 3)),
            "Hits@10": float(np.mean(ranks <= 10)),
            "MeanRank": float(np.mean(ranks)),
        }
