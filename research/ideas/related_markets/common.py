"""Shared helpers: Dixon-Coles goal model lookup table and side-market probability functions.

A side market is described by (smt, line, side) where smt in {totals, both_teams_to_score, spreads,
soccer_exact_score, soccer_halftime_result}. Probabilities are computed on a (lambda_home, lambda_away) grid and the
grid point matching a 1X2 (pH, pA) is found by nearest neighbour in logit space.
"""
import numpy as np
from scipy.stats import poisson
from scipy.spatial import cKDTree

G = 11  # goals 0..10


def score_matrix(lh, la, rho):
    """lh, la: arrays (n,) -> (n, G, G) DC probabilities."""
    i = np.arange(G)
    ph = poisson.pmf(i[None, :], lh[:, None]); pa = poisson.pmf(i[None, :], la[:, None])
    M = ph[:, :, None] * pa[:, None, :]
    M[:, 0, 0] *= 1 - lh * la * rho
    M[:, 0, 1] *= 1 + lh * rho
    M[:, 1, 0] *= 1 + la * rho
    M[:, 1, 1] *= 1 - rho
    return M / M.sum((1, 2), keepdims=True)


class Table:
    def __init__(self, rho=-0.08, lo=0.05, hi=4.5, step=0.0125):
        v = np.arange(lo, hi, step)
        LH, LA = np.meshgrid(v, v, indexing="ij")
        self.lh, self.la = LH.ravel(), LA.ravel()
        self.rho = rho
        out = []
        for s in range(0, len(self.lh), 20000):
            out.append(score_matrix(self.lh[s:s + 20000], self.la[s:s + 20000], rho).astype(np.float32))
        self.M = np.concatenate(out)
        i = np.arange(G)
        self.D = (i[:, None] - i[None, :])          # home - away
        self.T = (i[:, None] + i[None, :])
        self.pH = self.M[:, self.D > 0].sum(1); self.pA = self.M[:, self.D < 0].sum(1)
        lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        self._lg = lg
        self.tree = cKDTree(np.c_[lg(self.pH), lg(self.pA)])

    def idx(self, pH, pA):
        return self.tree.query(np.c_[self._lg(np.asarray(pH, float)), self._lg(np.asarray(pA, float))])[1]

    def prob(self, ix, smt, line, outA, home_name=None):
        """P(token A of the market resolves Yes) for grid rows ix. outA = first outcome label."""
        M = self.M[ix]
        if smt == "totals":
            return M[:, self.T > line].sum(1)                       # token A = Over
        if smt == "both_teams_to_score":
            return M[:, 1:, 1:].sum((1, 2))                        # token A = Yes
        raise ValueError(smt)

    def spread(self, ix, line, home_is_A):
        """Spread: token A team (-|line|) covers. home_is_A: A team is the home team."""
        M = self.M[ix]; L = abs(line)
        return M[:, self.D > L].sum(1) if home_is_A else M[:, -self.D > L].sum(1)

    def exact(self, ix, h, a):
        return self.M[ix][:, h, a]

    def exact_other(self, ix):
        M = self.M[ix]; i = np.arange(G)
        m = (i[:, None] <= 3) & (i[None, :] <= 3)
        return 1 - M[:, m].sum(1)
