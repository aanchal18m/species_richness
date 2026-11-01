"""Parts 5-6 machinery: spatially honest cross-validation on the equal-area grid.

Random k-fold lets a model score well by interpolating between a test cell and
its training neighbours a few km away. Two remedies, both reported:

  Blocked k-fold   cells are grouped into contiguous square blocks on the grid
                   lattice; whole blocks go to folds. Block size is explicit; it
                   should later be set from the residual autocorrelation range.

  NNDM LOO         (Mila et al. 2022) leave-one-out in which each held-out cell
                   also loses training neighbours, until the distribution of
                   CV nearest-neighbour distances matches the distribution of
                   prediction-cell -> nearest-training-cell distances. The CV
                   then faces the same extrapolation distances the map does.

All distances are Euclidean in the grid's Albers equal-area CRS (metres). That
projection is not equidistant; its distance error over India is bounded by its
scale factors, which tests/test_spatial_cv.py checks against geodesics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from grid import INDIA_AEA


# --- coordinates ------------------------------------------------------------
def aea_xy(lon, lat, crs: str = INDIA_AEA) -> np.ndarray:
    import pyproj
    t = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = t.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.column_stack([x, y])


# --- blocked k-fold ---------------------------------------------------------
def block_ids(xy: np.ndarray, block_size_m: float, origin=(0.0, 0.0)) -> np.ndarray:
    """Square blocks on the lattice: block (i, j) holds points with
    x0 + i*b <= x < x0 + (i+1)*b. Pass the grid's (x0, y0) as origin so a block
    is an exact union of whole cells when b is a multiple of the cell size."""
    xy = np.asarray(xy, float)
    i = np.floor((xy[:, 0] - origin[0]) / block_size_m).astype(np.int64)
    j = np.floor((xy[:, 1] - origin[1]) / block_size_m).astype(np.int64)
    return i * 1_000_003 + j


def blocked_folds(blocks: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Assign whole blocks to k folds, balancing cell counts: blocks in random
    order, then largest first, each to the currently smallest fold (LPT)."""
    blocks = np.asarray(blocks)
    labels, counts = np.unique(blocks, return_counts=True)
    if len(labels) < k:
        raise ValueError(f"{len(labels)} blocks cannot fill {k} folds; use smaller blocks")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    order = order[np.argsort(-counts[order], kind="stable")]
    load = np.zeros(k, dtype=np.int64)
    fold_of_block = {}
    for b in order:
        f = int(np.argmin(load))
        fold_of_block[labels[b]] = f
        load[f] += counts[b]
    return np.array([fold_of_block[b] for b in blocks], dtype=np.int64)


def iter_folds(folds: np.ndarray):
    """Yield (fold, train_idx, test_idx)."""
    for f in np.unique(folds):
        yield f, np.flatnonzero(folds != f), np.flatnonzero(folds == f)


# --- NNDM leave-one-out -----------------------------------------------------
def _ecdf_at(sorted_sample: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_sample, r, side="right") / len(sorted_sample)


def ecdf_gap(a: np.ndarray, b: np.ndarray, phi: float | None = None) -> float:
    """sup_r |F_a(r) - F_b(r)| over r <= phi: a Kolmogorov-Smirnov distance
    between two distance distributions."""
    a, b = np.sort(a), np.sort(b)
    r = np.concatenate([a, b])
    if phi is not None:
        r = r[r <= phi]
    return float(np.max(np.abs(_ecdf_at(a, r) - _ecdf_at(b, r)))) if len(r) else 0.0


@dataclass
class NNDMResult:
    exclude: list            # per training point: indices removed from its training set (incl. itself)
    gjstar: np.ndarray       # final CV nearest-neighbour distance per training point
    gj: np.ndarray           # plain LOO nearest-neighbour distances (the random-LOO control)
    gij: np.ndarray          # prediction point -> nearest training point distances
    phi: float
    iterations: int = field(default=0)

    def train_index(self, i: int, n: int) -> np.ndarray:
        mask = np.ones(n, bool)
        mask[self.exclude[i]] = False
        return np.flatnonzero(mask)


def nndm_loo(train_xy, pred_xy, phi: float | None = None, max_iter: int | None = None,
             k0: int = 32) -> NNDMResult:
    """NNDM LOO CV (Mila et al. 2022, Algorithm 1).

    While the CV ECDF lies above the prediction ECDF at some r <= phi (CV
    distances too short), take the smallest such CV distance and remove that
    point's current nearest neighbour from its training set."""
    train_xy = np.asarray(train_xy, float)
    pred_xy = np.asarray(pred_xy, float)
    n = len(train_xy)
    tree = cKDTree(train_xy)
    gij = tree.query(pred_xy, k=1)[0]
    phi = float(np.max(gij)) if phi is None else float(phi)

    def neighbours(i, kk):
        """Neighbours of i nearest first, i itself removed (with duplicate
        coordinates i need not come back first)."""
        d, j = tree.query(train_xy[i], k=kk)
        keep = j != i
        return d[keep], j[keep]

    kk = min(k0 + 1, n)
    nb = [neighbours(i, kk) for i in range(n)]
    dist, idx = [d for d, _ in nb], [j for _, j in nb]
    ptr = np.zeros(n, dtype=np.int64)                  # position of current NN in each list
    gj = np.array([d[0] for d in dist])
    gjstar = gj.copy()
    exclude = [[i] for i in range(n)]
    gij_sorted = np.sort(gij)
    max_iter = max_iter or 50 * n * 32

    it = 0
    while it < max_iter:
        f_cv = _ecdf_at(np.sort(gjstar), gjstar)
        f_pred = _ecdf_at(gij_sorted, gjstar)
        cand = np.flatnonzero((f_cv > f_pred) & (gjstar <= phi))
        if len(cand) == 0:
            break
        i = cand[np.argmin(gjstar[cand])]
        exclude[i].append(int(idx[i][ptr[i]]))         # drop its current nearest neighbour
        ptr[i] += 1
        if ptr[i] >= len(idx[i]):                      # neighbour list exhausted: fetch more
            dist[i], idx[i] = neighbours(i, min(2 * (len(idx[i]) + 1), n))
            if ptr[i] >= len(idx[i]):                  # every other point excluded
                gjstar[i] = np.inf
                it += 1
                continue
        gjstar[i] = dist[i][ptr[i]]
        it += 1
    return NNDMResult([np.array(e) for e in exclude], gjstar, gj, gij, phi, it)
