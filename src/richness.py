"""
Coverage-based rarefaction for incidence (checklist) data.

WHY THIS MODULE EXISTS
----------------------
Raw species counts per grid cell are not a measure of biodiversity; they are a
measure of biodiversity multiplied by how hard anyone looked. A cell with 400
eBird checklists will show more species than an ecologically identical cell with
20, purely because rare species need many visits to be detected at all.

If we modelled raw counts, the model's single strongest predictor would be
survey effort wearing an environmental disguise -- cells near cities are both
well-surveyed AND warm/low/built-up, so the model can score well by learning
"near a city" and reporting it as a climate effect. Every zone comparison we
later make would then be comparing survey intensity between zones rather than
ecology.

Chao et al. (2014) give the fix: standardise every cell to the same SAMPLE
COVERAGE -- the estimated proportion of the cell's total species-incidence mass
that the observed species account for. Comparing cells at equal coverage is
comparing them at equal completeness of inventory, which is the thing we
actually want held constant.

NOTATION (incidence / checklist data, following Chao et al. 2014)
-----------------------------------------------------------------
  T    number of sampling units (checklists) in the cell
  Y_i  number of checklists in which species i was detected (1..T)
  U    total incidences, sum of Y_i
  Q1   number of species detected in exactly one checklist  ("uniques")
  Q2   number of species detected in exactly two checklists  ("duplicates")

Q1 and Q2 carry the information about what is missing: many singletons means
many undetected species remain.

Reference: Chao, A. et al. (2014) Rarefaction and extrapolation with Hill
numbers: a framework for sampling and estimation in species diversity studies.
Ecological Monographs 84(1), 45-67.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


# ----------------------------------------------------------------------------
# Basic incidence statistics
# ----------------------------------------------------------------------------

def incidence_stats(Y: np.ndarray, T: int) -> dict:
    """Summarise an incidence-frequency vector.

    Y : array of per-species detection counts (only species with Y_i > 0)
    T : number of checklists in the cell
    """
    Y = np.asarray(Y, dtype=float)
    Y = Y[Y > 0]
    return {
        "T": int(T),
        "S_obs": int(Y.size),
        "U": float(Y.sum()),
        "Q1": int(np.sum(Y == 1)),
        "Q2": int(np.sum(Y == 2)),
    }


def chao2(Y: np.ndarray, T: int) -> float:
    """Chao2 asymptotic richness estimator for incidence data.

    S_obs plus an estimate of the number of species present but never detected.
    The correction term is driven by the singleton/doubleton ratio: if you are
    still finding species only once, you have not finished looking.
    """
    st = incidence_stats(Y, T)
    S_obs, Q1, Q2, T = st["S_obs"], st["Q1"], st["Q2"], st["T"]
    if T < 2:
        return float(S_obs)
    corr = (T - 1) / T
    if Q2 > 0:
        return S_obs + corr * (Q1 ** 2) / (2 * Q2)
    # Q2 == 0: bias-corrected form, avoids division by zero
    return S_obs + corr * Q1 * (Q1 - 1) / 2


# ----------------------------------------------------------------------------
# Rarefaction: expected richness and coverage at a reduced number of checklists
# ----------------------------------------------------------------------------

def _log_falling_ratio(n: int | float, y: float, t: float) -> float:
    """log[ C(n - y, t) / C(n, t) ] computed with log-gammas.

    This is the probability that a species with incidence y is MISSED when we
    draw t of the n checklists at random. Using gammaln keeps it stable for
    large T and admits fractional t, which we need for coverage standardisation.
    Returns -inf when the species cannot possibly be missed (n - y < t).
    """
    if n - y < t:
        return -np.inf
    return (
        gammaln(n - y + 1) - gammaln(n - y - t + 1)
        - (gammaln(n + 1) - gammaln(n - t + 1))
    )


def rarefy_richness(Y: np.ndarray, T: int, t: float) -> float:
    """Expected number of species in a random subsample of t checklists.

    E[S(t)] = S_obs - sum_i P(species i missed in t of T checklists)

    At t == T every term is zero and this returns S_obs exactly. That identity
    is the sharpest unit test available for this function.
    """
    Y = np.asarray(Y, dtype=float)
    Y = Y[Y > 0]
    if t <= 0:
        return 0.0
    if t > T:
        raise ValueError("rarefaction only; t must not exceed T")
    logs = np.array([_log_falling_ratio(T, y, t) for y in Y])
    return float(Y.size - np.exp(logs[np.isfinite(logs)]).sum())


def rarefy_coverage(Y: np.ndarray, T: int, t: float) -> float:
    """Expected sample coverage at t checklists, for t < T.

    C(t) = 1 - sum_i (Y_i / U) * [ (T-Y_i)_t / (T-1)_t ]

    where (n)_t is a falling factorial. Note the two DIFFERENT bases: the
    numerator falls from T - Y_i, the denominator from T - 1. Collapsing them
    to a single base is an easy mistake and makes coverage come back as exactly
    1.0 for every cell, which silently disables the whole correction.

    Only species with T - Y_i >= t contribute; the rest cannot be missed in a
    draw of t units and their term is zero.

    Undefined at t == T -- use observed_coverage there. `coverage_at` picks the
    right one for you.
    """
    Y = np.asarray(Y, dtype=float)
    Y = Y[Y > 0]
    U = Y.sum()
    if U == 0 or t <= 0:
        return 0.0
    if t >= T:
        raise ValueError("rarefaction coverage is defined only for t < T; "
                         "use observed_coverage at the full sample")
    contrib = Y[(T - Y) >= t]
    if contrib.size == 0:
        return 1.0
    logs = (gammaln(T - contrib + 1) - gammaln(T - contrib - t + 1)
            - gammaln(T) + gammaln(T - t))
    return float(1.0 - np.sum((contrib / U) * np.exp(logs)))


def coverage_at(Y: np.ndarray, T: int, t: float) -> float:
    """Sample coverage at t checklists, valid across the whole range 0 < t <= T."""
    if t >= T:
        return observed_coverage(Y, T)
    return rarefy_coverage(Y, T, t)


def observed_coverage(Y: np.ndarray, T: int) -> float:
    """Turing-style coverage estimate for the full sample of T checklists."""
    st = incidence_stats(Y, T)
    U, Q1, Q2, T = st["U"], st["Q1"], st["Q2"], st["T"]
    if U == 0:
        return 0.0
    if Q1 == 0:
        return 1.0
    if Q2 > 0:
        f = ((T - 1) * Q1) / ((T - 1) * Q1 + 2 * Q2)
    else:
        f = ((T - 1) * (Q1 - 1)) / ((T - 1) * (Q1 - 1) + 2) if Q1 > 1 else 0.0
    return float(1.0 - (Q1 / U) * f)


# ----------------------------------------------------------------------------
# The function the pipeline actually calls
# ----------------------------------------------------------------------------

def coverage_standardised_richness(
    Y: np.ndarray, T: int, target_coverage: float = 0.95, tol: float = 1e-6
) -> dict:
    """Richness rarefied to a common coverage -- the project's response variable.

    Finds the (possibly fractional) number of checklists t* at which expected
    coverage equals `target_coverage`, then returns expected richness there.
    Cells are then compared at equal inventory completeness rather than at
    equal, or unequal, survey effort.

    Returns a dict so the caller keeps the diagnostics, not just the number.
    A cell whose maximum attainable coverage is below target is reported as
    `attainable=False` and must be excluded from model fitting -- it has not
    been surveyed well enough to carry a trustworthy response value.
    """
    st = incidence_stats(Y, T)
    if st["S_obs"] == 0 or T < 2:
        return {**st, "attainable": False, "t_star": np.nan,
                "coverage_full": 0.0, "richness": np.nan}

    c_full = observed_coverage(Y, T)
    if c_full < target_coverage:
        return {**st, "attainable": False, "t_star": np.nan,
                "coverage_full": c_full, "richness": np.nan,
                "chao2": chao2(Y, T)}

    # Coverage is monotone increasing in t, so bisect.
    lo, hi = 1.0, float(T)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if coverage_at(Y, T, mid) < target_coverage:
            lo = mid
        else:
            hi = mid
    t_star = hi
    return {
        **st,
        "attainable": True,
        "t_star": t_star,
        "coverage_full": c_full,
        "richness": rarefy_richness(Y, T, t_star),
        "chao2": chao2(Y, T),
    }
