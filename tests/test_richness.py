"""Validation of the rarefaction estimator by simulation.

We cannot check this code against real data, because with real data we never
know the true richness. So we simulate: build a species pool whose size we
choose, generate checklists from it, and ask whether the estimator recovers
what we put in.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from richness import (incidence_stats, chao2, rarefy_richness, coverage_at,
                      observed_coverage, coverage_standardised_richness)

rng = np.random.default_rng(20260105)


def simulate_cell(S_true, T, a=0.3, b=3.0, seed=None):
    """Generate T checklists from a pool of S_true species.

    Each species i has an occupancy probability p_i -- the chance it appears on
    any given checklist. Drawing p from Beta(0.3, 3) produces the strongly
    right-skewed pattern real bird communities show: a few species on almost
    every list, a long tail of species seen once a season.
    """
    r = np.random.default_rng(seed)
    p = r.beta(a, b, size=S_true)
    detections = r.random((T, S_true)) < p          # T x S boolean matrix
    Y = detections.sum(axis=0)                       # incidence per species
    return Y[Y > 0], p


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return cond

results = []

# -- Test 1: the exact identity E[S(T)] == S_obs -----------------------------
print("\n[1] Rarefying to the full sample must return the observed richness exactly")
Y, _ = simulate_cell(300, 120, seed=1)
st = incidence_stats(Y, 120)
e_full = rarefy_richness(Y, 120, 120)
results.append(check("E[S(T)] == S_obs", abs(e_full - st["S_obs"]) < 1e-8,
                     f"E[S(T)]={e_full:.10f}  S_obs={st['S_obs']}"))

# -- Test 2: monotonicity ----------------------------------------------------
print("\n[2] Richness and coverage must both increase with more checklists")
ts = np.array([2, 5, 10, 20, 40, 80, 120])
sr = np.array([rarefy_richness(Y, 120, t) for t in ts])
cv = np.array([coverage_at(Y, 120, t) for t in ts])
results.append(check("richness increasing", bool(np.all(np.diff(sr) > 0))))
results.append(check("coverage increasing", bool(np.all(np.diff(cv) > 0))))
print(f"        t:        {'  '.join(f'{t:>6d}' for t in ts)}")
print(f"        E[S(t)]:  {'  '.join(f'{v:6.1f}' for v in sr)}")
print(f"        C(t):     {'  '.join(f'{v:6.3f}' for v in cv)}")

# -- Test 3: Chao2 converges on the truth as sampling deepens ----------------
print("\n[3] Chao2 must exceed observed richness and approach the truth as T grows")
print("    One FIXED community sampled at increasing depth. (Redrawing the")
print("    community at each T would compare different ecologies, not different")
print("    effort -- a mistake that makes the curve look non-monotone.)")
S_true = 250
p_fixed = np.random.default_rng(4242).beta(0.5, 2.5, size=S_true)
ok_bound, rows = True, []
for T in [25, 50, 100, 200, 400, 800]:
    r = np.random.default_rng(100 + T)
    Y = (r.random((T, S_true)) < p_fixed).sum(axis=0); Y = Y[Y > 0]
    s_obs = incidence_stats(Y, T)["S_obs"]
    c2 = chao2(Y, T)
    ok_bound &= (c2 >= s_obs - 1e-9)
    rows.append((T, s_obs, c2, 100 * s_obs / S_true, 100 * c2 / S_true))
print(f"\n        {'T':>5} {'S_obs':>7} {'Chao2':>8} {'S_obs %':>9} {'Chao2 %':>9}   (truth = 250)")
for T, s, c, ps, pc in rows:
    print(f"        {T:>5} {s:>7} {c:>8.1f} {ps:>8.0f}% {pc:>8.0f}%")
results.append(check("Chao2 >= S_obs always", ok_bound))
results.append(check("Chao2 beats raw count at every depth",
                     all(c > s for _, s, c, _, _ in rows)))
results.append(check("Chao2 within 5% of truth at T=800",
                     abs(rows[-1][2] - S_true) / S_true < 0.05,
                     f"Chao2={rows[-1][2]:.1f}"))

# -- Test 4: the point of the whole exercise ---------------------------------
print("\n[4] THE KEY TEST -- identical ecology, different survey effort")
print("    Twelve cells drawn from the SAME 250-species pool with the SAME detection")
print("    probabilities. Only the number of checklists differs. True richness is")
print("    constant, so any variation the estimator reports is an artefact of effort.")
S_true = 250
shared = np.random.default_rng(7).beta(0.3, 3.0, size=S_true)   # one fixed community
efforts = [15, 20, 30, 45, 60, 90, 120, 180, 250, 350, 500, 700]
raw, std, eff_used = [], [], []
print(f"\n        {'T':>5} {'S_obs':>7} {'coverage':>9} {'std. richness':>14}")
for T in efforts:
    r = np.random.default_rng(900 + T)
    det = r.random((T, S_true)) < shared
    Y = det.sum(axis=0); Y = Y[Y > 0]
    out = coverage_standardised_richness(Y, T, target_coverage=0.90)
    tag = f"{out['richness']:14.1f}" if out["attainable"] else f"{'below target':>14}"
    print(f"        {T:>5} {out['S_obs']:>7} {out['coverage_full']:>9.3f} {tag}")
    if out["attainable"]:
        raw.append(out["S_obs"]); std.append(out["richness"]); eff_used.append(T)

raw, std, eff_used = np.array(raw), np.array(std), np.array(eff_used)
r_raw = np.corrcoef(np.log(eff_used), raw)[0, 1]
r_std = np.corrcoef(np.log(eff_used), std)[0, 1]
cv_raw = raw.std() / raw.mean()
cv_std = std.std() / std.mean()
print(f"\n        correlation with log(effort):  raw S_obs {r_raw:+.3f}   standardised {r_std:+.3f}")
print(f"        coefficient of variation:      raw S_obs {cv_raw:.3f}    standardised {cv_std:.3f}")
results.append(check("raw richness is strongly effort-driven", r_raw > 0.9,
                     f"r={r_raw:+.3f}"))
results.append(check("standardised richness is not", abs(r_std) < 0.5,
                     f"r={r_std:+.3f}"))
results.append(check("standardisation shrinks spurious variation", cv_std < cv_raw / 2,
                     f"{cv_raw:.3f} -> {cv_std:.3f}"))

# -- Test 5: cells too thin to use are refused -------------------------------
print("\n[5] Under-surveyed cells must be refused, not guessed at")
Y, _ = simulate_cell(250, 4, seed=55)
out = coverage_standardised_richness(Y, 4, target_coverage=0.95)
results.append(check("thin cell flagged unattainable", not out["attainable"],
                     f"coverage={out['coverage_full']:.3f}"))

print("\n" + "=" * 72)
print(f"  {sum(results)}/{len(results)} checks passed")
print("=" * 72)
sys.exit(0 if all(results) else 1)
