"""Tests for src/spatial_cv.py.

Run: python tests/test_spatial_cv.py   (or pytest)
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import spatial_cv as sc  # noqa: E402
from grid import INDIA_AEA  # noqa: E402

CITIES = {  # lon, lat
    "Kanyakumari": (77.54, 8.08), "Srinagar": (74.80, 34.08), "Mumbai": (72.88, 19.08),
    "Kolkata": (88.36, 22.57), "Chennai": (80.27, 13.08), "Delhi": (77.21, 28.61),
    "Leh": (77.58, 34.16), "Guwahati": (91.74, 26.14), "Ahmedabad": (72.57, 23.02),
    "Imphal": (93.94, 24.82), "Port Blair": (92.73, 11.62), "Bengaluru": (77.59, 12.97),
}
PAIRS = [("Kanyakumari", "Srinagar"), ("Mumbai", "Kolkata"), ("Chennai", "Delhi"),
         ("Leh", "Guwahati"), ("Ahmedabad", "Imphal"), ("Port Blair", "Chennai"),
         ("Bengaluru", "Kanyakumari"), ("Delhi", "Leh")]


def _scale_bounds():
    """Smallest and largest Albers scale factor (along meridians or parallels)
    over India's extent: any distance ratio must lie between them."""
    import pyproj
    lon, lat = np.meshgrid(np.linspace(68, 98, 31), np.linspace(6, 37.5, 64))
    f = pyproj.Proj(INDIA_AEA).get_factors(lon.ravel(), lat.ravel())
    s = np.concatenate([f.meridional_scale, f.parallel_scale])
    return s.min(), s.max()


def test_equal_area_distances_agree_with_geodesics_within_projection_bounds():
    import pyproj
    geod = pyproj.Geod(ellps="WGS84")
    lo, hi = _scale_bounds()
    ratios = {}
    for a, b in PAIRS:
        (x1, y1), (x2, y2) = sc.aea_xy(*zip(CITIES[a], CITIES[b]))
        d_aea = np.hypot(x2 - x1, y2 - y1)
        d_geo = geod.inv(*CITIES[a], *CITIES[b])[2]
        ratios[f"{a}-{b}"] = d_aea / d_geo
    bad = {k: round(v, 4) for k, v in ratios.items() if not lo - 1e-6 <= v <= hi + 1e-6}
    assert not bad, f"outside projection scale bounds [{lo:.4f}, {hi:.4f}]: {bad}"
    # 10 km neighbours at the latitude extremes, both directions
    for lat in (8.0, 34.0):
        for dlon, dlat in ((0.0, 0.09), (0.1, 0.0)):
            (x1, y1), (x2, y2) = sc.aea_xy([78, 78 + dlon], [lat, lat + dlat])
            r = np.hypot(x2 - x1, y2 - y1) / geod.inv(78, lat, 78 + dlon, lat + dlat)[2]
            assert lo - 1e-6 <= r <= hi + 1e-6, (lat, dlon, dlat, r)
    assert hi - lo < 0.06, (lo, hi)          # the distortion stays a few percent over India


# --- blocked folds -----------------------------------------------------------
def _lattice(seed=0):
    """Irregular 10 km lattice ~600 x 400 km with holes, like a coastline."""
    rng = np.random.default_rng(seed)
    i, j = np.meshgrid(np.arange(60), np.arange(40))
    xy = np.column_stack([i.ravel() * 10_000 + 5_000, j.ravel() * 10_000 + 5_000]).astype(float)
    keep = rng.random(len(xy)) > 0.15
    keep &= ~((xy[:, 0] > 400_000) & (xy[:, 1] > 300_000))   # a missing corner
    return xy[keep]


def _check_folds(xy, blocks, folds, k, max_ratio=None):
    for f, train, test in sc.iter_folds(folds):
        shared = set(blocks[test]) & set(blocks[train])
        assert not shared, f"fold {f}: {len(shared)} blocks split between train and test"
    sizes = np.bincount(folds, minlength=k)
    # Whole blocks cannot always split evenly (22 equal blocks into 5 folds forces
    # 5 vs 4); largest-first assignment guarantees max - min <= the largest block.
    largest_block = np.unique(blocks, return_counts=True)[1].max()
    assert sizes.min() > 0 and sizes.max() - sizes.min() <= largest_block, (sizes, largest_block)
    if max_ratio is not None:
        assert sizes.max() / sizes.min() <= max_ratio, sizes


def test_blocked_folds_keep_blocks_whole_and_balanced():
    xy = _lattice()
    blocks = sc.block_ids(xy, 100_000)
    folds = sc.blocked_folds(blocks, k=5, seed=3)
    _check_folds(xy, blocks, folds, 5)
    for b in np.unique(blocks):                 # contiguous squares no wider than the block
        pts = xy[blocks == b]
        assert np.ptp(pts[:, 0]) < 100_000 and np.ptp(pts[:, 1]) < 100_000


def test_blocked_folds_on_the_real_grid():
    import geopandas as gpd
    from build_grid import GRID, load_meta
    if not GRID.exists():
        raise unittest.SkipTest("grid.gpkg not built")
    g = gpd.read_file(GRID, layer="grid")
    meta = load_meta()
    xy = np.column_stack([g.geometry.centroid.x, g.geometry.centroid.y])
    blocks = sc.block_ids(xy, 200_000, origin=(meta["x0"], meta["y0"]))
    folds = sc.blocked_folds(blocks, k=5)
    _check_folds(xy, blocks, folds, 5, max_ratio=1.05)
    # lattice-aligned blocks of 200 km hold whole 10 km cells: 20 x 20 at most
    assert np.bincount(np.unique(blocks, return_inverse=True)[1]).max() <= 400


def test_too_few_blocks_is_an_error():
    try:
        sc.blocked_folds(np.array([1, 1, 2, 2]), k=3)
    except ValueError:
        pass
    else:
        raise AssertionError("3 folds from 2 blocks accepted")


# --- NNDM --------------------------------------------------------------------
def _clustered_vs_spread(seed=0, n_clusters=6, per=50, n_pred=1000, extent=1_000_000):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, extent, (n_clusters, 2))
    train = np.concatenate([c + rng.normal(0, 15_000, (per, 2)) for c in centres])
    pred = rng.uniform(0, extent, (n_pred, 2))
    return train, pred


def _one_sided(a, b, phi):
    """max over r <= phi of F_a(r) - F_b(r)."""
    a, b = np.sort(a), np.sort(b)
    r = np.concatenate([a, b])
    r = r[r <= phi]
    return float(np.max(sc._ecdf_at(a, r) - sc._ecdf_at(b, r)))


def test_nndm_matches_prediction_distances_where_random_loo_fails():
    """NNDM can only lengthen CV distances, so its guarantee is one-sided: the CV
    ECDF never lies above the prediction ECDF for r <= phi. With few, widely
    spaced clusters the other side keeps a residual -- distances jump from
    within-cluster to between-cluster -- so the two-sided match is asserted on a
    design whose clusters resolve the distance distribution."""
    for kw, max_gap in (({}, None), ({"n_clusters": 20, "per": 15}, 0.08)):
        train, pred = _clustered_vs_spread(**kw)
        res = sc.nndm_loo(train, pred)
        gap_nndm = sc.ecdf_gap(res.gjstar, res.gij, res.phi)
        gap_loo = sc.ecdf_gap(res.gj, res.gij, res.phi)
        assert gap_loo > 0.6, f"{kw}: control should fail badly, random LOO gap {gap_loo:.3f}"
        assert _one_sided(res.gjstar, res.gij, res.phi) <= 1e-12, kw
        assert gap_nndm < gap_loo / 4, (kw, gap_nndm, gap_loo)
        if max_gap is not None:
            assert gap_nndm < max_gap, (kw, gap_nndm)


def test_nndm_changes_little_when_training_already_matches():
    rng = np.random.default_rng(5)
    train = rng.uniform(0, 1_000_000, (300, 2))
    pred = rng.uniform(0, 1_000_000, (1000, 2))
    res = sc.nndm_loo(train, pred)
    extra = np.mean([len(e) - 1 for e in res.exclude])
    assert extra < 1.0, f"{extra:.2f} neighbours removed per point on average"
    assert sc.ecdf_gap(res.gjstar, res.gij, res.phi) <= sc.ecdf_gap(res.gj, res.gij, res.phi) + 0.02


def test_nndm_distances_are_consistent_with_exclusions():
    """gjstar[i] must be the distance from i to its nearest NON-excluded point."""
    train, pred = _clustered_vs_spread(seed=2, per=30)
    res = sc.nndm_loo(train, pred)
    for i in range(len(train)):
        keep = res.train_index(i, len(train))
        assert i not in keep
        brute = np.min(np.hypot(*(train[keep] - train[i]).T))
        assert np.isclose(brute, res.gjstar[i]), (i, brute, res.gjstar[i])


if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS  {name}")
        except unittest.SkipTest as e:
            skipped += 1
            print(f"SKIP  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{passed + failed} checks passed" + (f", {skipped} skipped" if skipped else ""))
    sys.exit(1 if failed else 0)
