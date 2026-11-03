"""Turn a spatial neighbour graph into CellPhoneDB microenvironments.

CellPhoneDB tests every pair of cell types by default. On spatial data that is
wrong: two cell types that never sit near each other cannot interact. v5 lets
you pass a *microenvironments* file (cell_type -> microenvironment); only cell
types sharing a microenvironment are tested. A spatial niche is exactly a
microenvironment, so we:

1. build a radius neighbour graph over the spots (the ``radius`` is the knob
   you set in radius-based spatial LR tools),
2. describe each spot by the cell-type composition of its neighbourhood,
3. cluster spots into niches, and
4. list, per niche, the labels that are actually present.

At a graft border the mixed niche will contain both ``human_*`` and ``rat_*``
labels, so CellPhoneDB tests them against each other -- which, combined with the
cross-species database, is what surfaces cross-border interactions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def radius_neighbors(coords: np.ndarray, radius: float) -> list[np.ndarray]:
    """For each spot, indices of spots within ``radius`` (excluding self)."""
    tree = cKDTree(coords)
    nbrs = tree.query_ball_point(coords, r=radius)
    return [np.array([j for j in nb if j != i]) for i, nb in enumerate(nbrs)]


def neighborhood_composition(
    labels: np.ndarray, neighbors: list[np.ndarray], categories: list[str]
) -> pd.DataFrame:
    """Fraction of each label among each spot's neighbours (incl. self)."""
    cat_idx = {c: i for i, c in enumerate(categories)}
    comp = np.zeros((len(labels), len(categories)))
    for i, nb in enumerate(neighbors):
        idx = np.append(nb, i).astype(int)
        for j in idx:
            comp[i, cat_idx[labels[j]]] += 1
        comp[i] /= len(idx)
    return pd.DataFrame(comp, columns=categories)


def build_microenvironments(
    adata,
    label_key: str = "label",
    radius: float = 2.0,
    n_niches: int = 6,
    presence_threshold: float = 0.05,
    spatial_key: str = "spatial",
    random_state: int = 0,
) -> pd.DataFrame:
    """Return a CellPhoneDB microenvironments table (``cell_type``, ``microenvironment``).

    Parameters
    ----------
    radius
        Neighbourhood radius in the units of ``adata.obsm[spatial_key]``. For a
        Visium grid, ~1 selects the 6 immediate neighbours; larger values widen
        the border zone considered.
    n_niches
        Number of spatial niches (KMeans clusters over neighbourhood
        composition). Each becomes one microenvironment.
    presence_threshold
        A label is listed in a niche if its mean neighbourhood fraction within
        that niche exceeds this value (drops incidental contamination).
    """
    from sklearn.cluster import KMeans

    coords = np.asarray(adata.obsm[spatial_key])
    labels = np.asarray(adata.obs[label_key].astype(str))
    categories = sorted(set(labels))

    neighbors = radius_neighbors(coords, radius)
    comp = neighborhood_composition(labels, neighbors, categories)

    n_niches = min(n_niches, len(comp))
    km = KMeans(n_clusters=n_niches, random_state=random_state, n_init=10)
    niche = km.fit_predict(comp.values)

    rows = []
    for k in range(n_niches):
        mask = niche == k
        if not mask.any():
            continue
        mean_comp = comp.values[mask].mean(0)
        present = [categories[j] for j in range(len(categories)) if mean_comp[j] > presence_threshold]
        for lab in present:
            rows.append({"cell_type": lab, "microenvironment": f"niche_{k}"})

    me = pd.DataFrame(rows).drop_duplicates()
    # every label must appear in >=1 microenvironment or CellPhoneDB drops it;
    # attach any orphan label to its single most-common niche.
    missing = set(categories) - set(me.cell_type)
    for lab in missing:
        dom = np.bincount(niche[labels == lab]).argmax()
        me.loc[len(me)] = {"cell_type": lab, "microenvironment": f"niche_{dom}"}
    adata.obs["niche"] = pd.Categorical([f"niche_{k}" for k in niche])
    return me.reset_index(drop=True)


def cross_species_niches(me: pd.DataFrame) -> list[str]:
    """Names of microenvironments spanning >=2 species (labels are ``species_celltype``)."""
    out = []
    for env, grp in me.groupby("microenvironment"):
        sp = {c.split("_", 1)[0] for c in grp.cell_type}
        if len(sp) >= 2:
            out.append(env)
    return out


# --- scope: restrict which units enter the analysis ----------------------
# By default CellPhoneDB's per-cell-type mean is over *all* cells of that type.
# These helpers subset first, so e.g. the human-fibroblast mean is taken only
# over the border region you care about.


def restrict_to_region(units, regions, region_key: str = "region"):
    """Keep only units whose ``obs[region_key]`` is in ``regions``.

    Use with your own spatial annotations (e.g. 'graft_border', 'host_distal').
    ``regions`` may be a single label or a list.
    """
    if isinstance(regions, str):
        regions = [regions]
    if region_key not in units.obs:
        raise KeyError(f"{region_key!r} not in obs; available: {list(units.obs.columns)}")
    mask = units.obs[region_key].astype(str).isin(regions).values
    if not mask.any():
        raise ValueError(f"no units in regions {regions}")
    return units[mask].copy()


def celltype_proximity(units, label_key: str = "label", radius: float = 1.0,
                       spatial_key: str = "spatial", min_frac: float = 0.0) -> set:
    """Which cell-type pairs are co-located within ``radius``.

    Returns a set of ``frozenset({labelA, labelB})`` (including self-pairs) for
    pairs where at least ``min_frac`` of A-units have a B-unit within ``radius``.
    Used to make proximity *ligand-class-aware*: a small ``radius`` = physical
    contact (juxtacrine ligands), a larger one = diffusion (secreted ligands).
    Because deconvolved pseudo-cells share their spot's coordinates, cell types
    in the same spot are always in contact (distance 0).
    """
    from collections import defaultdict

    coords = np.asarray(units.obsm[spatial_key])
    labels = np.asarray(units.obs[label_key].astype(str))
    neighbors = radius_neighbors(coords, radius)
    pair_hits = defaultdict(int)
    label_n = defaultdict(int)
    for i, nb in enumerate(neighbors):
        a = labels[i]
        label_n[a] += 1
        seen = set(labels[nb]) if len(nb) else set()
        seen.add(a)  # same spot / same unit type counts as in-contact
        for b in seen:
            pair_hits[(a, b)] += 1
    prox = set()
    for (a, b), c in pair_hits.items():
        if label_n[a] and c / label_n[a] > min_frac:
            prox.add(frozenset((a, b)))
    return prox


def restrict_to_window(units, center, radius: float, spatial_key: str = "spatial"):
    """Keep units within ``radius`` of a ``center`` (coords, or an obs_name).

    Implements the 'spot g, radius r' idea: a local disk around a chosen point.
    """
    coords = np.asarray(units.obsm[spatial_key])
    if isinstance(center, str):
        center = coords[units.obs_names.get_loc(center)]
    center = np.asarray(center, dtype=float)
    d = np.linalg.norm(coords - center, axis=1)
    mask = d <= radius
    if not mask.any():
        raise ValueError("no units within the window")
    return units[mask].copy()
