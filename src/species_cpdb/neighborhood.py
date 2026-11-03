"""Per-spot / per-cell *adjacency* engine (the finer alternative to CellPhoneDB).

CellPhoneDB scores interactions at the cell-type level: it asks "is the ligand's
mean high in type A and the receptor's mean high in type B", gated (via
microenvironments) to co-located types. It does **not** ask whether the *specific*
ligand-high cells sit next to the *specific* receptor-high cells.

This engine does. For a directed pair ``ligand(S1) -> receptor(S2)`` and cell
types ``A (in S1)`` and ``B (in S2)`` it scores the mean of
``ligand[i] * receptor[j]`` over spatial edges where an A-cell ``i`` is within
``radius`` of a B-cell ``j``. Significance comes from a permutation null that
shuffles expression *within each species* (preserving the two-species structure
and the graph), so a low p-value means high-ligand and high-receptor cells are
adjacent more than expected -- true spatial co-localisation, not just co-presence.

Ligand/receptor gene lists, partner species and cross-species confidence are read
from the ``interaction_meta.csv`` produced alongside the cross-species database,
so this reuses the same database and namespace as the CellPhoneDB engine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .spatial import radius_neighbors


@dataclass
class NeighborhoodResult:
    hits_long: pd.DataFrame     # every tested interaction x cell-pair (with pvalue, mean)
    units: object = None

    def cross_species_hits(self, pvalue: float = 0.05, cross_only: bool = True) -> pd.DataFrame:
        df = self.hits_long
        out = df[df["pvalue"] <= pvalue]
        if cross_only:
            out = out[out["cross_species"]]
        return out.sort_values("weighted_score", ascending=False).reset_index(drop=True)


def _parse_partners(interactors: str) -> tuple[list[str], list[str]]:
    """'ssc:TGFB1->hsa:TGFBR1+hsa:TGFBR2' -> (['ssc:TGFB1'], ['hsa:TGFBR1','hsa:TGFBR2'])."""
    a, b = interactors.split("->", 1)
    return a.split("+"), b.split("+")


def neighborhood_ligrec(
    units,
    interaction_meta: pd.DataFrame,
    radius: float = 1.5,
    iterations: int = 1000,
    label_key: str = "label",
    spatial_key: str = "spatial",
    min_edges: int = 5,
    seed: int = 0,
) -> NeighborhoodResult:
    """Adjacency-based ligand-receptor test over a spatial neighbour graph.

    Parameters
    ----------
    units
        AnnData of analysis units (pseudo-cells or cells) with prefixed
        ``var_names`` matching the database, ``.obs[label_key]`` = ``species_celltype``
        and ``.obsm[spatial_key]``.
    interaction_meta
        The ``interaction_meta.csv`` from the cross-species database.
    radius
        Neighbourhood radius (an A-cell and B-cell within this distance form an edge).
    min_edges
        Skip an (interaction, A->B) cell-pair with fewer than this many A->B edges.
    """
    rng = np.random.default_rng(seed)
    coords = np.asarray(units.obsm[spatial_key])
    labels = np.asarray(units.obs[label_key].astype(str))
    species = np.array([lab.split("_", 1)[0] for lab in labels])
    var_pos = {g: i for i, g in enumerate(units.var_names)}
    # keep sparse (HD / single-cell sections are large); densify per interaction
    X = units.X.tocsc() if hasattr(units.X, "tocsc") else np.asarray(units.X)

    # directed edge list (i -> j) for neighbours within radius
    nbrs = radius_neighbors(coords, radius)
    src = np.repeat(np.arange(len(labels)), [len(nb) for nb in nbrs])
    dst = np.concatenate(nbrs) if len(nbrs) and any(len(nb) for nb in nbrs) else np.array([], dtype=int)
    dst = dst.astype(int)

    label_codes, label_uniques = pd.factorize(labels)
    src_lab = label_codes[src]
    dst_lab = label_codes[dst]
    src_sp = species[src]
    dst_sp = species[dst]

    rows = []
    for rec in interaction_meta.itertuples(index=False):
        lig_genes, rec_genes = _parse_partners(rec.interactors)
        if any(g not in var_pos for g in lig_genes + rec_genes):
            continue  # gene not measured in this dataset
        s1, s2 = rec.species_a, rec.species_b
        # ligand score per cell (mean of ligand genes), receptor score (min over
        # complex subunits). Gene prefixes already zero out the wrong species.
        lig = _dense_cols(X, [var_pos[g] for g in lig_genes]).mean(1)
        rcp = _dense_cols(X, [var_pos[g] for g in rec_genes]).min(1)

        # edges valid for this interaction: A-cell (species s1) -> B-cell (species s2)
        emask = (src_sp == s1) & (dst_sp == s2)
        if not emask.any():
            continue
        e_src, e_dst = src[emask], dst[emask]
        e_group_a, e_group_b = src_lab[emask], dst_lab[emask]
        # unique (A,B) group per edge
        pair_key = e_group_a.astype(np.int64) * len(label_uniques) + e_group_b
        uniq_pairs, grp = np.unique(pair_key, return_inverse=True)
        counts = np.bincount(grp, minlength=len(uniq_pairs))

        prod = lig[e_src] * rcp[e_dst]
        obs = np.bincount(grp, weights=prod, minlength=len(uniq_pairs)) / np.maximum(counts, 1)

        # permutation null: shuffle ligand within s1 cells, receptor within s2 cells
        s1_idx = np.where(species == s1)[0]
        s2_idx = np.where(species == s2)[0]
        exceed = np.zeros(len(uniq_pairs))
        for _ in range(iterations):
            lig_p = lig.copy()
            rcp_p = rcp.copy()
            lig_p[s1_idx] = lig[rng.permutation(s1_idx)]
            rcp_p[s2_idx] = rcp[rng.permutation(s2_idx)]
            prod_p = lig_p[e_src] * rcp_p[e_dst]
            perm = np.bincount(grp, weights=prod_p, minlength=len(uniq_pairs)) / np.maximum(counts, 1)
            exceed += perm >= obs
        pvals = (1.0 + exceed) / (iterations + 1.0)

        for k, pk in enumerate(uniq_pairs):
            if counts[k] < min_edges:
                continue
            a_lab = label_uniques[pk // len(label_uniques)]
            b_lab = label_uniques[pk % len(label_uniques)]
            rows.append({
                "interacting_pair": rec.interactors.replace("->", "_"),
                "cell_pair": f"{a_lab}|{b_lab}",
                "pvalue": float(pvals[k]),
                "mean": float(obs[k]),
                "n_edges": int(counts[k]),
                "species_a": s1, "species_b": s2,
                "cross_species": bool(rec.cross_species),
                "direction": (f"{s1}->{s2}" if s1 != s2 else f"within-{s1}"),
                "confidence": float(rec.confidence),
            })

    hits = pd.DataFrame(rows)
    if len(hits):
        hits["weighted_score"] = -np.log10(hits["pvalue"].clip(lower=1.0 / (iterations + 1))) * hits["confidence"]
    return NeighborhoodResult(hits_long=hits, units=units)


def _dense_cols(X, cols):
    """Columns of a dense or sparse matrix as a dense ndarray."""
    sub = X[:, cols]
    return sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
