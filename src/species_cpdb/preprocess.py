"""Turn platform-specific spatial data into a common set of *analysis units*.

Different platforms resolve to different things:

* ``visium_v1`` -- spots are mixtures of several cells. Given cell2location
  (or Tangram) **abundances** per spot and a cell-type **reference signature**,
  we attribute each spot's *observed* expression to its cell types by their
  ``abundance x signature`` share (cell2location's own generative assumption),
  producing one pseudo-cell per (spot, cell type). Pseudo-cells inherit the
  spot's coordinates, so cell types sharing a spot are co-located (within-spot
  interactions) and neighbouring spots drive between-spot interactions.

* ``xenium`` / ``merfish`` / ``cosmx`` -- already single cell. Each segmented
  cell is one unit; no deconvolution.

* ``visium_hd`` -- two routes:

  - **bin2cell** (recommended): 2 µm bins are grouped into segmented cells by
    `bin2cell <https://github.com/Teichlab/bin2cell>`_ (``b2c.bin_to_cell``).
    Each cell is one unit, see :func:`from_bin2cell`. Coordinates are converted
    from full-resolution pixels to **microns**, so radii are given in µm.
  - **binned deconvolution**: 8/16 µm bins + cell2location abundances, the same
    pseudo-cell route as ``visium_v1``.

Every path returns an AnnData of units with a two-species prefixed
``var_names``, ``.obs['label']`` = ``species_celltype``, ``.obs['spot']`` (origin
spot/cell id) and ``.obsm['spatial']`` -- exactly what
:func:`species_cpdb.spatial.build_microenvironments` and
:func:`species_cpdb.run.run_analysis` consume.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp_sparse

from ._namespace import add_prefix

PLATFORMS = {"visium_v1", "visium_hd", "xenium", "merfish", "cosmx"}
SINGLE_CELL_PLATFORMS = {"xenium", "merfish", "cosmx"}


def _infer_gene_species(signature: pd.DataFrame, celltype_species: dict) -> dict:
    """Assign each signature gene to the species of the cell types expressing it."""
    gene_species = {}
    for g in signature.columns:
        col = signature[g]
        expressing = col[col > 0].index
        specs = {celltype_species[c] for c in expressing if c in celltype_species}
        if len(specs) == 1:
            gene_species[g] = specs.pop()
        elif specs:
            # tie-break by which species contributes more signal
            by_sp = {}
            for c in expressing:
                by_sp[celltype_species[c]] = by_sp.get(celltype_species[c], 0) + col[c]
            gene_species[g] = max(by_sp, key=by_sp.get)
    return gene_species


def deconvolve_to_pseudocells(
    spot_adata,
    abundance: pd.DataFrame,
    signature: pd.DataFrame,
    celltype_species: dict,
    min_abundance: float = 0.1,
    spatial_key: str = "spatial",
    carry_obs: list[str] | None = None,
    layer: str | None = None,
) -> ad.AnnData:
    """Visium v1: attribute observed spot expression to per-cell-type pseudo-cells.

    Parameters
    ----------
    spot_adata
        Spots x genes AnnData of *observed* expression (bare gene symbols, human
        uppercase / rat title-case as they come off the two references), with
        coordinates in ``.obsm[spatial_key]``.
    abundance
        Spots x cell-types cell abundances (index = spot names).
    signature
        Cell-types x genes reference signature (cell2location ``inf_aver``).
    celltype_species
        Maps each cell type -> 'human' / 'rat'.
    min_abundance
        A (spot, cell type) pseudo-cell is created only where abundance exceeds
        this (drops cell types that aren't really present in the spot).
    carry_obs
        ``spot_adata.obs`` columns to copy onto the pseudo-cells (e.g. a region
        annotation like ``graft_border``).
    """
    spots = [s for s in spot_adata.obs_names if s in abundance.index]
    abundance = abundance.loc[spots]
    obs_src = spot_adata[spots]
    X = obs_src.layers[layer] if layer else obs_src.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    obs_genes = list(spot_adata.var_names)
    gene_pos = {g: i for i, g in enumerate(obs_genes)}

    gene_species = _infer_gene_species(signature, celltype_species)
    species_list = sorted(set(celltype_species.values()))

    # merged (prefixed) feature space, restricted to genes present in observed data
    var_names, var_pos = [], {}
    for g in signature.columns:
        if g in gene_pos and g in gene_species:
            name = add_prefix(g, gene_species[g])
            var_pos[name] = len(var_names)
            var_names.append(name)

    coords = np.asarray(spot_adata[spots].obsm[spatial_key])
    rows, obs_records, row_coords = [], [], []

    for sp in species_list:
        C = [c for c in signature.index if celltype_species.get(c) == sp and c in abundance.columns]
        G = [g for g in signature.columns if gene_species.get(g) == sp and g in gene_pos]
        if not C or not G:
            continue
        A = abundance[C].values                                   # n_spots x k
        S = signature.loc[C, G].values                            # k x m
        O = X[:, [gene_pos[g] for g in G]]                        # n_spots x m observed
        denom = A @ S                                             # n_spots x m
        denom_safe = np.where(denom > 0, denom, 1.0)
        Gpref = [add_prefix(g, sp) for g in G]
        col_idx = [var_pos[gp] for gp in Gpref]

        for si in range(len(spots)):
            for ci, c in enumerate(C):
                if A[si, ci] <= min_abundance:
                    continue
                # share of observed expression attributed to this cell type
                share = (A[si, ci] * S[ci, :]) / denom_safe[si, :]
                attr = share * O[si, :]
                row = np.zeros(len(var_names))
                row[col_idx] = attr
                rows.append(row)
                # cell-type keys may already encode species (e.g. 'human_fibroblast')
                label = c if str(c).startswith(f"{sp}_") else f"{sp}_{c}"
                rec = {"label": label, "cell_type": c, "species": sp, "spot": spots[si],
                       "abundance": float(A[si, ci])}
                if carry_obs:
                    for k in carry_obs:
                        rec[k] = spot_adata.obs.loc[spots[si], k]
                obs_records.append(rec)
                row_coords.append(coords[si])

    units = ad.AnnData(X=np.array(rows))
    units.var_names = var_names
    units.obs = pd.DataFrame(obs_records).reset_index(drop=True)
    units.obs_names = [f"{r['spot']}::{r['label']}" for r in obs_records]
    for col in ("label", "cell_type", "species", "spot"):
        units.obs[col] = pd.Categorical(units.obs[col])
    units.obsm[spatial_key] = np.array(row_coords)
    return units


def cells_as_units(
    adata,
    celltype_key: str,
    species,
    spatial_key: str = "spatial",
    var_is_prefixed: bool = False,
    carry_obs: list[str] | None = None,
) -> ad.AnnData:
    """Single-cell platforms (Xenium/MERFISH/CosMx): one cell = one unit.

    ``species`` is either an ``obs`` column name holding 'human'/'rat' per cell,
    or a dict mapping cell type -> species. Genes are prefixed per the cell's
    species unless ``var_is_prefixed`` (already ``hsa:``/``rno:``).
    """
    units = adata.copy()
    if isinstance(species, str):
        sp_per_cell = units.obs[species].astype(str).values
    else:
        sp_per_cell = np.array([species[c] for c in units.obs[celltype_key].astype(str)])

    if not var_is_prefixed:
        X, var_names = _prefix_by_cell_species(units.X, units.var_names, sp_per_cell)
        out = ad.AnnData(X=X)
        out.var_names = var_names
        out.obs = units.obs.copy()
        out.obsm[spatial_key] = np.asarray(units.obsm[spatial_key])
        units = out

    units.obs["species"] = pd.Categorical(sp_per_cell)
    units.obs["label"] = pd.Categorical(
        [f"{s}_{c}" for s, c in zip(sp_per_cell, adata.obs[celltype_key].astype(str))]
    )
    units.obs["spot"] = units.obs_names  # each cell is its own "spot"
    return units


def _prefix_by_cell_species(X, genes, sp_per_cell, keep_genes=None):
    """Split a cells x genes matrix into per-species prefixed blocks (sparse).

    A cell only expresses its own species' genes, so each species gets its own
    block of ``prefix:gene`` columns holding that species' cells (zeros elsewhere)
    -- human and rat symbols never collide. Stays sparse: Visium HD / Xenium
    sections are hundreds of thousands of cells. ``keep_genes`` (prefixed names)
    optionally drops columns the database can never use.
    """
    X = sp_sparse.csr_matrix(X)
    blocks, var_names = [], []
    for sp in sorted(set(sp_per_cell)):
        names = [add_prefix(str(g), sp) for g in genes]
        cols = [i for i, n in enumerate(names) if keep_genes is None or n in keep_genes]
        if not cols:
            continue
        row_mask = sp_sparse.diags((sp_per_cell == sp).astype(np.float64))
        blocks.append((row_mask @ X[:, cols]).tocsr())
        var_names += [names[i] for i in cols]
    if not blocks:
        raise ValueError("no genes left after prefixing; check species / gene symbols vs the database")
    out = sp_sparse.hstack(blocks).tocsr()
    out.eliminate_zeros()
    return out, var_names


def _microns_per_pixel(adata) -> float | None:
    """``microns_per_pixel`` from Space Ranger HD scalefactors in ``uns['spatial']``."""
    spatial = adata.uns.get("spatial", {})
    for lib in spatial.values():
        mpp = lib.get("scalefactors", {}).get("microns_per_pixel")
        if mpp:
            return float(mpp)
    return None


def from_bin2cell(
    cdata,
    celltype_key: str,
    species,
    *,
    spatial_key: str = "spatial",
    microns_per_pixel: float | None = None,
    min_bins: int = 1,
    labels_source: list[str] | None = None,
    layer: str | None = None,
    normalize: str | bool = "auto",
    target_sum: float = 1e4,
    keep_genes=None,
    carry_obs: list[str] | None = None,
) -> ad.AnnData:
    """Visium HD: segmented cells from bin2cell -> analysis units (one cell = one unit).

    ``cdata`` is the **cell-level** AnnData returned by ``b2c.bin_to_cell`` (not the
    2 µm bin object). Its format, which this function relies on:

    * ``X`` -- summed counts of the cell's 2 µm bins (sparse CSR). Raw integers, or
      non-integer if ``b2c.destripe`` was run on the bins (destriping rescales
      ``X``). bin2cell does **not** normalise.
    * ``obs_names`` -- the segmentation label ids as strings; ``obs`` has
      ``object_id``, ``bin_count`` (bins per cell), ``array_row``/``array_col`` and,
      with joint H&E+GEX labels, ``<labels_key>_source`` ('primary'/'secondary').
    * ``obsm['spatial']`` -- mean bin position in **full-resolution image pixels**;
      ``uns['spatial'][library]['scalefactors']['microns_per_pixel']`` (Space
      Ranger HD) converts it to µm.
    * no cell types -- annotate them yourself (label transfer, clustering, ...)
      into ``obs[celltype_key]`` before calling this.

    Parameters
    ----------
    celltype_key
        ``obs`` column with each cell's annotated cell type.
    species
        Which species each cell is: an ``obs`` column ('human'/'rat'/... per cell),
        a dict cell type -> species, or one species name for a single-species
        section (e.g. ``"human"``).
    microns_per_pixel
        Override the scale factor; otherwise read from ``uns['spatial']``.
        Coordinates are left in pixels (with a warning) if neither is available.
    min_bins
        Drop objects made of fewer 2 µm bins (segmentation debris).
    labels_source
        Keep only cells whose ``<labels_key>_source`` is in this list, e.g.
        ``["primary"]`` for H&E-segmented cells only. Needs that obs column.
    layer
        Use ``layers[layer]`` instead of ``X`` (e.g. a saved ``"counts"`` layer).
    normalize
        ``"auto"`` (default) total-count normalises to ``target_sum`` unless the
        data are already log-transformed (``uns['log1p']`` present, as scanpy
        marks it); ``True``/``False`` to force. CellPhoneDB expects normalised,
        unscaled expression.
    keep_genes
        Prefixed gene names to keep (e.g. the database's genes); cuts memory on a
        whole-transcriptome HD section.
    """
    cdata = cdata[cdata.obs["bin_count"] >= min_bins] if "bin_count" in cdata.obs else cdata
    if labels_source is not None:
        src = [c for c in cdata.obs.columns if c.endswith("_source")]
        if not src:
            raise KeyError("labels_source given but no '<labels_key>_source' column in obs "
                           "(only present when b2c.salvage_secondary_labels was used)")
        cdata = cdata[cdata.obs[src[0]].astype(str).isin(labels_source)]
    if celltype_key not in cdata.obs:
        raise KeyError(f"{celltype_key!r} not in obs -- annotate cell types on the bin2cell "
                       f"output first; available: {list(cdata.obs.columns)}")
    cdata = cdata[cdata.obs[celltype_key].notna()]

    cell_types = cdata.obs[celltype_key].astype(str).values
    if isinstance(species, dict):
        sp_per_cell = np.array([species[c] for c in cell_types])
    elif species in cdata.obs.columns:
        sp_per_cell = cdata.obs[species].astype(str).values
    else:
        from .species import SPECIES

        if species not in SPECIES:
            raise KeyError(f"species {species!r} is neither an obs column nor a known species")
        sp_per_cell = np.full(cdata.n_obs, species)

    X = cdata.layers[layer] if layer else cdata.X
    if normalize == "auto":
        normalize = "log1p" not in cdata.uns
    X = sp_sparse.csr_matrix(X, dtype=np.float64)
    if normalize:
        totals = np.asarray(X.sum(1)).ravel()
        X = sp_sparse.diags(target_sum / np.where(totals > 0, totals, 1.0)) @ X
    X, var_names = _prefix_by_cell_species(X, cdata.var_names, sp_per_cell, keep_genes=keep_genes)

    mpp = microns_per_pixel or _microns_per_pixel(cdata)
    coords = np.asarray(cdata.obsm[spatial_key], dtype=float)
    if mpp:
        coords = coords * mpp
    else:
        warnings.warn("no microns_per_pixel found; coordinates stay in pixels, so set radii in pixels")

    obs = pd.DataFrame(index=[str(i) for i in cdata.obs_names])
    for k in ["bin_count"] + list(carry_obs or []):
        if k in cdata.obs:
            obs[k] = cdata.obs[k].values
    obs["cell_type"] = pd.Categorical(cell_types)
    obs["species"] = pd.Categorical(sp_per_cell)
    obs["label"] = pd.Categorical([f"{s}_{c}" for s, c in zip(sp_per_cell, cell_types)])
    obs["spot"] = obs.index  # each cell is its own "spot"

    units = ad.AnnData(X=X, obs=obs)
    units.var_names = var_names
    units.obsm[spatial_key] = coords
    units.uns["spatial_units"] = "um" if mpp else "pixel"
    return units


def prepare_units(platform: str, **kwargs) -> ad.AnnData:
    """Dispatch to the right platform preparation. See module docstring."""
    if platform not in PLATFORMS:
        raise ValueError(f"unknown platform {platform!r}; known: {sorted(PLATFORMS)}")
    if platform == "visium_v1" or (platform == "visium_hd" and "abundance" in kwargs):
        return deconvolve_to_pseudocells(**kwargs)
    if platform == "visium_hd":
        return from_bin2cell(**kwargs)
    return cells_as_units(**kwargs)
