"""Run the real CellPhoneDB statistical method on cross-species spatial data."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CpdbResult:
    """Holds the CellPhoneDB output tables plus our cross-species annotation."""

    means: pd.DataFrame
    pvalues: pd.DataFrame
    significant_means: pd.DataFrame
    deconvoluted: pd.DataFrame
    meta: pd.DataFrame  # interaction_meta (cross-species confidence etc.)
    raw: dict
    units: object = None  # the analysis-units AnnData (for ligand-class proximity)

    def annotate(self) -> pd.DataFrame:
        """Interaction-level table with species/direction/confidence columns.

        CellPhoneDB regenerates ``id_cp_interaction`` on database build, so we
        recover the species of each partner from the ``partner_a``/``partner_b``
        prefixes in its own output (robust to the id churn) and join our
        confidence via the namespaced partner ids stored in the meta sidecar.
        """
        pv = self.pvalues.copy()
        pv["species_a"] = pv["partner_a"].map(_partner_species)
        pv["species_b"] = pv["partner_b"].map(_partner_species)
        pv["cross_species"] = (
            pv["species_a"].notna() & pv["species_b"].notna()
            & (pv["species_a"] != pv["species_b"])
        )
        pv["direction"] = np.where(
            pv["cross_species"],
            pv["species_a"].astype(str) + "->" + pv["species_b"].astype(str),
            "within-" + pv["species_a"].astype(str),
        )
        # join confidence via namespaced partner ids
        key = pv["partner_a"].map(_strip_kind) + "|" + pv["partner_b"].map(_strip_kind)
        conf_map = {
            f"{a}|{b}": c
            for a, b, c in zip(self.meta["partner_a_id"], self.meta["partner_b_id"], self.meta["confidence"])
        }
        pv["confidence"] = key.map(conf_map).fillna(1.0)
        # member genes ('ssc:TGFB1->hsa:TGFBR1+hsa:TGFBR2'): complexes are named in
        # CellPhoneDB's output, but folding / sequence lookup needs the subunits
        if "interactors" in self.meta.columns:
            members = {f"{a}|{b}": m for a, b, m in zip(
                self.meta["partner_a_id"], self.meta["partner_b_id"], self.meta["interactors"])}
            pv["interactors"] = key.map(members)
        return pv

    def cross_species_hits(
        self,
        pvalue: float = 0.05,
        cross_only: bool = True,
        ligand_class_proximity: bool = False,
        radius_secreted: float = 2.0,
        radius_contact: float = 1.0,
        proximity_min_frac: float = 0.1,
        label_key: str = "label",
        separator: str = "|",
    ) -> pd.DataFrame:
        """Long-format significant interactions, annotated by species/direction.

        Columns include interacting_pair, cell_pair, pvalue, mean, cross_species,
        direction ('rat->human'/'human->rat'/'within-*'), confidence and a
        ``weighted_score`` (-log10 p x confidence) for ranking.

        ligand_class_proximity
            Apply *ligand-class-aware* proximity: secreted ligands may act at
            ``radius_secreted`` (diffusion) but membrane/juxtacrine ligands only
            where the two cell types are within ``radius_contact`` (contact).
            Requires ``self.units`` (set by :func:`run_analysis`).
        """
        pv = self.annotate()
        mn = self.means
        extra = ["species_a", "species_b", "cross_species", "direction", "confidence", "interactors"]
        id_cols = [c for c in pv.columns if c in _META_COLS or c in extra]
        pair_cols = [c for c in pv.columns if c not in id_cols]

        long = pv.melt(id_vars=id_cols, value_vars=pair_cols,
                       var_name="cell_pair", value_name="pvalue")
        mlong = mn.melt(id_vars=[c for c in mn.columns if c in _META_COLS], value_vars=pair_cols,
                        var_name="cell_pair", value_name="mean")
        long = long.merge(mlong[["interacting_pair", "cell_pair", "mean"]],
                          on=["interacting_pair", "cell_pair"], how="left")
        long = long[long["pvalue"] <= pvalue].copy()
        if cross_only:
            long = long[long["cross_species"]].copy()
        if ligand_class_proximity:
            long = self._apply_ligand_class_proximity(
                long, radius_secreted, radius_contact, proximity_min_frac, label_key, separator)
        long["weighted_score"] = -np.log10(long["pvalue"].clip(lower=1e-4)) * long["confidence"]
        return long.sort_values("weighted_score", ascending=False).reset_index(drop=True)

    def _apply_ligand_class_proximity(self, long, radius_secreted, radius_contact, min_frac, label_key, separator):
        """Keep secreted hits within diffusion range; juxtacrine only at contact."""
        if self.units is None:
            raise ValueError("ligand_class_proximity needs self.units (use run_analysis/analyze_spatial)")
        from .spatial import celltype_proximity

        diffuse = celltype_proximity(self.units, label_key, radius_secreted, min_frac=min_frac)
        contact = celltype_proximity(self.units, label_key, radius_contact, min_frac=min_frac)
        if "secreted" not in long.columns:
            return long
        pairs = long["cell_pair"].str.split(separator, n=1, expand=True)
        pair_sets = [frozenset((a, b)) for a, b in zip(pairs[0], pairs[1])]
        secreted = long["secreted"].fillna(True).astype(bool).values
        keep = [
            (ps in diffuse) if sec else (ps in contact)
            for ps, sec in zip(pair_sets, secreted)
        ]
        return long[keep].copy()


def analyze_spatial(
    units,
    db_zip: str,
    out_dir: str,
    *,
    scope: str = "global",
    region=None,
    region_key: str = "region",
    window_center=None,
    window_radius: float | None = None,
    use_microenvs: bool = True,
    engine: str = "celltype",
    radius: float = 1.5,
    n_niches: int = 6,
    label_key: str = "label",
    iterations: int = 1000,
    threads: int = 4,
    pvalue: float = 0.05,
    seed: int = 0,
):
    """Compose scope restriction + the chosen spatial engine.

    ``units`` is a prepared AnnData (see :func:`species_cpdb.preprocess.prepare_units`).

    scope
        * ``'global'``  -- all units (means over the whole tissue).
        * ``'region'``  -- restrict to ``region`` via ``obs[region_key]`` (your
          spatial annotations, e.g. 'graft_border'); means become region-local.
        * ``'window'``  -- restrict to a disk of ``window_radius`` around
          ``window_center`` (coords or an obs_name).
    engine
        * ``'celltype'``     -- wrap real CellPhoneDB (cell-type means, spatial
          microenvironments gate which pairs are tested). Returns a ``CpdbResult``.
        * ``'neighborhood'`` -- native per-spot adjacency test (ligand-high next to
          receptor-high), via :mod:`species_cpdb.neighborhood`. Returns a
          ``NeighborhoodResult``. ``radius`` is the adjacency distance here.
    use_microenvs
        (celltype engine) Gate tested cell-type pairs to spatial niches.
    """
    import os as _os
    from .spatial import build_microenvironments, restrict_to_region, restrict_to_window

    if scope == "region":
        if region is None:
            raise ValueError("scope='region' needs `region`")
        units = restrict_to_region(units, region, region_key=region_key)
    elif scope == "window":
        if window_center is None or window_radius is None:
            raise ValueError("scope='window' needs `window_center` and `window_radius`")
        units = restrict_to_window(units, window_center, window_radius)
    elif scope != "global":
        raise ValueError(f"unknown scope {scope!r}")

    if engine == "neighborhood":
        from .neighborhood import neighborhood_ligrec

        meta = pd.read_csv(_os.path.join(_os.path.dirname(db_zip), "interaction_meta.csv"))
        return neighborhood_ligrec(units, meta, radius=radius, iterations=iterations,
                                   label_key=label_key, seed=seed)
    if engine != "celltype":
        raise ValueError(f"unknown engine {engine!r}")

    me = build_microenvironments(units, label_key=label_key, radius=radius, n_niches=n_niches) \
        if use_microenvs else None
    return run_analysis(
        units, db_zip=db_zip, out_dir=out_dir, label_key=label_key, microenvs=me,
        iterations=iterations, threads=threads, pvalue=pvalue, seed=seed,
    )


def _strip_kind(partner: str) -> str:
    """'simple:hsa_Q15389' -> 'hsa_Q15389'; 'complex:hsa:Foo' -> 'hsa:Foo'."""
    if isinstance(partner, str) and ":" in partner:
        return partner.split(":", 1)[1]
    return partner


def _partner_species(partner: str):
    """Species of a CellPhoneDB output partner string, from its prefix token."""
    from ._namespace import PREFIX

    inv = {v: k for k, v in PREFIX.items()}
    core = _strip_kind(partner)
    if not isinstance(core, str):
        return None
    # species token runs up to the first '_' or ':'
    token = core.replace(":", "_").split("_", 1)[0]
    return inv.get(token)


# Columns in CellPhoneDB means/pvalues that are interaction metadata, not cell pairs
_META_COLS = {
    "id_cp_interaction", "interacting_pair", "partner_a", "partner_b",
    "gene_a", "gene_b", "secreted", "receptor_a", "receptor_b", "annotation_strategy",
    "is_integrin", "directionality", "classification", "is_ppi",
}


def run_analysis(
    adata,
    db_zip: str,
    out_dir: str,
    label_key: str = "label",
    microenvs: pd.DataFrame | None = None,
    interaction_meta: str | pd.DataFrame | None = None,
    counts_data: str = "gene_name",
    iterations: int = 1000,
    threshold: float = 0.1,
    pvalue: float = 0.05,
    threads: int = 4,
    seed: int = 0,
) -> CpdbResult:
    """Write inputs and call ``cpdb_statistical_analysis_method.call``.

    Parameters
    ----------
    adata
        Normalised (unscaled) expression with ``.obs[label_key]`` and a
        two-species prefixed ``var_names`` matching the database.
    db_zip
        Path to the cross-species ``cellphonedb.zip`` from
        :func:`species_cpdb.database.build_cross_species_db`.
    microenvs
        Microenvironments table from :func:`species_cpdb.spatial.build_microenvironments`.
        If None, all cell-type pairs are tested (non-spatial behaviour).
    interaction_meta
        The ``interaction_meta.csv`` produced alongside the db, used to annotate
        which hits are cross-species. Path or DataFrame.
    """
    from cellphonedb.src.core.methods import cpdb_statistical_analysis_method

    os.makedirs(out_dir, exist_ok=True)
    counts_fp = os.path.join(out_dir, "counts.h5ad")
    meta_fp = os.path.join(out_dir, "meta.tsv")

    adata.write_h5ad(counts_fp)
    meta_df = pd.DataFrame({"Cell": adata.obs_names, "cell_type": adata.obs[label_key].astype(str).values})
    meta_df.to_csv(meta_fp, sep="\t", index=False)

    microenvs_fp = None
    if microenvs is not None:
        microenvs_fp = os.path.join(out_dir, "microenvs.tsv")
        microenvs.to_csv(microenvs_fp, sep="\t", index=False)

    raw = cpdb_statistical_analysis_method.call(
        cpdb_file_path=db_zip,
        meta_file_path=meta_fp,
        counts_file_path=counts_fp,
        counts_data=counts_data,
        microenvs_file_path=microenvs_fp,
        output_path=out_dir,
        iterations=iterations,
        threshold=threshold,
        pvalue=pvalue,
        threads=threads,
        debug_seed=seed,
        output_suffix="xspecies",
    )

    if interaction_meta is None:
        interaction_meta = os.path.join(os.path.dirname(db_zip), "interaction_meta.csv")
    meta = (pd.read_csv(interaction_meta) if isinstance(interaction_meta, str) else interaction_meta)

    return CpdbResult(
        means=raw["means"],
        pvalues=raw["pvalues"],
        significant_means=raw.get("significant_means"),
        deconvoluted=raw.get("deconvoluted"),
        meta=meta,
        raw=raw,
        units=adata,
    )
