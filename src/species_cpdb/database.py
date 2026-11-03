"""Build a *cross-species* CellPhoneDB custom database.

Given the curated human CellPhoneDB interactome and a human<->rat ortholog map,
we emit a new CellPhoneDB database in which every gene/protein/complex is
species-prefixed (see :mod:`species_cpdb._namespace`) and every human
interaction ``A - B`` is expanded into up to four variants:

    hsa:A - hsa:B     within human      (native, confidence 1.0)
    rno:A - rno:B     within rat        (assumed conserved if orthologs exist)
    rno:A - hsa:B     cross-species     (rat ligand  -> human receptor)
    hsa:A - rno:B     cross-species     (human ligand -> rat receptor)

The cross-species variants are the whole point: they let CellPhoneDB test
whether, across a graft border, a rat cell's ligand could engage a human
cell's receptor. Because no ground-truth cross-species interactome exists, each
variant carries a **confidence** in ``0..1`` derived from ortholog quality /
sequence conservation of the *foreign* (substituted) partner. That confidence
is written to a sidecar (``interaction_meta.csv``) and used later for weighting
and annotating dotplots -- CellPhoneDB itself treats every row equally.

Assumptions / caveats (v1):
* Protein annotations (secreted / receptor / transmembrane / integrin) are
  assumed conserved between orthologs.
* A within-rat interaction is assumed valid whenever orthologs exist for both
  partners (conservation of the interaction is not independently verified).
* Confidence uses conservation of the substituted partner as a proxy for
  "will the native partner still recognise the foreign one" -- it is a
  prediction, not a measurement.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from importlib import resources

import numpy as np
import pandas as pd

from ._namespace import PREFIX, add_prefix
from .confidence import ConfidenceModel, get_confidence, identity_factor  # noqa: F401 (re-export)

# --- ortholog table -------------------------------------------------------


def load_orthologs(path: str | None = None, species: list[str] | None = None,
                   source: str = "seed") -> pd.DataFrame:
    """Load a **long-format** human->species ortholog table.

    Columns: ``species, human_symbol, target_symbol, orthology_type, pct_identity``.
    One row per (species, human gene): ``target_symbol`` is that gene's ortholog
    symbol in ``species`` and ``pct_identity`` its protein % identity to human.

    Parameters
    ----------
    path
        Explicit CSV to load (overrides ``source``).
    source
        ``"ensembl"`` -- the bundled **real** Ensembl table covering the whole
        CellPhoneDB interactome across every supported species (build once, pick
        your species per analysis). ``"seed"`` (default) -- the tiny *illustrative*
        rat/mouse/pig/rhesus seed used by the offline demo/tests.
    species
        Optionally filter to a subset of species.
    """
    if path is not None:
        df = pd.read_csv(path)
    else:
        fname = "orthologs_ensembl.csv" if source == "ensembl" else "ortholog_seed.csv"
        res = resources.files("species_cpdb.data").joinpath(fname)
        if source == "ensembl" and not res.is_file():
            raise FileNotFoundError(
                "bundled Ensembl orthologs not found; run "
                "`species-cpdb fetch-orthologs --cpdb-dir data/cpdb --species all` to build it.")
        with res.open() as fh:
            df = pd.read_csv(fh)
    needed = {"species", "human_symbol", "target_symbol", "orthology_type", "pct_identity"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"ortholog table missing columns: {missing}. Expected long format "
            "(species, human_symbol, target_symbol, orthology_type, pct_identity)."
        )
    if species is not None:
        df = df[df["species"].isin(species)].copy()
    return df


def ortholog_map(orthologs: pd.DataFrame) -> dict[str, dict]:
    """``{species: {human_symbol: record}}`` from a long ortholog table.

    Each record exposes ``.target_symbol``, ``.pct_identity``, ``.orthology_type``.
    """
    out: dict[str, dict] = {}
    for rec in orthologs.itertuples(index=False):
        out.setdefault(rec.species, {}).setdefault(rec.human_symbol, rec)
    return out


# --- reference (human) CellPhoneDB db -------------------------------------


@dataclass
class ReferenceDB:
    gene: pd.DataFrame
    protein: pd.DataFrame
    complex: pd.DataFrame
    interaction: pd.DataFrame
    uniprot_to_gene: dict
    complex_members: dict  # complex_name -> [uniprot, ...]

    @property
    def simple_uniprots(self) -> set:
        return set(self.uniprot_to_gene)


def load_reference_db(cpdb_dir: str) -> ReferenceDB:
    """Load the four ``*_input.csv`` files of a released CellPhoneDB database."""
    gene = pd.read_csv(os.path.join(cpdb_dir, "gene_input.csv"))
    protein = pd.read_csv(os.path.join(cpdb_dir, "protein_input.csv"))
    cplx = pd.read_csv(os.path.join(cpdb_dir, "complex_input.csv"))
    inter = pd.read_csv(os.path.join(cpdb_dir, "interaction_input.csv"))

    uniprot_to_gene = dict(zip(gene.uniprot, gene.gene_name))
    complex_members = {}
    for _, r in cplx.iterrows():
        members = [r[f"uniprot_{i}"] for i in range(1, 6) if pd.notna(r.get(f"uniprot_{i}"))]
        complex_members[r.complex_name] = members
    return ReferenceDB(gene, protein, cplx, inter, uniprot_to_gene, complex_members)


# --- namespacing helpers --------------------------------------------------


def _ns_uniprot(uniprot: str, species: str) -> str:
    return f"{PREFIX[species]}_{uniprot}"


def _ns_complex(complex_name: str, species: str) -> str:
    return add_prefix(complex_name, species)


class CrossSpeciesBuilder:
    """Expand the human CellPhoneDB db into an N-species db keyed by orthology.

    ``species`` is the list of species to include (must contain ``"human"``, the
    reference). For each human interaction ``A-B`` it emits every ``S1:A - S2:B``
    variant across the requested species (within- and cross-species), each with a
    confidence from the conservation of its non-human partners.
    """

    def __init__(self, ref: ReferenceDB, orthologs: pd.DataFrame, species: list[str] | None = None,
                 confidence: ConfidenceModel | None = None):
        self.ref = ref
        self.confidence = confidence or get_confidence("orthology")
        self.ortho_by_species = ortholog_map(orthologs)
        # species to build: human + whatever the ortholog table / request covers
        if species is None:
            species = ["human"] + sorted(self.ortho_by_species)
        if "human" not in species:
            species = ["human"] + list(species)
        self.species = list(dict.fromkeys(species))  # dedupe, keep order
        self._rows = {"gene": [], "protein": [], "complex": [], "interaction": []}
        self._seen = {"gene": set(), "protein": set(), "complex": set()}
        self._meta = []  # sidecar rows
        self._counter = 0

    # -- partner introspection --
    def _partner_genes(self, partner: str) -> list[str]:
        """Gene symbols backing a partner (uniprot or complex_name)."""
        if partner in self.ref.uniprot_to_gene:
            return [self.ref.uniprot_to_gene[partner]]
        if partner in self.ref.complex_members:
            return [self.ref.uniprot_to_gene.get(u) for u in self.ref.complex_members[partner]]
        return [None]

    def _partner_uniprots(self, partner: str) -> list[str]:
        if partner in self.ref.uniprot_to_gene:
            return [partner]
        if partner in self.ref.complex_members:
            return list(self.ref.complex_members[partner])
        return []

    def _has_orthologs(self, genes: list[str], species: str) -> bool:
        if None in genes:
            return False
        if species == "human":
            return True
        omap = self.ortho_by_species.get(species, {})
        return all(g in omap for g in genes)

    def _partner_confidence(self, partner: str, species: str) -> tuple[float, str]:
        """Confidence (+provenance) for a partner in ``species`` (human = 1.0)."""
        return self.confidence.partner_score(self._partner_genes(partner), species, self.ortho_by_species)

    # -- emit namespaced entities (deduped) --
    def _emit_gene(self, uniprot: str, species: str):
        key = (uniprot, species)
        if key in self._seen["gene"]:
            return
        grow = self.ref.gene[self.ref.gene.uniprot == uniprot]
        if grow.empty:
            return
        grow = grow.iloc[0]
        hsym = grow.gene_name
        if species == "human":
            sym = hsym
        else:
            rec = self.ortho_by_species.get(species, {}).get(hsym)
            if rec is None:
                return
            sym = rec.target_symbol
        self._rows["gene"].append(
            {
                "gene_name": add_prefix(sym, species),
                "uniprot": _ns_uniprot(uniprot, species),
                "hgnc_symbol": add_prefix(sym, species),
                "ensembl": f"{PREFIX[species]}_{grow.ensembl}",
            }
        )
        self._seen["gene"].add(key)
        self._emit_protein(uniprot, species)

    def _emit_protein(self, uniprot: str, species: str):
        key = (uniprot, species)
        if key in self._seen["protein"]:
            return
        prow = self.ref.protein[self.ref.protein.uniprot == uniprot]
        if prow.empty:
            return
        d = prow.iloc[0].to_dict()
        d["uniprot"] = _ns_uniprot(uniprot, species)
        d["protein_name"] = f"{PREFIX[species]}_{d['protein_name']}"
        self._rows["protein"].append(d)
        self._seen["protein"].add(key)

    def _emit_complex(self, complex_name: str, species: str):
        key = (complex_name, species)
        if key in self._seen["complex"]:
            return
        crow = self.ref.complex[self.ref.complex.complex_name == complex_name]
        if crow.empty:
            return
        d = crow.iloc[0].to_dict()
        d["complex_name"] = _ns_complex(complex_name, species)
        for i in range(1, 6):
            col = f"uniprot_{i}"
            if pd.notna(d.get(col)):
                self._emit_gene(d[col], species)
                d[col] = _ns_uniprot(d[col], species)
        self._rows["complex"].append(d)
        self._seen["complex"].add(key)

    def _ns_partner(self, partner: str, species: str) -> str:
        if partner in self.ref.complex_members:
            self._emit_complex(partner, species)
            return _ns_complex(partner, species)
        self._emit_gene(partner, species)
        return _ns_uniprot(partner, species)

    def _species_symbol(self, human_symbol: str, species: str) -> str:
        """Human symbol -> that species' ortholog symbol (matches the db gene names)."""
        if species == "human":
            return human_symbol
        rec = self.ortho_by_species.get(species, {}).get(human_symbol)
        return rec.target_symbol if rec else human_symbol

    def _partner_label(self, partner: str, species: str) -> str:
        # use the species' actual ortholog symbols so the label matches the real
        # prefixed gene names (e.g. rno:Tgfb1, not rno:TGFB1) -- the neighborhood
        # engine parses this label to look up expression.
        return "+".join(
            add_prefix(self._species_symbol(g, species), species) for g in self._partner_genes(partner)
        )

    def _add_interaction(self, row, sp_a: str, sp_b: str, confidence: float, cross: bool,
                         confidence_method: str = "native"):
        self._counter += 1
        cpid = f"CPI-XS-{self._counter:06d}"
        pa = self._ns_partner(row.partner_a, sp_a)
        pb = self._ns_partner(row.partner_b, sp_b)
        label = f"{self._partner_label(row.partner_a, sp_a)}->{self._partner_label(row.partner_b, sp_b)}"
        self._rows["interaction"].append(
            {
                "id_cp_interaction": cpid,
                "partner_a": pa,
                "partner_b": pb,
                "protein_name_a": pa,
                "protein_name_b": pb,
                "annotation_strategy": getattr(row, "annotation_strategy", "curated"),
                "source": f"species_cpdb:{getattr(row, 'source', 'cellphonedb')}",
                "interactors": label,
                "classification": getattr(row, "classification", np.nan),
                "directionality": getattr(row, "directionality", np.nan),
                "is_ppi": getattr(row, "is_ppi", True),
                "curator": f"species_cpdb:{getattr(row, 'curator', 'cellphonedb')}",
            }
        )
        self._meta.append(
            {
                "id_cp_interaction": cpid,
                "interactors": label,
                "partner_a_id": pa,
                "partner_b_id": pb,
                "species_a": sp_a,
                "species_b": sp_b,
                "cross_species": cross,
                "confidence": round(float(confidence), 4),
                "confidence_method": confidence_method,
                "base_interactors": row.interactors,
                "classification": getattr(row, "classification", np.nan),
                "directionality": getattr(row, "directionality", np.nan),
            }
        )

    def build(self, restrict_to_orthologs: bool = True, keep_human_only: bool = False):
        """Populate all rows. Returns self.

        For each human interaction ``A-B`` and every ordered species pair
        ``(S1, S2)`` among ``self.species``, emit ``S1:A - S2:B`` when the
        orthologs exist (human is always available). With two species this is the
        original 4 variants; with N species it is up to N^2.

        restrict_to_orthologs
            Only expand interactions where every non-human species in the set has
            orthologs for all partner genes (so the interaction is buildable in
            all of them). Keeps the output balanced across species.
        keep_human_only
            Also emit the within-human interaction for rows that fail the above
            (useful for real runs; off keeps the demo focused on cross-species).
        """
        non_human = [s for s in self.species if s != "human"]
        for row in self.ref.interaction.itertuples(index=False):
            genes_a = self._partner_genes(row.partner_a)
            genes_b = self._partner_genes(row.partner_b)
            if None in genes_a or None in genes_b:
                continue  # partner not resolvable to genes (skip odd rows)

            # which species can host each partner
            ok_a = {s: self._has_orthologs(genes_a, s) for s in self.species}
            ok_b = {s: self._has_orthologs(genes_b, s) for s in self.species}

            fully_covered = all(ok_a[s] and ok_b[s] for s in non_human)
            if restrict_to_orthologs and not fully_covered:
                if keep_human_only:
                    self._add_interaction(row, "human", "human", 1.0, cross=False, confidence_method="native")
                continue

            # cache per-(partner, species) confidence
            conf_a = {s: self._partner_confidence(row.partner_a, s) for s in self.species if ok_a[s]}
            conf_b = {s: self._partner_confidence(row.partner_b, s) for s in self.species if ok_b[s]}

            for s1 in self.species:
                if not ok_a[s1]:
                    continue
                for s2 in self.species:
                    if not ok_b[s2]:
                        continue
                    ca, na = conf_a[s1]
                    cb, nb = conf_b[s2]
                    conf = min(ca, cb)
                    cross = s1 != s2
                    if s1 == "human" and s2 == "human":
                        method = "native"
                    elif na == "native":
                        method = nb
                    elif nb == "native":
                        method = na
                    else:
                        method = f"{na}|{nb}"
                    self._add_interaction(row, s1, s2, conf, cross=cross, confidence_method=method)
        return self

    def write(self, out_dir: str, cpdb_dir: str):
        """Write the five ``*_input.csv`` files + sidecar, then build the .zip."""
        os.makedirs(out_dir, exist_ok=True)
        gene_df = pd.DataFrame(self._rows["gene"])[["gene_name", "uniprot", "hgnc_symbol", "ensembl"]]
        protein_df = pd.DataFrame(self._rows["protein"])[list(self.ref.protein.columns)]
        complex_df = pd.DataFrame(self._rows["complex"])[list(self.ref.complex.columns)]
        # CellPhoneDB's sanity checks merge complex uniprot_* against protein
        # uniprot. On a small subsetted db a member column (e.g. uniprot_5) can
        # be entirely empty; CellPhoneDB re-reads it from CSV as float64 and the
        # merge against string uniprots raises. A full-ortholog db never hits
        # this (some complex always has 5 subunits). Workaround: seed any fully
        # empty member column with an existing member of the first complex. This
        # is numerically inert -- complex expression is a min/mean over members,
        # so a duplicated subunit changes nothing -- and it forces object dtype.
        for i in range(2, 6):
            col = f"uniprot_{i}"
            complex_df[col] = complex_df[col].astype(object)
            if len(complex_df) and complex_df[col].isna().all():
                complex_df.loc[complex_df.index[0], col] = complex_df.loc[
                    complex_df.index[0], "uniprot_1"
                ]
        inter_df = pd.DataFrame(self._rows["interaction"])
        meta_df = pd.DataFrame(self._meta)

        gene_df.to_csv(os.path.join(out_dir, "gene_input.csv"), index=False)
        protein_df.to_csv(os.path.join(out_dir, "protein_input.csv"), index=False)
        complex_df.to_csv(os.path.join(out_dir, "complex_input.csv"), index=False)
        inter_df.to_csv(os.path.join(out_dir, "interaction_input.csv"), index=False)
        meta_df.to_csv(os.path.join(out_dir, "interaction_meta.csv"), index=False)

        # CellPhoneDB requires a transcription_factor_input.csv with a specific
        # schema, and a sources/uniprot_synonyms.tsv. We don't model receptor->TF
        # signalling here, so emit an empty TF file with the correct columns.
        tf_cols = [
            "id_cp_interaction", "partner_receptor", "partner_TF",
            "protein_name_receptor", "protein_name_TF", "receptor_id",
            "TF_symbol", "Effect", "Source", "Curator", "Xrefs_or_figures",
            "Brivanlou_class",
        ]
        pd.DataFrame(columns=tf_cols).to_csv(
            os.path.join(out_dir, "transcription_factor_input.csv"), index=False
        )
        os.makedirs(os.path.join(out_dir, "sources"), exist_ok=True)
        src_syn = os.path.join(cpdb_dir, "sources", "uniprot_synonyms.tsv")
        if os.path.exists(src_syn):
            shutil.copy(src_syn, os.path.join(out_dir, "sources", "uniprot_synonyms.tsv"))
        else:
            pd.DataFrame(columns=["uniprot", "synonym"]).to_csv(
                os.path.join(out_dir, "sources", "uniprot_synonyms.tsv"), sep="\t", index=False
            )
        return {
            "genes": len(gene_df),
            "proteins": len(protein_df),
            "complexes": len(complex_df),
            "interactions": len(inter_df),
            "cross_species_interactions": int(meta_df.cross_species.sum()) if len(meta_df) else 0,
        }


def build_cross_species_db(
    cpdb_dir: str,
    out_dir: str,
    orthologs: pd.DataFrame | None = None,
    species: list[str] = ("human", "rat"),
    restrict_to_orthologs: bool = True,
    keep_human_only: bool = False,
    make_zip: bool = True,
    confidence: str | ConfidenceModel = "orthology",
    interface_table: pd.DataFrame | None = None,
    orthologs_source: str = "seed",
) -> dict:
    """End-to-end: load reference + orthologs, expand, write, and zip the db.

    Parameters
    ----------
    confidence
        Which cross-species confidence model to use: ``"orthology"`` (default),
        ``"interface"`` (binding-residue conservation, falls back to orthology),
        or a :class:`species_cpdb.confidence.ConfidenceModel` instance.
    interface_table
        Optional interface-conservation table for ``confidence="interface"``
        (defaults to the bundled illustrative seed).

    Returns a summary dict; the resulting ``<out_dir>/cellphonedb.zip`` is the
    file to pass to the statistical method.
    """
    from cellphonedb.utils import db_utils

    species = list(dict.fromkeys(["human"] + list(species)))  # ensure human, dedupe
    ref = load_reference_db(cpdb_dir)
    if orthologs is None:
        orthologs = load_orthologs(species=[s for s in species if s != "human"], source=orthologs_source)
    model = confidence if isinstance(confidence, ConfidenceModel) else get_confidence(confidence, interface_table)
    builder = CrossSpeciesBuilder(ref, orthologs, species=species, confidence=model).build(
        restrict_to_orthologs=restrict_to_orthologs, keep_human_only=keep_human_only
    )
    summary = builder.write(out_dir, cpdb_dir)
    summary["confidence"] = model.name
    summary["species"] = species
    if make_zip:
        db_utils.create_db(out_dir)
        # create_db writes a timestamped cellphonedb_<date>.zip; pick the newest
        # and also expose a stable cellphonedb.zip symlink/copy for convenience.
        import glob

        zips = sorted(glob.glob(os.path.join(out_dir, "cellphonedb_*.zip")), key=os.path.getmtime)
        if zips:
            stable = os.path.join(out_dir, "cellphonedb.zip")
            shutil.copy(zips[-1], stable)
            summary["db_zip"] = stable
            summary["db_zip_timestamped"] = zips[-1]
    return summary
