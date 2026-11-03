"""Structural viability filtering of cross-species hits (AlphaFold-Multimer / AF3).

The third evidence axis -- **physical bindability**. Expression (CellPhoneDB) and
proximity (spatial) say two cells *could* talk; folding the actual cross-species
complex (pig ligand + human receptor) says whether they can plausibly *bind*.

Run as a downstream filter on the *shortlist*: fold the handful of interactions
that survived expression + proximity, never the whole database.

What AlphaFold gives you (per folded complex):
* ``iptm``          -- interface predicted TM-score (0..1); AF's confidence in the
                       relative chain placement. The primary interface-quality metric.
* ``interface_pae`` -- predicted aligned error across the interface (Angstrom); lower = better.
* ``pdockq``        -- pLDDT+contacts interface score (0..1).
* ``bsa``           -- buried surface area at the interface (Angstrom^2); interface size.

Important: AlphaFold predicts *structure*, not *affinity*. ipTM/PAE are structural
confidence, not thermodynamics. For a real binding-affinity cutoff, estimate dG
from the folded model with PRODIGY / FoldX and filter on ``delta_g`` (kcal/mol,
more negative = tighter). This module supports both: an interface-confidence
cutoff (ipTM/PAE/pDockQ) and, when available, an affinity cutoff (``delta_g``).

Two ways to get the ``scores``:
* let the tool run it -- :func:`run_structural_filter` folds each shortlisted
  complex through a pluggable backend (ColabFold / Boltz), optionally runs PRODIGY
  for dG, and applies your cutoffs, in one call. Like cell2location, the backend
  uses your GPU if available (``device="auto"|"gpu"|"cpu"``) and is an *opt-in*
  dependency -- only needed when you call this function.
* or produce them yourself and pass ``scores`` to :func:`filter_hits_structurally`.
  :func:`write_fold_inputs` writes the ready-to-fold multimer FASTAs.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Literature-informed defaults. ipTM ~0.5 is the common "plausible complex" line;
# interface PAE < ~10-15 A indicates a confident interface; pDockQ > 0.23 is the
# widely used acceptable-model threshold. delta_g <= -6 kcal/mol ~ a real,
# non-weak interaction (PRODIGY-scale). Tune per your validation.
DEFAULT_THRESHOLDS = {
    "min_iptm": 0.50,
    "max_interface_pae": 15.0,
    "min_pdockq": 0.23,
    "min_bsa": None,        # optional; e.g. 400 A^2
    "max_delta_g": None,    # set e.g. -6.0 to require an affinity floor (needs PRODIGY/FoldX)
}

_SCORE_COLS = ["iptm", "ptm", "interface_pae", "pdockq", "bsa", "delta_g"]


def filter_hits_structurally(
    hits: pd.DataFrame,
    scores: pd.DataFrame | None = None,
    predictor=None,
    *,
    key: str = "interacting_pair",
    min_iptm: float | None = DEFAULT_THRESHOLDS["min_iptm"],
    max_interface_pae: float | None = DEFAULT_THRESHOLDS["max_interface_pae"],
    min_pdockq: float | None = DEFAULT_THRESHOLDS["min_pdockq"],
    min_bsa: float | None = DEFAULT_THRESHOLDS["min_bsa"],
    max_delta_g: float | None = DEFAULT_THRESHOLDS["max_delta_g"],
    keep_unscored: bool = False,
) -> pd.DataFrame:
    """Filter/annotate cross-species ``hits`` by structural viability.

    Applies whichever thresholds are set *and* whose column is present in
    ``scores``. Interface-confidence cutoffs: ``min_iptm``, ``max_interface_pae``,
    ``min_pdockq``, ``min_bsa``. Affinity cutoff: ``max_delta_g`` (keep complexes
    at least this tight, e.g. ``-6.0``; requires a ``delta_g`` column you produced
    with PRODIGY/FoldX on the folded model).

    Parameters
    ----------
    scores
        One row per interaction: column ``key`` plus any of
        ``iptm, ptm, interface_pae, pdockq, bsa, delta_g``. Produce by folding the
        shortlisted complexes (see :func:`write_fold_inputs`).
    predictor
        On-the-fly folding hook -- **not implemented** (needs GPU + AF); raises.
    keep_unscored
        Keep interactions absent from ``scores`` (only if True).
    """
    if predictor is not None:
        raise NotImplementedError(
            "on-the-fly AlphaFold prediction is a stub (needs a GPU + AF databases); "
            "fold the shortlist externally and pass a precomputed `scores` table. "
            "Use write_fold_inputs() to generate the multimer FASTAs."
        )
    if scores is None:
        raise NotImplementedError(
            "structural filtering needs a precomputed `scores` table. Fold the "
            "cross-species complexes for the shortlisted hits (AlphaFold-Multimer / "
            "AF3) and pass columns: interacting_pair, iptm[, interface_pae, pdockq, "
            "bsa, delta_g]. write_fold_inputs() prepares the FASTAs."
        )
    if key not in scores.columns:
        raise ValueError(f"`scores` must contain a {key!r} column")

    merged = hits.merge(scores, on=key, how="left")
    present = [c for c in _SCORE_COLS if c in merged.columns]
    scored = merged[present].notna().any(axis=1) if present else pd.Series(False, index=merged.index)

    keep = pd.Series(True, index=merged.index)
    checks = [
        (min_iptm, "iptm", "ge"),
        (max_interface_pae, "interface_pae", "le"),
        (min_pdockq, "pdockq", "ge"),
        (min_bsa, "bsa", "ge"),
        (max_delta_g, "delta_g", "le"),
    ]
    for thresh, col, op in checks:
        if thresh is not None and col in merged.columns:
            passes = merged[col] >= thresh if op == "ge" else merged[col] <= thresh
            # a missing metric is ignored (not a failure); only a present value that
            # violates the threshold drops the row.
            keep &= passes | merged[col].isna()

    keep = keep.where(scored, other=keep_unscored)
    n_drop = int((~keep).sum())
    if n_drop:
        warnings.warn(f"structural filter removed {n_drop} of {len(merged)} hit rows")
    return merged[keep].reset_index(drop=True)


def example_scores_template(hits: pd.DataFrame, key: str = "interacting_pair") -> pd.DataFrame:
    """Blank scores template (unique interactions) to fill from your folding run."""
    pairs = pd.Index(hits[key].unique(), name=key)
    out = pd.DataFrame({key: pairs})
    for c in _SCORE_COLS:
        out[c] = np.nan
    return out


def _interaction_partners(interacting_pair: str):
    """'ssc:TGFB1_hsa:TGFBR1+hsa:TGFBR2' -> (['ssc:TGFB1'], ['hsa:TGFBR1','hsa:TGFBR2'])."""
    # split on the first '_' that separates the two partners (each partner may be a
    # '+'-joined complex); the interacting_pair uses '_' between partner_a and partner_b.
    if "->" in interacting_pair:
        a, b = interacting_pair.split("->", 1)
    else:
        # our neighborhood/celltype labels join partners with '_'; genes carry ':'
        # so split on the '_' that is not inside a prefixed token
        a, b = interacting_pair.split("_", 1)
    return a.split("+"), b.split("+")


def _partners_by_pair(df: pd.DataFrame, key: str) -> dict:
    """``{pair: (ligand_genes, receptor_genes)}`` for each unique ``df[key]``.

    Uses the member-gene ``interactors`` column when present (celltype-engine hits
    name complexes, e.g. ``hsa:TGFbeta_receptor1``, which can't be folded as-is);
    otherwise parses ``key`` itself (neighborhood-engine pairs are member genes).
    """
    out = {}
    has_members = "interactors" in df.columns
    for pair, members in zip(df[key], df["interactors"] if has_members else df[key]):
        if pair in out:
            continue
        src = members if (has_members and isinstance(members, str)) else pair
        out[pair] = _interaction_partners(src)
    return out


def _resolve_sequences(partners: dict, sequences, sequence_cache_dir):
    """Use the given ``sequences``, or fetch every chain from UniProt (cached)."""
    if sequences is not None:
        return sequences
    from .sequences import fetch_sequences

    genes = [g for lig, rec in partners.values() for g in list(lig) + list(rec)]
    return fetch_sequences(genes, cache_dir=sequence_cache_dir)


def write_fold_inputs(
    hits: pd.DataFrame,
    sequences: dict | None = None,
    out_dir: str = "fold_inputs",
    key: str = "interacting_pair",
    cross_only: bool = True,
    sequence_cache_dir: str | None = None,
) -> pd.DataFrame:
    """Write one multimer FASTA per unique (cross-species) interaction, ready to fold.

    ``sequences`` maps a prefixed gene id (e.g. ``'ssc:TGFB1'``, ``'hsa:TGFBR1'``)
    to its amino-acid sequence. Leave it ``None`` to fetch each species' sequence
    from UniProt automatically (cached locally; see :mod:`species_cpdb.sequences`).
    Each FASTA has one chain per ligand/receptor subunit -- exactly the complex to
    fold in AlphaFold-Multimer/AF3.

    Returns a manifest DataFrame (``interacting_pair``, ``fasta``, ``n_chains``,
    ``missing_sequences``) so you know which pairs are ready and which lack a sequence.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = hits
    if cross_only and "cross_species" in df.columns:
        df = df[df["cross_species"]]
    partners = _partners_by_pair(df, key)
    sequences = _resolve_sequences(partners, sequences, sequence_cache_dir)
    manifest = []
    for pair, (lig, rec) in partners.items():
        chains = list(lig) + list(rec)
        missing = [c for c in chains if c not in sequences]
        safe = pair.replace(":", "-").replace("+", "_").replace("->", "_to_")
        fasta = os.path.join(out_dir, f"{safe}.fasta")
        if not missing:
            with open(fasta, "w") as fh:
                for c in chains:
                    fh.write(f">{c}\n{sequences[c]}\n")
        manifest.append({
            key: pair, "fasta": fasta if not missing else None,
            "n_chains": len(chains), "missing_sequences": ",".join(missing),
        })
    return pd.DataFrame(manifest)


# ==========================================================================
# Running the folding + affinity prediction FROM the tool (opt-in backends).
# ==========================================================================

@dataclass
class FoldResult:
    """Metrics + model path from folding one complex."""
    interacting_pair: str
    model_path: str | None = None
    iptm: float | None = None
    ptm: float | None = None
    interface_pae: float | None = None
    pdockq: float | None = None
    extra: dict = field(default_factory=dict)


class FoldingBackend:
    """Interface for a structure-prediction backend.

    ``fold`` takes named chains ``[(chain_id, sequence), ...]`` (the ligand and
    receptor subunits of one cross-species complex) and returns a
    :class:`FoldResult`. ``device`` is ``'auto' | 'gpu' | 'cpu'`` and behaves like
    cell2location's accelerator arg -- the backend uses a GPU if available.
    """

    name = "base"

    def fold(self, name: str, chains: list, out_dir: str, device: str = "auto") -> FoldResult:
        raise NotImplementedError


class MockFoldingBackend(FoldingBackend):
    """Deterministic fake backend for testing / dry runs (no GPU, no deps).

    Returns canned metrics; ``scores_by_pair`` maps interacting_pair -> dict of
    metric values, else ``default`` is used.
    """

    name = "mock"

    def __init__(self, scores_by_pair: dict | None = None, default: dict | None = None):
        self.scores_by_pair = scores_by_pair or {}
        self.default = default or {"iptm": 0.7, "ptm": 0.75, "interface_pae": 8.0, "pdockq": 0.4}

    def fold(self, name, chains, out_dir, device="auto"):
        s = dict(self.default)
        s.update(self.scores_by_pair.get(name, {}))
        return FoldResult(interacting_pair=name, model_path=None, **{
            k: s.get(k) for k in ("iptm", "ptm", "interface_pae", "pdockq")})


def _write_multimer_fasta(name: str, chains: list, out_dir: str, colabfold_style: bool) -> str:
    safe = name.replace(":", "-").replace("+", "_").replace("->", "_to_")
    path = os.path.join(out_dir, f"{safe}.fasta")
    with open(path, "w") as fh:
        if colabfold_style:
            # ColabFold: one record, chains ':'-separated
            fh.write(f">{safe}\n{':'.join(seq for _, seq in chains)}\n")
        else:
            for cid, seq in chains:
                fh.write(f">{cid}\n{seq}\n")
    return path


def _interface_pae_from_matrix(pae, chain_lengths) -> float:
    """Mean PAE over inter-chain residue blocks (the interface)."""
    import numpy as np

    pae = np.asarray(pae)
    bounds, start = [], 0
    for L in chain_lengths:
        bounds.append((start, start + L)); start += L
    vals = []
    for a in range(len(bounds)):
        for b in range(len(bounds)):
            if a == b:
                continue
            (i0, i1), (j0, j1) = bounds[a], bounds[b]
            vals.append(pae[i0:i1, j0:j1])
    return float(np.concatenate([v.ravel() for v in vals]).mean()) if vals else float("nan")


class ColabFoldBackend(FoldingBackend):
    """Fold via a local ColabFold install (``colabfold_batch``).

    Install separately (localcolabfold); GPU strongly recommended but CPU works.
    Best-effort output parsing (ColabFold versions vary) -- adjust ``scores_glob``/
    ``model_glob`` if your version names files differently.
    """

    name = "colabfold"

    def __init__(self, executable: str = "colabfold_batch", num_models: int = 1,
                 extra_args: tuple = (), scores_glob: str = "*scores_rank_001*.json",
                 model_glob: str = "*rank_001*.pdb"):
        self.executable = executable
        self.num_models = num_models
        self.extra_args = list(extra_args)
        self.scores_glob = scores_glob
        self.model_glob = model_glob

    def fold(self, name, chains, out_dir, device="auto"):
        if shutil.which(self.executable) is None:
            raise RuntimeError(
                f"{self.executable!r} not found. Install localcolabfold, or use a "
                "different backend / precomputed scores.")
        os.makedirs(out_dir, exist_ok=True)
        fasta = _write_multimer_fasta(name, chains, out_dir, colabfold_style=True)
        sub = os.path.join(out_dir, "colabfold_" + os.path.splitext(os.path.basename(fasta))[0])
        env = dict(os.environ)
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["JAX_PLATFORM_NAME"] = "cpu"
        cmd = [self.executable, "--num-models", str(self.num_models), *self.extra_args, fasta, sub]
        subprocess.run(cmd, check=True, env=env)

        res = FoldResult(interacting_pair=name)
        scores = sorted(glob.glob(os.path.join(sub, self.scores_glob)))
        if scores:
            with open(scores[0]) as fh:
                d = json.load(fh)
            res.iptm = d.get("iptm")
            res.ptm = d.get("ptm")
            if "pae" in d:
                res.interface_pae = _interface_pae_from_matrix(d["pae"], [len(s) for _, s in chains])
        models = sorted(glob.glob(os.path.join(sub, self.model_glob)))
        res.model_path = models[0] if models else None
        return res


class CommandFoldingBackend(FoldingBackend):
    """Generic backend: run any folding CLI via a user-supplied command builder + parser.

    ``command(fasta, out_dir, device) -> list[str]`` and
    ``parse(out_dir, chains) -> FoldResult``. Lets you wire Boltz, Chai, AF3, etc.
    without changing the tool.
    """

    name = "command"

    def __init__(self, command, parse, colabfold_style_fasta: bool = False):
        self._command = command
        self._parse = parse
        self.colabfold_style = colabfold_style_fasta

    def fold(self, name, chains, out_dir, device="auto"):
        os.makedirs(out_dir, exist_ok=True)
        fasta = _write_multimer_fasta(name, chains, out_dir, self.colabfold_style)
        subprocess.run(self._command(fasta, out_dir, device), check=True)
        r = self._parse(out_dir, chains)
        r.interacting_pair = name
        return r


def predict_affinity_prodigy(pdb_path: str, temperature: float = 25.0,
                             executable: str = "prodigy") -> float:
    """Predicted binding free energy dG (kcal/mol) via PRODIGY (CPU, fast).

    Tries the ``prodigy_prot`` Python package, then the ``prodigy`` CLI. Install
    with ``pip install prodigy-prot``.
    """
    try:
        from prodigy_prot.predict_IC import Prodigy  # type: ignore
        from Bio.PDB import PDBParser

        structure = PDBParser(QUIET=True).get_structure("m", pdb_path)
        prod = Prodigy(structure, temp=temperature)
        prod.predict()
        return float(prod.ba_val)
    except Exception:  # noqa: BLE001 -- fall back to CLI
        pass
    if shutil.which(executable) is None:
        raise RuntimeError("PRODIGY not available (pip install prodigy-prot).")
    out = subprocess.run([executable, pdb_path, "--temperature", str(temperature)],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if "Predicted binding affinity" in line:
            return float(line.split(":")[-1].strip().split()[0])
    raise RuntimeError("could not parse PRODIGY output")


def run_structural_filter(
    hits: pd.DataFrame,
    sequences: dict | None = None,
    out_dir: str = "structure",
    backend: FoldingBackend | None = None,
    *,
    sequence_cache_dir: str | None = None,
    device: str = "auto",
    run_prodigy: bool = True,
    prodigy_temperature: float = 25.0,
    key: str = "interacting_pair",
    cross_only: bool = True,
    min_iptm: float | None = DEFAULT_THRESHOLDS["min_iptm"],
    max_interface_pae: float | None = DEFAULT_THRESHOLDS["max_interface_pae"],
    min_pdockq: float | None = DEFAULT_THRESHOLDS["min_pdockq"],
    max_delta_g: float | None = DEFAULT_THRESHOLDS["max_delta_g"],
    keep_unscored: bool = False,
):
    """Fold each shortlisted complex, (optionally) run PRODIGY, apply your cutoffs.

    This is the "the tool runs AlphaFold" entry point. For every unique
    cross-species interaction it builds the multimer, folds it via ``backend``
    (using your GPU when ``device`` allows), reads ipTM / interface-PAE / pDockQ,
    optionally estimates dG with PRODIGY from the model, then filters on your
    thresholds. Fold only the shortlist you pass in.

    ``sequences`` (``{prefixed_gene: aa_seq}``) is optional: by default every
    chain's sequence is fetched from UniProt for the right species and cached in
    ``sequence_cache_dir`` (``~/.cache/species_cpdb``). ``backend`` defaults to
    :class:`ColabFoldBackend`.

    Returns ``(filtered_hits, scores)`` -- ``scores`` is the per-interaction metric
    table (also usable later with :func:`filter_hits_structurally`).
    """
    os.makedirs(out_dir, exist_ok=True)
    backend = backend or ColabFoldBackend()
    df = hits[hits["cross_species"]] if (cross_only and "cross_species" in hits.columns) else hits
    partners = _partners_by_pair(df, key)
    sequences = _resolve_sequences(partners, sequences, sequence_cache_dir)
    rows = []
    for pair, (lig, rec) in partners.items():
        genes = list(lig) + list(rec)
        missing = [g for g in genes if g not in sequences]
        if missing:
            warnings.warn(f"skipping {pair}: no sequence for {missing}")
            continue
        chains = [(g, sequences[g]) for g in genes]
        fr = backend.fold(pair, chains, out_dir=os.path.join(out_dir, "folds"), device=device)
        row = {key: pair, "iptm": fr.iptm, "ptm": fr.ptm,
               "interface_pae": fr.interface_pae, "pdockq": fr.pdockq, "bsa": None, "delta_g": None}
        if run_prodigy and fr.model_path:
            try:
                row["delta_g"] = predict_affinity_prodigy(fr.model_path, prodigy_temperature)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"PRODIGY failed for {pair}: {e}")
        rows.append(row)

    scores = pd.DataFrame(rows, columns=[key, "iptm", "ptm", "interface_pae", "pdockq", "bsa", "delta_g"])
    filtered = filter_hits_structurally(
        hits, scores=scores, key=key, min_iptm=min_iptm, max_interface_pae=max_interface_pae,
        min_pdockq=min_pdockq, max_delta_g=max_delta_g, keep_unscored=keep_unscored)
    return filtered, scores
