#!/usr/bin/env python3
"""Generate every table, figure and numeric macro of the thesis from the result files.

Every number that appears in ``thesis/`` is produced here from the committed
result files (``results.json``, ``reeval_*.json``, ``stats_wilcoxon.json``,
per-user CSVs).  Nothing is typed by hand into the LaTeX sources: the prose
uses the macros of ``tables/numbers.tex`` and the tables are ``\\input``.

Run from the repository root::

    python thesis/scripts/build_assets.py

Outputs (overwritten on every run):

* ``thesis/tables/*.tex``   booktabs tables
* ``thesis/tables/numbers.tex``  ``\\newcommand`` macros for the prose
* ``thesis/figures/*.pdf``  matplotlib figures

Python 3.9 compatible (typing module, no ``X | Y``).
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT / "thesis"
TABLES = THESIS / "tables"
FIGURES = THESIS / "figures"
RES = ROOT / "results" / "federated"

GROUPS = ("sparse", "medium", "dense")
GROUP_LABEL = {"sparse": "Sparse", "medium": "Medium", "dense": "Dense"}
# Strip the timestamps matplotlib writes into PDFs so that re-running the script on
# unchanged results leaves the committed figures byte-identical.
PDF_META = {"CreationDate": None, "ModDate": None}


# ----------------------------------------------------------------------------
# Run registry (the only place where run ids are typed)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RunSpec:
    """One federated run that the thesis cites."""

    key: str
    module: str
    run_id: str
    label: str
    placement: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not (RES / self.module / self.run_id).is_dir():
            raise FileNotFoundError(f"run directory missing: {self.module}/{self.run_id}")

    @property
    def dir(self) -> Path:
        return RES / self.module / self.run_id

    @property
    def short_id(self) -> str:
        return self.run_id.split("-")[-1]


RUNS: List[RunSpec] = [
    RunSpec("baseline", "baseline", "20260725-093704-e81518", "Baseline (all-global BPR-MF)",
            "none (all parameters global)"),
    RunSpec("personalized", "personalized", "20260610-064423-f18e64", "Personalized (split learning)",
            "user vector kept on device"),
    RunSpec("pfedrec", "pfedrec", "20260726-083121-cf72e0", "PFedRec (+ final calibration)",
            "private scoring head", "final head calibration, 1 epoch"),
    RunSpec("adaptive", "adaptive", "20260613-152226-869730", "Adaptive (dual-level, $\\alpha$ blend)",
            "blend with population prototype + PersonalMLP"),
]
PFEDREC_NOCAL = RunSpec("pfedrec_nocal", "pfedrec", "20260608-071106-ef41ab",
                        "PFedRec (no calibration)", "private scoring head", "no calibration")
PFEDREC_COLD = RunSpec("pfedrec_cold", "pfedrec", "20260527-085237-573e19",
                       "PFedRec, cold-evaluation artefact", "private scoring head", "INVALID: cold eval")

CF72E0 = RUNS[2].dir
POSTTUNE = CF72E0 / "posttune"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f4(x: float) -> str:
    return f"{x:.4f}"


def f3(x: float) -> str:
    return f"{x:.3f}"


def signed(x: float, nd: int = 4) -> str:
    """Signed number; negatives are boxed so that TeX never breaks a line after the minus."""
    s = f"{x:+.{nd}f}"
    return rf"\mbox{{{s}}}" if x < 0 else s


def pfmt(p: float, structural: bool = False) -> str:
    """Format a p-value; structural W^- = 0 cases are labelled, not printed."""
    if structural:
        return r"W$^-$=0"
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p == 0.0:
        return r"$<10^{-300}$"
    if p < 1e-10:
        return r"$<10^{-10}$"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    if p >= 0.9995:
        return "1.0"
    return f"{p:.3g}"


def ci(lo: float, hi: float, nd: int = 4) -> str:
    return rf"\mbox{{[{lo:+.{nd}f}, {hi:+.{nd}f}]}}"


def macro_name(s: str) -> str:
    """LaTeX macro names may contain letters only."""
    out = []
    for part in s.replace("@", "At").replace("/", " ").replace("_", " ").replace("-", " ").split():
        out.append(part[:1].upper() + part[1:])
    name = "".join(out)
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
              "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine", ".": "Dot"}
    return "".join(digits.get(ch, ch) for ch in name)


class Numbers:
    """Collects ``\\newcommand`` macros for ``tables/numbers.tex``."""

    def __init__(self) -> None:
        self.items: List[Tuple[str, str, str]] = []
        self.seen: set = set()

    def add(self, name: str, value: str, comment: str = "") -> None:
        if name in self.seen:
            raise ValueError(f"duplicate macro {name}")
        self.seen.add(name)
        self.items.append((name, value, comment))

    def write(self, path: Path) -> None:
        lines = ["% AUTO-GENERATED by thesis/scripts/build_assets.py -- do not edit by hand.",
                 "% One macro per cited number; the comment names the source file and key.", ""]
        for name, value, comment in self.items:
            lines.append(f"\\newcommand{{\\{name}}}{{{value}}}" + (f"  % {comment}" if comment else ""))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table(name: str, body: str, header_comment: str) -> None:
    text = ("% AUTO-GENERATED by thesis/scripts/build_assets.py -- do not edit by hand.\n"
            f"% {header_comment}\n" + body.rstrip() + "\n")
    (TABLES / name).write_text(text, encoding="utf-8")
    print(f"  wrote tables/{name}")


# ----------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------
@dataclass
class RunData:
    spec: RunSpec
    results: dict
    validity: dict
    best: dict
    last: dict
    cfg: dict
    best_round: int
    history: Dict[int, dict]

    @property
    def git_commit(self) -> str:
        return self.results["_manifest"]["git_commit"][:7]

    @property
    def wall_hours(self) -> Optional[float]:
        """Wall-clock hours between the run-id timestamp and the results timestamp."""
        from datetime import datetime
        try:
            start = datetime.strptime(self.spec.run_id[:15], "%Y%m%d-%H%M%S")
            end = datetime.fromisoformat(self.results["timestamp"])
        except (ValueError, KeyError):
            return None
        return (end - start).total_seconds() / 3600.0


def load_run(spec: RunSpec) -> RunData:
    results = load_json(spec.dir / "results.json")
    validity = load_json(spec.dir / "EVAL_VALIDITY.json")
    fm = results["final_metrics"]
    history = {int(k): v for k, v in results["eval_metrics_history"].items()}
    cfg = dict(results["federated_config"])
    # Values that the BPR modules take from the mode profile
    # (scripts/foundation/fedrec_foundation/mode.py, thesis_crossdevice_main) rather than
    # from federated_config; the manifest records the negatives actually used.
    cfg.setdefault("num_negatives", results["_manifest"].get("num_train_negatives"))
    if cfg.get("mode") == "thesis_crossdevice_main":
        cfg.setdefault("local_epochs", 1)
        cfg.setdefault("optimizer", "adam")
        cfg.setdefault("loss", "bpr")
    return RunData(spec, results, validity, fm["best"], fm["last"], cfg, int(fm["best_round"]), history)


def metric(block: dict, name: str, group: Optional[str] = None) -> float:
    """Read ``sampled_<name>`` with either ``/group`` or ``_group`` suffix convention."""
    if group is None:
        return float(block[f"sampled_{name}"])
    for key in (f"sampled_{name}/{group}", f"sampled_{name}_{group}"):
        if key in block:
            return float(block[key])
    raise KeyError(f"{name}/{group}")


def n_users(block: dict, group: Optional[str] = None) -> int:
    if group is None:
        return int(block["evaluated_users"])
    for key in (f"evaluated_users/{group}", f"evaluated_users_{group}"):
        if key in block:
            return int(block[key])
    raise KeyError(group)


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def table_dataset(num: Numbers) -> None:
    ratings = 0
    users: set = set()
    items: set = set()
    per_user: Counter = Counter()
    with open(ROOT / "data" / "ml-1m" / "ratings.dat", encoding="latin-1") as fh:
        for line in fh:
            u, i, _r, _t = line.rstrip("\n").split("::")
            ratings += 1
            users.add(u)
            items.add(i)
            per_user[u] += 1
    counts = sorted(per_user.values())
    train_counts = [c - 1 for c in counts]  # one held-out test item per user
    groups = Counter("sparse" if c < 30 else "medium" if c < 100 else "dense" for c in train_counts)
    density = 100.0 * ratings / (len(users) * len(items))
    num.add("DsRatings", f"{ratings:,}", "data/ml-1m/ratings.dat")
    num.add("DsUsers", f"{len(users):,}", "data/ml-1m/ratings.dat")
    num.add("DsItems", f"{len(items):,}", "data/ml-1m/ratings.dat")
    num.add("DsDensity", f"{density:.2f}", "ratings / (users x items), percent")
    num.add("DsMinRatings", str(counts[0]), "min ratings per user")
    num.add("DsMedianRatings", f"{statistics.median(counts):.0f}", "median ratings per user")
    num.add("DsMeanRatings", f"{statistics.mean(counts):.1f}", "mean ratings per user")
    num.add("DsMaxRatings", f"{counts[-1]:,}", "max ratings per user")
    num.add("DsExactlyTwenty", str(sum(1 for c in counts if c == 20)), "users with exactly 20 ratings")
    for g in GROUPS:
        num.add(f"DsN{GROUP_LABEL[g]}", f"{groups[g]:,}", f"users in group {g} by training interactions")
    body = r"""\begin{tabular}{lr}
\toprule
Statistic & Value \\
\midrule
Users (= clients) & %s \\
Items & %s \\
Ratings (all treated as implicit positives) & %s \\
Density & %s\%% \\
Ratings per user: min / median / mean / max & %s / %s / %s / %s \\
Training interactions per user (one test item held out) & ratings $-$ 1 \\
Sparse users ($[0, 30)$ training interactions) & %s \\
Medium users ($[30, 100)$) & %s \\
Dense users ($[100, \infty)$) & %s \\
\bottomrule
\end{tabular}""" % (
        f"{len(users):,}", f"{len(items):,}", f"{ratings:,}", f"{density:.2f}",
        counts[0], f"{statistics.median(counts):.0f}", f"{statistics.mean(counts):.1f}", f"{counts[-1]:,}",
        f"{groups['sparse']:,}", f"{groups['medium']:,}", f"{groups['dense']:,}")
    write_table("tab_dataset.tex", body, "MovieLens 1M statistics from data/ml-1m/ratings.dat; groups by training count")


def table_runs(runs: Dict[str, RunData], nocal: RunData, num: Numbers) -> None:
    rows = []
    for key in ("baseline", "personalized", "pfedrec", "adaptive"):
        rd = runs[key]
        cal = "yes" if key == "pfedrec" else "no"
        hours = rd.wall_hours
        rows.append((rd.spec.label, rd.spec.short_id, rd.git_commit, str(rd.best_round), cal,
                     rd.validity["status"], f"{hours:.0f}" if hours else "--"))
        num.add(f"Run{macro_name(key)}Id", rd.spec.run_id, "run id")
        num.add(f"Run{macro_name(key)}Short", rd.spec.short_id, "run id suffix")
        num.add(f"Run{macro_name(key)}Commit", rd.git_commit, "_manifest.git_commit")
        num.add(f"Run{macro_name(key)}BestRound", str(rd.best_round), "final_metrics.best_round")
        if hours:
            num.add(f"Run{macro_name(key)}Hours", f"{hours:.0f}", "wall-clock hours from run-id timestamp to results timestamp")
    hours = nocal.wall_hours
    rows.append((nocal.spec.label, nocal.spec.short_id, nocal.git_commit, str(nocal.best_round), "no",
                 nocal.validity["status"], f"{hours:.0f}" if hours else "--"))
    num.add("RunPfedrecNocalShort", nocal.spec.short_id, "run id suffix")
    num.add("RunPfedrecNocalBestRound", str(nocal.best_round), "final_metrics.best_round")
    body = [r"\begin{tabular}{llllllr}", r"\toprule",
            r"Method & Run & Commit & Best round & Calibration & Validity & Hours$^{\dagger}$ \\",
            r"\midrule"]
    for r in rows:
        body.append(" & ".join(r) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_runs.tex", "\n".join(body),
                "run provenance from results.json (_manifest.git_commit, final_metrics.best_round) and EVAL_VALIDITY.json")


def table_main(runs: Dict[str, RunData], nocal: RunData, centralized: dict, num: Numbers) -> None:
    order = ("baseline", "personalized", "pfedrec", "adaptive")
    body = [r"\begin{tabular}{lcrrrrrr}", r"\toprule",
            r"Method & Best round & NDCG@10 & Sparse & Medium & Dense & HR@10 & HR@10 sparse \\",
            r"\midrule"]
    for key in order:
        rd = runs[key]
        b = rd.best
        vals = [metric(b, "ndcg@10")] + [metric(b, "ndcg@10", g) for g in GROUPS] + \
               [metric(b, "hr@10"), metric(b, "hr@10", "sparse")]
        body.append(f"{rd.spec.label} & {rd.best_round} & " + " & ".join(f4(v) for v in vals) + r" \\")
        mk = macro_name(key)
        num.add(f"Ndcg{mk}", f4(vals[0]), f"{rd.spec.short_id} final_metrics.best sampled_ndcg@10")
        for g, v in zip(GROUPS, vals[1:4]):
            num.add(f"Ndcg{mk}{GROUP_LABEL[g]}", f4(v), f"{rd.spec.short_id} sampled_ndcg@10/{g}")
        num.add(f"Hr{mk}", f4(vals[4]), f"{rd.spec.short_id} sampled_hr@10")
        num.add(f"Hr{mk}Sparse", f4(vals[5]), f"{rd.spec.short_id} sampled_hr@10/sparse")
    body.append(r"\midrule")
    b = nocal.best
    vals = [metric(b, "ndcg@10")] + [metric(b, "ndcg@10", g) for g in GROUPS] + \
           [metric(b, "hr@10"), metric(b, "hr@10", "sparse")]
    body.append(f"{nocal.spec.label} & {nocal.best_round} & " + " & ".join(f4(v) for v in vals) + r" \\")
    num.add("NdcgPfedrecNocal", f4(vals[0]), f"{nocal.spec.short_id} sampled_ndcg@10")
    num.add("HrPfedrecNocal", f4(vals[4]), f"{nocal.spec.short_id} sampled_hr@10")
    for g, v in zip(GROUPS, vals[1:4]):
        num.add(f"NdcgPfedrecNocal{GROUP_LABEL[g]}", f4(v), f"{nocal.spec.short_id} sampled_ndcg@10/{g}")
    cm = centralized["metrics"]
    body.append(r"Centralized BPR-MF$^{\ddagger}$ & -- & "
                f"{f4(cm['ndcg@10'])} & -- & -- & -- & {f4(cm['hr@10'])} & -- \\\\")
    num.add("NdcgCentralBpr", f4(cm["ndcg@10"]), "results/centralized/bpr_mf_centralized_results.json")
    num.add("HrCentralBpr", f4(cm["hr@10"]), "results/centralized/bpr_mf_centralized_results.json")
    cp = centralized["params"]
    num.add("CentralBprDim", str(cp["embedding_dim"]), "centralized params")
    num.add("CentralBprEpochs", str(cp["n_epochs"]), "centralized params")
    num.add("CentralBprNeg", str(cp["num_negatives"]), "centralized params")
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_main.tex", "\n".join(body),
                "final_metrics.best of each run (6040 users, best-round restore); centralized from results/centralized")
    # relative gains used in the prose
    base = metric(runs["baseline"].best, "ndcg@10")
    pers = metric(runs["personalized"].best, "ndcg@10")
    num.add("GainPersonalizedVsBaselinePct", f"{100.0 * (pers - base) / base:.1f}", "(personalized - baseline)/baseline, percent")
    num.add("GainPersonalizedVsBaselineSparsePct",
            f"{100.0 * (metric(runs['personalized'].best, 'ndcg@10', 'sparse') - metric(runs['baseline'].best, 'ndcg@10', 'sparse')) / metric(runs['baseline'].best, 'ndcg@10', 'sparse'):.1f}",
            "sparse group, percent")
    common = sorted(set(runs["pfedrec"].history) & set(nocal.history))
    num.add("DeltaInLoopNocalMax",
            f4(max(abs(runs["pfedrec"].history[r]["sampled_ndcg@10"] - nocal.history[r]["sampled_ndcg@10"]) for r in common)),
            "max |in-loop NDCG@10(cf72e0) - in-loop NDCG@10(ef41ab)| over common rounds (same client sequence)")
    num.add("DeltaPfedrecCalibration", signed(metric(runs["pfedrec"].best, "ndcg@10") - metric(nocal.best, "ndcg@10")),
            "cf72e0 - ef41ab NDCG@10 (confounded by best round and code version)")


def table_main_groups(runs: Dict[str, RunData]) -> None:
    """Per-group table with user counts, NDCG@10, HR@10 for the four methods."""
    body = [r"\begin{tabular}{llrrrr}", r"\toprule",
            r"Method & Group & Users & NDCG@10 & HR@10 & MRR \\", r"\midrule"]
    for key in ("baseline", "personalized", "pfedrec", "adaptive"):
        rd = runs[key]
        b = rd.best
        first = True
        for g in ("overall",) + GROUPS:
            gg = None if g == "overall" else g
            try:
                mrr = f4(metric(b, "mrr", gg))
            except KeyError:
                mrr = "--"
            body.append(f"{rd.spec.label if first else ''} & {g} & {n_users(b, gg):,} & "
                        f"{f4(metric(b, 'ndcg@10', gg))} & {f4(metric(b, 'hr@10', gg))} & {mrr} \\\\")
            first = False
        body.append(r"\midrule")
    body[-1] = r"\bottomrule"
    body.append(r"\end{tabular}")
    write_table("tab_main_groups.tex", "\n".join(body), "final_metrics.best per group; MRR only where the run logged it")


def table_last_round(runs: Dict[str, RunData]) -> None:
    """In-loop last-round metrics (604 sampled clients) next to the best-round full-population metrics."""
    body = [r"\begin{tabular}{lrrrrr}", r"\toprule",
            r"Method & Best round & Best, 6040 users & In-loop at best (604) & In-loop at 100 (604) & Full pop.\ at 100 \\",
            r"\midrule"]
    for key in ("baseline", "personalized", "pfedrec", "adaptive"):
        rd = runs[key]
        inloop_best = rd.history[rd.best_round]["sampled_ndcg@10"]
        inloop_last = rd.history[max(rd.history)]["sampled_ndcg@10"]
        body.append(f"{rd.spec.label} & {rd.best_round} & {f4(metric(rd.best, 'ndcg@10'))} & "
                    f"{f4(inloop_best)} & {f4(inloop_last)} & \\pending{{re-score}} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_last_round.tex", "\n".join(body),
                "eval_metrics_history (in-loop, sampled clients) vs final_metrics.best; full-pop round-100 column pending")


def table_pfedrec_rows(runs: Dict[str, RunData], nocal: RunData, qu: dict, num: Numbers) -> None:
    """The three registered PFedRec rows (posttune/README.md section 5) plus the no-calibration run."""
    base = qu["baseline_metrics"]
    cells = qu["metrics_by_cell"]
    rows = [
        ("PFedRec, scored with the shared item table (thesis number)", base),
        ("PFedRec, post-tuning, clean negatives ($E=1$)", cells["a0.00_E1_clean"]),
        ("PFedRec, post-tuning, official-code negatives ($E=1$)", cells["a0.00_E1_leaky"]),
    ]
    body = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r"Scoring protocol & NDCG@10 & Sparse & Medium & Dense & HR@10 & MRR \\", r"\midrule"]
    for label, m in rows:
        body.append(f"{label} & {f4(metric(m, 'ndcg@10'))} & " +
                    " & ".join(f4(metric(m, "ndcg@10", g)) for g in GROUPS) +
                    f" & {f4(metric(m, 'hr@10'))} & {f4(metric(m, 'mrr'))} \\\\")
    body.append(r"\midrule")
    b = nocal.best
    body.append(f"PFedRec without final calibration (run {nocal.spec.short_id}, best round {nocal.best_round}) & "
                f"{f4(metric(b, 'ndcg@10'))} & " + " & ".join(f4(metric(b, 'ndcg@10', g)) for g in GROUPS) +
                f" & {f4(metric(b, 'hr@10'))} & -- \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_pfedrec_rows.tex", "\n".join(body),
                "posttune/reeval_qu_diag_best.json baseline_metrics + metrics_by_cell (E=1 cells); ef41ab final_metrics.best")
    for cell, mk in (("a0.00_E1_clean", "EOneClean"), ("a0.00_E1_leaky", "EOneLeaky"),
                     ("a0.00_E3_clean", "EThreeClean"), ("a0.00_E3_leaky", "EThreeLeaky")):
        m = cells[cell]
        num.add(f"Ndcg{mk}", f4(metric(m, "ndcg@10")), f"metrics_by_cell.{cell}")
        num.add(f"Hr{mk}", f4(metric(m, "hr@10")), f"metrics_by_cell.{cell}")
        for g in GROUPS:
            num.add(f"Ndcg{mk}{GROUP_LABEL[g]}", f4(metric(m, "ndcg@10", g)), f"metrics_by_cell.{cell}/{g}")
    num.add("NdcgPfedrecMrr", f4(metric(base, "mrr")), "baseline_metrics sampled_mrr")
    tc = qu["trust_checks"]
    num.add("PtElapsedSec", f"{tc['elapsed_sec']:.0f}", "trust_checks.elapsed_sec")
    num.add("PtNHitClean", f"{tc['n_hit']['clean']:,}", "trust_checks.n_hit.clean")
    num.add("PtNHitLeaky", f"{tc['n_hit']['leaky']:,}", "trust_checks.n_hit.leaky")
    num.add("PtLrItem", f"{tc['lr_item']:,.0f}", "trust_checks.lr_item")
    num.add("PtRowNormMedian", f"{tc['global_row_norm_median']:.2f}", "trust_checks.global_row_norm_median")
    num.add("PtRowShiftMedianEOne", f"{tc['mean_row_shift_median']['clean_E1']:.2f}", "trust_checks.mean_row_shift_median.clean_E1")
    num.add("PtRowShiftMedianEThree", f"{tc['mean_row_shift_median']['clean_E3']:.2f}", "trust_checks.mean_row_shift_median.clean_E3")
    num.add("PtGlobalsSha", qu["globals_sha256"][:12] + "\\ldots", "globals_sha256 (prefix)")


def _scope(stats: dict, cell: str, scope: str) -> dict:
    return stats["cells"][cell]["by_scope"][scope]


def table_posttune_wilcoxon(stats: dict, per_user: List[Dict[str, str]], cached: List[Dict[str, str]],
                            num: Numbers) -> None:
    base_ndcg = {r["pid"]: float(r["sampled_ndcg@10"]) for r in cached}
    cells = ("a0.00_E1_clean", "a0.00_E1_leaky", "a0.00_E3_clean", "a0.00_E3_leaky")
    label = {"a0.00_E1_clean": "$E=1$, clean", "a0.00_E1_leaky": "$E=1$, official-code",
             "a0.00_E3_clean": "$E=3$, clean", "a0.00_E3_leaky": "$E=3$, official-code"}
    counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    zero_losers: Counter = Counter()
    for r in per_user:
        cell = f"a{float(r['alpha_table']):.2f}_E{r['epochs']}_{r['pool']}"
        d = float(r["sampled_ndcg@10"]) - base_ndcg[r["pid"]]
        key = "pos" if d > 0 else "neg" if d < 0 else "zero"
        counts[(cell, "overall")][key] += 1
        counts[(cell, r["user_group"])][key] += 1
        if d < 0 and float(r["sampled_ndcg@10"]) == 0.0 and r["test_row_touched"] == "1":
            zero_losers[cell] += 1
    body = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r"Cell & $\Delta$NDCG@10 [95\% CI] & $n_+/n_-/n_0$ & $p$ one-sided & $p$ two-sided & $p$ Pratt & $r$ \\",
            r"\midrule"]
    for cell in cells:
        s = _scope(stats, cell, "overall")
        c = counts[(cell, "overall")]
        structural = s["w_minus"] == 0
        body.append(f"{label[cell]} & {signed(s['mean_delta_ndcg'])} {ci(*s['ci95_ndcg'])} & "
                    f"{c['pos']}/{c['neg']}/{c['zero']} & "
                    f"{pfmt(s['p_one_sided'], structural)} & {pfmt(s['p_two_sided'], structural)} & "
                    f"{pfmt(s['p_pratt_one_sided'], structural)} & {s['rank_biserial_r']:+.3f} \\\\")
        mk = macro_name(cell.replace("a0.00_", ""))
        num.add(f"PtDelta{mk}", signed(s["mean_delta_ndcg"]), f"stats_wilcoxon cells.{cell}.by_scope.overall.mean_delta_ndcg")
        num.add(f"PtCi{mk}", ci(*s["ci95_ndcg"]), f"cells.{cell}.by_scope.overall.ci95_ndcg")
        num.add(f"PtDeltaHr{mk}", signed(s["mean_delta_hr"]), f"cells.{cell}.by_scope.overall.mean_delta_hr")
        num.add(f"PtPOne{mk}", pfmt(s["p_one_sided"], structural), f"cells.{cell}.p_one_sided")
        num.add(f"PtPTwo{mk}", pfmt(s["p_two_sided"], structural), f"cells.{cell}.p_two_sided")
        num.add(f"PtPPratt{mk}", pfmt(s["p_pratt_one_sided"], structural), f"cells.{cell}.p_pratt_one_sided")
        num.add(f"PtR{mk}", f"{s['rank_biserial_r']:+.3f}", f"cells.{cell}.rank_biserial_r")
        num.add(f"PtNPos{mk}", f"{c['pos']:,}", "per-user CSV: users with d_u > 0")
        num.add(f"PtNNeg{mk}", f"{c['neg']:,}", "per-user CSV: users with d_u < 0")
        num.add(f"PtNZero{mk}", f"{c['zero']:,}", "per-user CSV: users with d_u = 0")
        num.add(f"PtZeroLosers{mk}", f"{zero_losers[cell]:,}", "losers whose NDCG is exactly 0 and whose test row was drawn")
        num.add(f"PtWPlus{mk}", f"{s['w_plus']:,.0f}", f"cells.{cell}.w_plus")
        num.add(f"PtWMinus{mk}", f"{s['w_minus']:,.0f}", f"cells.{cell}.w_minus")
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_posttune_wilcoxon.tex", "\n".join(body),
                "stats_wilcoxon.json cells.<cell>.by_scope.overall; n+/n-/n0 recomputed from the two per-user CSVs; W^-=0 marks p-values fixed by construction (< 1e-10); delta HR in the per-group table")
    # Holm (one-sided) values
    for cell in cells:
        s = _scope(stats, cell, "overall")
        mk = macro_name(cell.replace("a0.00_", ""))
        num.add(f"PtPHolm{mk}", pfmt(s.get("p_holm_one_sided", float("nan")), s["w_minus"] == 0),
                f"cells.{cell}.p_holm_one_sided")

    # per-group appendix table
    body = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
            r"Cell & Group & $\Delta$NDCG@10 & 95\% CI & $\Delta$HR@10 & $p$ one-sided & $p$ Pratt & $n_+/n_-/n_0$ \\",
            r"\midrule"]
    for cell in cells:
        for g in GROUPS:
            s = _scope(stats, cell, g)
            c = counts[(cell, g)]
            structural = s["w_minus"] == 0 and c["pos"] > 60
            body.append(f"{label[cell]} & {g} & {signed(s['mean_delta_ndcg'])} & {ci(*s['ci95_ndcg'])} & "
                        f"{signed(s['mean_delta_hr'])} & {pfmt(s['p_one_sided'], structural)} & "
                        f"{pfmt(s['p_pratt_one_sided'], structural)} & {c['pos']}/{c['neg']}/{c['zero']} \\\\")
        body.append(r"\midrule")
    body[-1] = r"\bottomrule"
    body.append(r"\end{tabular}")
    write_table("tab_posttune_groups.tex", "\n".join(body),
                "stats_wilcoxon.json cells.<cell>.by_scope.<group>; structural label when W^-=0 and >60 users changed")
    for g in GROUPS:
        s = _scope(stats, "a0.00_E1_clean", g)
        num.add(f"PtDeltaEOneClean{GROUP_LABEL[g]}", signed(s["mean_delta_ndcg"]), f"E1_clean by_scope.{g}")
        s = _scope(stats, "a0.00_E1_leaky", g)
        num.add(f"PtDeltaEOneLeaky{GROUP_LABEL[g]}", signed(s["mean_delta_ndcg"]), f"E1_leaky by_scope.{g}")


def table_posttune_contrasts(stats: dict, per_user: List[Dict[str, str]], num: Numbers) -> None:
    by_cell: Dict[Tuple[str, str, str], float] = {}
    group_of: Dict[str, str] = {}
    for r in per_user:
        by_cell[(r["pid"], r["epochs"], r["pool"])] = float(r["sampled_ndcg@10"])
        group_of[r["pid"]] = r["user_group"]
    body = [r"\begin{tabular}{llrrrr}", r"\toprule",
            r"Contrast & Group & $\Delta$NDCG@10 (official $-$ clean) & 95\% CI & $p$ one-sided & $n_{\neq 0}$ \\",
            r"\midrule"]
    for e in ("1", "3"):
        key = f"a0.00_E{e}_leaky__minus__a0.00_E{e}_clean"
        con = stats["contrasts"][key]
        nz = Counter()
        for pid, g in group_of.items():
            d = by_cell[(pid, e, "leaky")] - by_cell[(pid, e, "clean")]
            if d != 0:
                nz["overall"] += 1
                nz[g] += 1
        for scope in ("overall",) + GROUPS:
            s = con["by_scope"][scope]
            structural = s["w_minus"] == 0 and nz[scope] > 60
            body.append(f"$E={e}$ & {scope} & {signed(s['mean_delta_ndcg'])} & {ci(*s['ci95_ndcg'])} & "
                        f"{pfmt(s['p_one_sided'], structural)} & {nz[scope]:,} \\\\")
            if scope == "overall":
                mk = "EOne" if e == "1" else "EThree"
                num.add(f"PtContrast{mk}", signed(s["mean_delta_ndcg"]), f"contrasts.{key}.by_scope.overall.mean_delta_ndcg")
                num.add(f"PtContrast{mk}Ci", ci(*s["ci95_ndcg"]), f"contrasts.{key}.by_scope.overall.ci95_ndcg")
                num.add(f"PtContrast{mk}NonZero", f"{nz['overall']:,}", "users whose two trajectories differ (per-user CSV)")
                num.add(f"PtContrast{mk}PerUser", f"{s['mean_delta_ndcg'] * 6040 / nz['overall']:.2f}",
                        "overall delta spread over the affected users only")
            if scope == "dense" and e == "1":
                num.add("PtContrastEOneDense", signed(s["mean_delta_ndcg"]), f"contrasts.{key}.by_scope.dense")
            if scope == "sparse" and e == "1":
                num.add("PtContrastEOneSparse", signed(s["mean_delta_ndcg"]), f"contrasts.{key}.by_scope.sparse")
        body.append(r"\midrule")
    body[-1] = r"\bottomrule"
    body.append(r"\end{tabular}")
    write_table("tab_posttune_contrasts.tex", "\n".join(body),
                "stats_wilcoxon.json contrasts.<A>__minus__<B>.by_scope; n_neq0 from the per-user CSV")


def table_posttune_cells(qu: dict) -> None:
    base = qu["baseline_metrics"]
    cells = qu["metrics_by_cell"]
    rows = [("Shared item table (reference)", base)] + [
        (lab, cells[c]) for c, lab in (("a0.00_E1_clean", "$E=1$, clean"), ("a0.00_E1_leaky", "$E=1$, official-code"),
                                       ("a0.00_E3_clean", "$E=3$, clean"), ("a0.00_E3_leaky", "$E=3$, official-code"))]
    body = [r"\begin{tabular}{llrrrr}", r"\toprule",
            r"Cell & Metric & Overall & Sparse & Medium & Dense \\", r"\midrule"]
    for lab, m in rows:
        for j, (name, met) in enumerate((("NDCG@10", "ndcg@10"), ("HR@10", "hr@10"), ("MRR", "mrr"))):
            cols = [f4(metric(m, met, g)) for g in (None,) + GROUPS]
            body.append(f"{lab if j == 0 else ''} & {name} & " + " & ".join(cols) + r" \\")
        body.append(r"\midrule")
    body[-1] = r"\bottomrule"
    body.append(r"\end{tabular}")
    write_table("tab_posttune_cells.tex", "\n".join(body),
                "posttune/reeval_qu_diag_best.json: one row per cell and metric, columns = overall and groups")


def table_head_counterfactuals(num: Numbers) -> None:
    variants = [
        ("cached", "Own calibrated head $w_u$ (reference)"),
        ("alpha_policy_hc", "Blend $\\alpha_u w_u + (1-\\alpha_u)\\bar w$, hierarchical-conditional $\\alpha_u$"),
        ("global_head", "Population-mean head $\\bar w$ (uniform mean of 6040 heads)"),
        ("global_head_weighted", "Population head, weighted by interaction count"),
        ("shuffled", "Another user's head (random permutation)"),
        ("random", "Freshly initialised head per user"),
        ("global_head_cold", "One shared freshly initialised head"),
    ]
    ref = load_json(CF72E0 / "reeval_cached_best.json")["metrics"]
    ref_ndcg = metric(ref, "ndcg@10")
    body = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r"Head used at scoring time & NDCG@10 & $\Delta$ vs.\ own head & Sparse & Medium & Dense & HR@10 \\",
            r"\midrule"]
    for key, lab in variants:
        j = load_json(CF72E0 / f"reeval_{key}_best.json")
        m = j["metrics"]
        v = metric(m, "ndcg@10")
        body.append(f"{lab} & {f4(v)} & {signed(v - ref_ndcg) if key != 'cached' else '--'} & " +
                    " & ".join(f4(metric(m, "ndcg@10", g)) for g in GROUPS) + f" & {f4(metric(m, 'hr@10'))} \\\\")
        mk = macro_name(key)
        num.add(f"NdcgHead{mk}", f4(v), f"reeval_{key}_best.json metrics.sampled_ndcg@10")
        num.add(f"DeltaHead{mk}", signed(v - ref_ndcg), f"reeval_{key}_best.json minus cached")
        num.add(f"AbsDeltaHead{mk}", f4(abs(v - ref_ndcg)), f"|reeval_{key}_best.json minus cached|")
        for g in GROUPS:
            num.add(f"NdcgHead{mk}{GROUP_LABEL[g]}", f4(metric(m, "ndcg@10", g)), f"reeval_{key}_best.json /{g}")
        if key == "alpha_policy_hc":
            a = j["alpha_summary"]
            num.add("HeadAlphaMean", f"{a['mean']:.3f}", "reeval_alpha_policy_hc_best.json alpha_summary.mean")
            num.add("HeadAlphaMin", f"{a['min']:.3f}", "alpha_summary.min")
            num.add("HeadAlphaMax", f"{a['max']:.3f}", "alpha_summary.max")
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_head_counterfactuals.tex", "\n".join(body),
                "reeval_<variant>_best.json of run cf72e0; same globals, same 100 candidates per user")
    # alpha grid: best single alpha
    grid = load_json(CF72E0 / "reeval_alpha_grid_best.json")
    best_alpha, best_val = None, -1.0
    for k, v in grid["metrics_by_alpha"].items():
        val = float(v["sampled_ndcg@10"])
        if val > best_val:
            best_val, best_alpha = val, float(k)
    num.add("HeadAlphaStarGlobal", f"{best_alpha:.2f}", "reeval_alpha_grid_best.json metrics_by_alpha: alpha with the highest overall NDCG@10")
    num.add("HeadAlphaStarNdcg", f4(best_val), "reeval_alpha_grid_best.json: NDCG@10 at that alpha")
    num.add("HeadAlphaStarRecorded", f"{float(grid['alpha_star_global']):.2f}", "reeval_alpha_grid_best.json alpha_star_global")
    num.add("HeadAlphaGridStep", f"{sorted(float(k) for k in grid['metrics_by_alpha'])[1]:.2f}", "grid step")
    num.add("HeadAlphaGridSize", str(len(grid["metrics_by_alpha"])), "number of alpha values on the grid")


def table_params_comm(runs: Dict[str, RunData], num: Numbers) -> None:
    """Parameter counts derived from the architectures (not measured bytes)."""
    n_users_, n_items = 6040, 3706
    d_mf = int(runs["baseline"].cfg["embedding_dim"])
    d_pf = int(runs["pfedrec"].cfg["latent_dim"])
    hidden = list(runs["adaptive"].cfg["mlp_hidden_dims"])
    base_global = n_users_ * d_mf + n_items * d_mf + n_users_ + n_items + 1
    split_global = n_items * d_mf + n_items + 1
    split_local = d_mf + 1
    pf_global = n_items * d_pf + 1  # item table + shared bias
    pf_local = d_pf
    mlp = 0
    prev = d_mf
    for h in hidden:
        mlp += prev * h + h
        prev = h
    mlp += prev * 1 + 1
    ad_local = split_local + mlp + 3  # concat fusion: Linear(2,1) = 3 params
    rows = [
        ("Baseline (all-global BPR-MF)", base_global, 0, base_global),
        ("Personalized (split learning)", split_global, split_local, split_global),
        ("PFedRec", pf_global, pf_local, pf_global),
        ("Adaptive (dual-level)", split_global, ad_local, split_global + d_mf),
    ]
    body = [r"\begin{tabular}{lrrr}", r"\toprule",
            r"Method & Global parameters (aggregated) & Local parameters per client & Uploaded per participating client per round$^{\dagger}$ \\",
            r"\midrule"]
    for lab, g, l, up in rows:
        body.append(f"{lab} & {g:,} & {l:,} & {up:,} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_params_comm.tex", "\n".join(body),
                "derived from federated_config dims (users 6040, items 3706); measured byte counts are pending")
    num.add("ParamsBaselineGlobal", f"{base_global:,}", "derived: (U+I)d + U + I + 1")
    num.add("ParamsSplitGlobal", f"{split_global:,}", "derived: I d + I + 1")
    num.add("ParamsPfedrecGlobal", f"{pf_global:,}", "derived: I d + 1")
    num.add("ParamsPfedrecLocal", f"{pf_local:,}", "derived: d")
    num.add("ParamsAdaptiveLocal", f"{ad_local:,}", "derived: user row + bias + PersonalMLP + fusion")
    num.add("ParamsPersonalMlp", f"{mlp:,}", f"derived: MLP {d_mf}->{'->'.join(map(str, hidden))}->1")
    num.add("ParamsAdaptiveUploadExtra", f"{d_mf}", "p_u sent as prototype contribution (floats)")
    num.add("DimMf", str(d_mf), "embedding_dim of baseline/personalized/adaptive")
    num.add("DimPfedrec", str(d_pf), "latent_dim of pfedrec")
    num.add("MlpHidden", "--".join(map(str, hidden)), "mlp_hidden_dims")


def table_hyperparams(runs: Dict[str, RunData]) -> None:
    keys = [("strategy", "Strategy"), ("model_type", "Model type"), ("embedding_dim", "Embedding dim"),
            ("latent_dim", "Latent dim"), ("learning_rate", "Learning rate"), ("lr", "Learning rate (head)"),
            ("lr_eta", "Item LR multiplier $\\eta$"), ("local_epochs", "Local epochs"),
            ("num_negatives", "Training negatives"), ("proximal_mu", "FedProx $\\mu$"), ("dropout", "Dropout"),
            ("fraction_train", "Participation $C$"), ("num_rounds", "Rounds $R$"), ("weight_policy", "Aggregation weights"),
            ("loss", "Loss"), ("optimizer", "Optimizer"), ("fusion_type", "Fusion"), ("mlp_hidden_dims", "PersonalMLP hidden"),
            ("prototype_momentum", "Prototype momentum"), ("checkpoint_rule", "Checkpoint rule"), ("run_seed", "Seed")]
    order = ("baseline", "personalized", "pfedrec", "adaptive")
    body = [r"\begin{tabular}{lllll}", r"\toprule", "Setting & " + " & ".join(
        {"baseline": "Baseline", "personalized": "Personalized", "pfedrec": "PFedRec", "adaptive": "Adaptive"}[k] for k in order) + r" \\",
            r"\midrule"]
    for key, lab in keys:
        vals = []
        present = False
        for k in order:
            v = runs[k].cfg.get(key)
            if v is None:
                vals.append("--")
            else:
                present = True
                vals.append(str(v).replace("_", r"\_").replace("[", "").replace("]", "").replace(", ", "--"))
        if present:
            body.append(f"{lab} & " + " & ".join(vals) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_hyperparams.tex", "\n".join(body), "federated_config of each cited run (results.json)")


def table_asymmetries(runs: Dict[str, RunData]) -> None:
    body = [r"\begin{tabular}{lllll}", r"\toprule",
            r"Aspect & Baseline & Personalized & PFedRec & Adaptive \\", r"\midrule",
            r"Loss & BPR & BPR & BCE (4 negatives) & BPR \\",
            f"Embedding dimension & {runs['baseline'].cfg['embedding_dim']} & {runs['personalized'].cfg['embedding_dim']} & "
            f"{runs['pfedrec'].cfg['latent_dim']} & {runs['adaptive'].cfg['embedding_dim']} \\\\",
            r"Optimizer & Adam & Adam & SGD (alternating) & Adam \\",
            f"Strategy & {runs['baseline'].cfg['strategy']} & {runs['personalized'].cfg['strategy']} & "
            f"{runs['pfedrec'].cfg['strategy']} & {runs['adaptive'].cfg['strategy']} \\\\",
            r"Final calibration of the local part & n/a & no & yes (1 epoch, items frozen) & no \\",
            f"Best round (selected on the test item) & {runs['baseline'].best_round} & {runs['personalized'].best_round} & "
            f"{runs['pfedrec'].best_round} & {runs['adaptive'].best_round} \\\\",
            r"Extra upload beyond the global part & none & none & none & $p_u$ (prototype) \\",
            r"\bottomrule", r"\end{tabular}"]
    write_table("tab_asymmetries.tex", "\n".join(body), "configuration asymmetries between the four cited runs")


def table_predictions(num: Numbers) -> None:
    body = r"""\begin{tabular}{@{}p{0.04\linewidth}>{\raggedright\arraybackslash}p{0.40\linewidth}>{\raggedright\arraybackslash}p{0.36\linewidth}>{\raggedright\arraybackslash}p{0.13\linewidth}@{}}
\toprule
\# & Prediction (registered before the run) & Observed & Verdict \\
\midrule
A1 & Clean negatives, $E=1$: NDCG@10 below the reference; $E=3$ below $E=1$ & \NdcgEOneClean{} ($\Delta$ \PtDeltaEOneClean{}, CI \PtCiEOneClean{}); $E=3$: \NdcgEThreeClean{} & Correct on the mean \\
A2 & Official-code negatives, $E=1$: $\Delta \geq +0.010$, one-sided $p<0.05$; $E=3 \geq E=1$ & $\Delta$ \PtDeltaEOneLeaky{} \PtCiEOneLeaky{}; $E=3$: \NdcgEThreeLeaky{} & Correct \\
A3 & Official $-$ clean $\geq +0.010$ at both $E$ & \PtContrastEOne{} and \PtContrastEThree{} & Correct \\
A4 & Dense users have the largest $|\Delta|$ in all four cells & holds in 4/4 cells (Table~\ref{tab:posttune-groups}) & Correct \\
A5 & No cell reaches 0.394 (half of the gap to the published 0.4436) & $E=3$, official-code negatives: \NdcgEThreeLeaky{} & Refuted \\
\bottomrule
\end{tabular}"""
    write_table("tab_predictions.tex", body, "posttune/README.md section 8 predictions vs RESULT.md section 4 (numbers via macros)")


# ----------------------------------------------------------------------------
# Staleness and cold-start numbers (contribution 4)
# ----------------------------------------------------------------------------
def staleness_numbers(runs: Dict[str, RunData], cached: List[Dict[str, str]], num: Numbers) -> List[int]:
    stale = [int(r["staleness_at_best"]) for r in cached]
    stale_sorted = sorted(stale)
    num.add("StaleMedian", f"{statistics.median(stale):.0f}", "reeval_cached_best_per_user.csv staleness_at_best median")
    num.add("StaleMean", f"{statistics.mean(stale):.1f}", "staleness_at_best mean")
    num.add("StalePNinety", f"{stale_sorted[int(0.9 * len(stale_sorted)) - 1]}", "staleness_at_best 90th percentile")
    num.add("StaleMax", f"{max(stale)}", "staleness_at_best max")
    num.add("StaleOverTen", f"{100.0 * sum(1 for s in stale if s > 10) / len(stale):.1f}", "percent of users with staleness > 10 rounds")
    num.add("StaleZero", f"{100.0 * sum(1 for s in stale if s == 0) / len(stale):.1f}", "percent of users that participated in the best round")
    cs = runs["pfedrec"].results["cold_starts"]
    per_round = cs["per_round"]
    first_full = next((i + 1 for i in range(len(per_round)) if sum(per_round[: i + 1]) >= 6040), None)
    num.add("ColdStartRoundOne", f"{per_round[0]}", "cold_starts.per_round[0] of cf72e0")
    num.add("ColdStartAllSeenRound", str(first_full) if first_full else "--", "first round by which every client has participated once")
    num.add("ClientsPerRound", f"{int(0.1 * 6040)}", "C x N")
    num.add("ExpectedParticipations", f"{0.1 * 100:.0f}", "C x R expected participations per client")
    # cold evaluation artefact numbers
    cold = load_json(PFEDREC_COLD.dir / "results.json")
    fm = cold["final_metrics"]
    num.add("ColdEvalNdcg", f4(fm["best"]["sampled_ndcg@10"]), "573e19 final_metrics.best (poisoned, cold heads)")
    hist = {int(k): v for k, v in cold["eval_metrics_history"].items()}
    num.add("ColdEvalInLoopNdcg", f4(hist[max(hist)]["sampled_ndcg@10"]), "573e19 in-loop NDCG@10 at the last round (604 users)")
    num.add("ColdEvalRatio", f"{hist[max(hist)]['sampled_ndcg@10'] / fm['best']['sampled_ndcg@10']:.1f}", "in-loop / cold ratio")
    num.add("RunPfedrecColdShort", PFEDREC_COLD.short_id, "run id suffix")
    # cache-miss counter of the valid runs
    for key in ("pfedrec", "adaptive"):
        b = runs[key].best
        if "d06_cache_misses" in b:
            num.add(f"CacheMisses{macro_name(key)}", str(int(b["d06_cache_misses"])), f"{key} final_metrics.best.d06_cache_misses")
    # best-round inflation: in-loop max minus mean of the last 10 in-loop rounds
    for key in ("baseline", "personalized", "pfedrec", "adaptive"):
        h = runs[key].history
        vals = [h[r]["sampled_ndcg@10"] for r in sorted(h)]
        infl = max(vals) - statistics.mean(vals[-10:])
        num.add(f"Inflation{macro_name(key)}", f"{100 * infl:.1f}", "in-loop max minus mean of last 10 rounds, NDCG@10 points")
    # NDCG@10 by "staleness at the best round" bucket on the calibrated run.
    # k = rounds between the best round r* and the user's last participation up to r* (reeval
    # staleness_at_best). Caveats recorded as macros: the k = 0 bucket is exactly the in-loop
    # sample on which r* was selected, and most users participated again after r*, so k is
    # not the age of the cached private part for them (caches are written until the last round,
    # best-round restore resets only the shared table).
    import numpy as np
    rng = np.random.default_rng(42)
    sel = runs["pfedrec"].results["selected_clients_per_round"]
    best_round = runs["pfedrec"].best_round
    k_zero = {int(r["pid"]) for r in cached if int(r["staleness_at_best"]) == 0}
    assert k_zero == {int(x) for x in sel[best_round - 1]}, "k=0 bucket must equal the clients of the best round"
    last_part = {}
    visits_to_best = {pid: 0 for pid in range(len(cached))}
    for rnd, ids in enumerate(sel, start=1):
        for pid in ids:
            last_part[int(pid)] = rnd
            if rnd <= best_round:
                visits_to_best[int(pid)] += 1
    after_best = sum(1 for pid, r in last_part.items() if r > best_round)
    num.add("StaleAfterBestPct", f"{100.0 * after_best / len(cached):.1f}",
            "percent of users whose last participation in the whole run is after the best round (selected_clients_per_round)")
    num.add("StaleInLoopAtBest", f4(runs["pfedrec"].history[best_round]["sampled_ndcg@10"]),
            "in-loop NDCG@10 of the 604 clients of the best round (eval_metrics_history)")

    def boot_ci(vals: List[float]) -> Tuple[float, float]:
        arr = np.asarray(vals)
        means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(10000)]
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    buckets = [("0 (clients of the best round)", "Zero", lambda k: k == 0),
               ("1--5", "OneFive", lambda k: 1 <= k <= 5), ("6--10", "SixTen", lambda k: 6 <= k <= 10),
               ("11--20", "ElevenTwenty", lambda k: 11 <= k <= 20), (">20", "OverTwenty", lambda k: k > 20),
               ("All $k \\geq 1$", "KOne", lambda k: k >= 1), ("All", "All", lambda k: True)]
    body = [r"\begin{tabular}{lrrrrc}", r"\toprule",
            r"$k$ (rounds since last participation up to $r^\ast$) & Users & Interactions & Visits $\leq r^\ast$ & NDCG@10 & 95\% CI \\",
            r"\midrule"]
    bucket_ndcg: Dict[str, float] = {}
    for label, mk, pred in buckets:
        rows = [r for r in cached if pred(int(r["staleness_at_best"]))]
        vals = [float(r["sampled_ndcg@10"]) for r in rows]
        nd = statistics.mean(vals)
        lo, hi = boot_ci(vals)
        ni = statistics.mean(float(r["n_interactions"]) for r in rows)
        vis = statistics.mean(visits_to_best[int(r["pid"])] for r in rows)
        bucket_ndcg[mk] = nd
        if mk == "KOne":
            body.append(r"\midrule")
        body.append(f"{label} & {len(rows):,} & {ni:.0f} & {vis:.1f} & {f4(nd)} & \\mbox{{[{f4(lo)}, {f4(hi)}]}} \\\\")
        num.add(f"StaleBucket{mk}N", f"{len(rows):,}", f"users with staleness_at_best in {label} (cf72e0 cached per-user CSV)")
        num.add(f"StaleBucket{mk}Ndcg", f4(nd), f"mean sampled_ndcg@10 of users with staleness_at_best in {label}")
        num.add(f"StaleBucket{mk}Ci", f"\\mbox{{[{f4(lo)}, {f4(hi)}]}}", "percentile bootstrap 95% CI of the bucket mean (10000 resamples, seed 42)")
        num.add(f"StaleBucket{mk}Visits", f"{vis:.1f}", f"mean participations up to the best round, staleness bucket {label}")
    body += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_staleness_buckets.tex", "\n".join(body),
                "cf72e0 reeval_cached_best_per_user.csv + selected_clients_per_round: NDCG@10 by staleness_at_best bucket (calibrated heads)")
    num.add("StaleBucketSpread", f"{100 * (bucket_ndcg['Zero'] - bucket_ndcg['OverTwenty']):.1f}",
            "NDCG@10 points, staleness bucket 0 minus bucket >20")
    num.add("StaleKZeroMinusRest", signed(bucket_ndcg["Zero"] - bucket_ndcg["KOne"]),
            "NDCG@10, bucket 0 minus all users with k >= 1")
    num.add("StaleSpreadKOne", f"{100 * (bucket_ndcg['OneFive'] - bucket_ndcg['OverTwenty']):.1f}",
            "NDCG@10 points, bucket 1-5 minus bucket >20")
    num.add("StaleSpreadKOneDelta", signed(bucket_ndcg["OneFive"] - bucket_ndcg["OverTwenty"]), "NDCG@10, bucket 1-5 minus bucket >20")
    g1 = np.asarray([float(r["sampled_ndcg@10"]) for r in cached if 1 <= int(r["staleness_at_best"]) <= 5])
    g2 = np.asarray([float(r["sampled_ndcg@10"]) for r in cached if int(r["staleness_at_best"]) > 20])
    diffs = [g1[rng.integers(0, len(g1), len(g1))].mean() - g2[rng.integers(0, len(g2), len(g2))].mean() for _ in range(10000)]
    num.add("StaleSpreadKOneCi", f"\\mbox{{[{float(np.percentile(diffs, 2.5)):+.4f}, {float(np.percentile(diffs, 97.5)):+.4f}]}}",
            "percentile bootstrap 95% CI of (mean 1-5 minus mean >20), independent resampling, 10000 resamples, seed 42")

    def pearson(pairs: List[Tuple[float, float]]) -> float:
        xs = [a for a, _ in pairs]; ys = [b for _, b in pairs]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in pairs)
        return cov / (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5

    def rfmt(r: float) -> str:
        return f"$-{abs(r):.3f}$" if r < 0 else f"{r:.3f}"

    all_pairs = [(float(r["staleness_at_best"]), float(r["sampled_ndcg@10"])) for r in cached]
    num.add("StaleCorr", rfmt(pearson(all_pairs)), "Pearson r(staleness_at_best, sampled_ndcg@10), all users")
    num.add("StaleCorrKOne", rfmt(pearson([pr for pr in all_pairs if pr[0] >= 1])), "Pearson r(staleness_at_best, sampled_ndcg@10), users with k >= 1")
    return stale


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def figures(runs: Dict[str, RunData], stale: List[int], per_user: List[Dict[str, str]],
            cached: List[Dict[str, str]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    colors = {"baseline": "#7f7f7f", "personalized": "#1f77b4", "pfedrec": "#d62728", "adaptive": "#2ca02c"}
    short = {"baseline": "Baseline", "personalized": "Personalized", "pfedrec": "PFedRec", "adaptive": "Adaptive"}

    # 1. training curves
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for key in ("baseline", "personalized", "pfedrec", "adaptive"):
        h = runs[key].history
        xs = sorted(h)
        ys = [h[r]["sampled_ndcg@10"] for r in xs]
        ax.plot(xs, ys, label=short[key], color=colors[key], lw=1.2)
        br = runs[key].best_round
        ax.plot([br], [h[br]["sampled_ndcg@10"]], "o", color=colors[key], ms=4)
    ax.set_xlabel("Round")
    ax.set_ylabel("In-loop NDCG@10 (sampled clients)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_training_curves.pdf", metadata=PDF_META)
    plt.close(fig)

    # 2. per-group bars
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    width = 0.2
    xs = range(4)
    for i, key in enumerate(("baseline", "personalized", "pfedrec", "adaptive")):
        b = runs[key].best
        vals = [metric(b, "ndcg@10")] + [metric(b, "ndcg@10", g) for g in GROUPS]
        ax.bar([x + (i - 1.5) * width for x in xs], vals, width, label=short[key], color=colors[key])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["Overall", "Sparse", "Medium", "Dense"])
    ax.set_ylabel("NDCG@10")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_group_bars.pdf", metadata=PDF_META)
    plt.close(fig)

    # 3. head counterfactuals
    variants = [("cached", "Own calibrated head"), ("alpha_policy_hc", "α-blend with population head"),
                ("global_head", "Population-mean head"), ("global_head_weighted", "Population head, weighted"),
                ("shuffled", "Another user's head"), ("random", "Fresh head per user"),
                ("global_head_cold", "One shared fresh head")]
    vals = [metric(load_json(CF72E0 / f"reeval_{k}_best.json")["metrics"], "ndcg@10") for k, _ in variants]
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.barh([lab for _, lab in variants][::-1], vals[::-1], color="#1f77b4")
    for y, v in enumerate(vals[::-1]):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("NDCG@10 (run cf72e0, same item table and candidates)")
    ax.set_xlim(0, 0.42)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_head_counterfactuals.pdf", metadata=PDF_META)
    plt.close(fig)

    # 4. per-user delta of post-tuning (E=1): clean vs official-code negatives
    base = {r["pid"]: float(r["sampled_ndcg@10"]) for r in cached}
    deltas = {"clean": [], "leaky": []}
    for r in per_user:
        if r["epochs"] == "1":
            deltas[r["pool"]].append(float(r["sampled_ndcg@10"]) - base[r["pid"]])
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4), sharey=True)
    for ax, (pool, lab) in zip(axes, (("clean", "Clean negatives"), ("leaky", "Official-code negatives"))):
        d = [x for x in deltas[pool] if x != 0.0]
        ax.hist(d, bins=40, color="#1f77b4")
        ax.set_title(f"{lab}\n{len(d):,} users changed", fontsize=8)
        ax.set_xlabel("ΔNDCG@10 per user (non-zero only)", fontsize=8)
    axes[0].set_ylabel("Users")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_posttune_delta.pdf", metadata=PDF_META)
    plt.close(fig)

    # 5. staleness at the best round (cf72e0) and first-participation curve
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4))
    axes[0].hist(stale, bins=range(0, max(stale) + 2), color="#d62728")
    axes[0].set_xlabel("Rounds since last participation at the best round")
    axes[0].set_ylabel("Clients")
    cs = runs["pfedrec"].results["cold_starts"]["per_round"]
    cum = [sum(cs[: i + 1]) for i in range(len(cs))]
    axes[1].plot(range(1, len(cum) + 1), cum, color="#d62728")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Clients seen at least once")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_staleness.pdf", metadata=PDF_META)
    plt.close(fig)
    print("  wrote 5 figures")


# ----------------------------------------------------------------------------
def main() -> None:
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    num = Numbers()
    runs = {spec.key: load_run(spec) for spec in RUNS}
    nocal = load_run(PFEDREC_NOCAL)
    centralized = load_json(ROOT / "results" / "centralized" / "bpr_mf_centralized_results.json")
    qu = load_json(POSTTUNE / "reeval_qu_diag_best.json")
    stats = load_json(POSTTUNE / "stats_wilcoxon.json")
    per_user = read_csv(POSTTUNE / "reeval_qu_diag_best_per_user.csv")
    cached = read_csv(CF72E0 / "reeval_cached_best_per_user.csv")
    if len(cached) != 6040:
        raise RuntimeError("cached per-user CSV must have 6040 rows")

    num.add("NClients", "6{,}040", "N")
    num.add("NRounds", str(runs["pfedrec"].cfg["num_rounds"]), "federated_config.num_rounds")
    num.add("ParticipationC", str(runs["pfedrec"].cfg["fraction_train"]), "federated_config.fraction_train")
    num.add("Seed", str(runs["pfedrec"].cfg["run_seed"]), "federated_config.run_seed")
    num.add("PublishedPfedrecNdcg", "0.4436", "PFedRec paper Table 2, ML-1M (Papers/digested/zhang_2023_pfedrec.md)")
    num.add("PublishedPfedrecHr", "0.7326", "PFedRec paper Table 2, ML-1M")
    num.add("PublishedCentralMfNdcg", "0.4137", "PFedRec paper Table 2, centralized MF on ML-1M")
    pfr = runs["pfedrec"].results["_manifest"].get("pfr08_verification", {})
    if pfr:
        num.add("OfficialCodeNdcg", f4(pfr["ref_ndcg"]), "_manifest.pfr08_verification.ref_ndcg (official code, same machine, config unconfirmed)")
        num.add("OfficialCodeHr", f4(pfr["ref_hr"]), "_manifest.pfr08_verification.ref_hr")
    leaky_e1 = metric(qu["metrics_by_cell"]["a0.00_E1_leaky"], "ndcg@10")
    num.add("GapLeakyEOneToPublished", f4(0.4436 - leaky_e1), "published 0.4436 minus E1 official-code cell")
    num.add("GapSharedToPublished", f4(0.4436 - metric(qu["baseline_metrics"], "ndcg@10")), "published 0.4436 minus thesis number")
    acfg = runs["adaptive"].results.get("adaptive_config", {})
    num.add("PrototypeMomentum", str(runs["adaptive"].cfg["prototype_momentum"]), "adaptive federated_config.prototype_momentum")
    num.add("ContrastiveLambda", str(acfg.get("contrastive_lambda", "--")), "adaptive_config.contrastive_lambda")
    num.add("AlphaMin", str(acfg.get("alpha_min", "--")), "adaptive_config.alpha_min")
    num.add("AlphaMax", str(acfg.get("alpha_max", "--")), "adaptive_config.alpha_max")

    print("tables:")
    table_dataset(num)
    table_runs(runs, nocal, num)
    table_main(runs, nocal, centralized, num)
    table_main_groups(runs)
    table_last_round(runs)
    table_pfedrec_rows(runs, nocal, qu, num)
    table_posttune_wilcoxon(stats, per_user, cached, num)
    table_posttune_contrasts(stats, per_user, num)
    table_posttune_cells(qu)
    table_head_counterfactuals(num)
    table_params_comm(runs, num)
    table_hyperparams(runs)
    table_asymmetries(runs)
    table_predictions(num)
    stale = staleness_numbers(runs, cached, num)
    print("figures:")
    figures(runs, stale, per_user, cached)
    num.write(TABLES / "numbers.tex")
    print(f"  wrote tables/numbers.tex ({len(num.items)} macros)")


if __name__ == "__main__":
    main()
