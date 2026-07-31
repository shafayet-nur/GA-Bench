#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_BASE = Path("/path/to/user/sci_ga_paper1/task1_completeness_awq")
SCORES_DIR = PROJECT_BASE / "task1_scores"
DEFAULT_OUT = SCORES_DIR / "inter_annotator"

HUMAN_DIR = Path(
    "/path/to/user/sci_ga_paper1/humman_annotation/humman_annotated_sheets"
)
DEFAULT_HUMAN1 = HUMAN_DIR / "annotation_sheet_1.xlsx"
DEFAULT_HUMAN2 = HUMAN_DIR / "annotation_sheet_2.xlsx"

DEFAULT_SRP = SCORES_DIR / "scores_qwen3_vl_32b_variantB.csv"
DEFAULT_NAIVE = SCORES_DIR / "scores_naive_qwen3_vl_32b_variantB.csv"

HUMAN_REL_COLS = {
    "Relation: Intro->Methods": "intro_to_methods",
    "Relation: Methods->Results": "methods_to_results",
    "Relation: Results->Discussion": "results_to_discussion",
}
HUMAN_REL_MAP = {"traceable": 1.0, "partial": 0.5, "not": 0.0}
AUTO_REL_MAP = {"traceable": 1.0, "partially_traceable": 0.5, "not_traceable": 0.0}
HUMAN_LEVEL_COL = "Completeness Level (0-4)"

def norm_id(s) -> str:
    s = str(s or "").strip().lower().replace("https://doi.org/", "").replace("doi:", "")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")

def load_human(path: Path) -> pd.DataFrame:

    sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    frames = []
    for name, df in sheets.items():
        if name.lower().startswith("01") or "instruction" in name.lower():
            continue
        df.columns = [str(c).strip() for c in df.columns]
        if "doi_folder" not in df.columns:
            continue
        df = df[df["doi_folder"].notna() & (df["doi_folder"].astype(str).str.strip() != "")]
        frames.append(df)
    h = pd.concat(frames, ignore_index=True)
    h["_id"] = h["doi_folder"].map(norm_id)

    keep = ["_id", HUMAN_LEVEL_COL] + list(HUMAN_REL_COLS)
    h = h[[c for c in keep if c in h.columns]].copy()

    h["level"] = pd.to_numeric(h[HUMAN_LEVEL_COL], errors="coerce")

    h = h.sort_values("level").drop_duplicates("_id", keep="first")
    return h

def load_scores(path: Path) -> pd.DataFrame:

    a = pd.read_csv(path, dtype=str)
    if "parse_ok" in a.columns:
        a = a[a["parse_ok"].astype(str).str.lower() != "false"]
    idc = "doi_safe" if "doi_safe" in a.columns else ("doi" if "doi" in a.columns else a.columns[0])
    a["_id"] = a[idc].map(norm_id)
    a["mlevel"] = pd.to_numeric(a.get("level", pd.Series(dtype=str)), errors="coerce")
    return a[["_id", "mlevel"]].dropna()

def _confusion(a, b, labels):

    idx = {v: i for i, v in enumerate(labels)}
    K = len(labels)
    O = np.zeros((K, K))
    for x, y in zip(a, b):
        O[idx[int(x)], idx[int(y)]] += 1
    return O

try:
    from sklearn.metrics import cohen_kappa_score as _skk
except Exception:
    _skk = None

def quadratic_weighted_kappa(a, b, K):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if len(a) == 0:
        return float("nan")
    if _skk is not None:
        return float(_skk(a, b, labels=list(range(K)), weights="quadratic"))
    O = _confusion(a, b, list(range(K)))
    n = O.sum()
    w = np.array([[((i - j) ** 2) / ((K - 1) ** 2) for j in range(K)] for i in range(K)])
    r1 = O.sum(axis=1); r2 = O.sum(axis=0)
    E = np.outer(r1, r2) / n
    den = (w * E).sum()
    return float(1 - (w * O).sum() / den) if den else float("nan")

def cohen_kappa(a, b, K):

    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if len(a) == 0:
        return float("nan")
    if _skk is not None:
        return float(_skk(a, b, labels=list(range(K))))
    O = _confusion(a, b, list(range(K)))
    n = O.sum()
    po = np.trace(O) / n
    r1 = O.sum(axis=1) / n
    r2 = O.sum(axis=0) / n
    pe = float((r1 * r2).sum())
    return float((po - pe) / (1 - pe)) if (1 - pe) else float("nan")

def spearman(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    m = x.notna() & y.notna()
    if m.sum() < 3 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return float("nan")
    return float(np.corrcoef(x[m].rank(), y[m].rank())[0, 1])

def exact_pct(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return round(100 * float((a == b).mean()), 2) if len(a) else float("nan")

def r3(x):
    return round(float(x), 3) if x == x else float("nan")

def main() -> None:
    ap = argparse.ArgumentParser(description="Task 1 inter-annotator agreement + model-vs-annotator validation")
    ap.add_argument("--human1", default=str(DEFAULT_HUMAN1))
    ap.add_argument("--human2", default=str(DEFAULT_HUMAN2))
    ap.add_argument("--srp", default=str(DEFAULT_SRP), help="primary SRP scores CSV")
    ap.add_argument("--naive", default=str(DEFAULT_NAIVE), help="naive baseline scores CSV (variant-matched)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    h1 = load_human(Path(args.human1))
    h2 = load_human(Path(args.human2))

    j = h1.merge(h2, on="_id", suffixes=("_1", "_2"))
    lv = j[["level_1", "level_2"]].dropna()
    n_overlap = len(j)
    n_level = len(lv)

    iaa_rows = []

    a_lv = lv["level_1"].astype(int).values
    b_lv = lv["level_2"].astype(int).values
    iaa_rows.append({
        "target": "completeness_level", "n": int(n_level),
        "exact_pct": exact_pct(a_lv, b_lv),
        "qwk": r3(quadratic_weighted_kappa(a_lv, b_lv, 5)),
        "cohen_kappa": r3(cohen_kappa(a_lv, b_lv, 5)),
        "spearman": r3(spearman(lv["level_1"], lv["level_2"])),
    })

    pooled_a, pooled_b = [], []
    for hcol, short in HUMAN_REL_COLS.items():
        c1, c2 = f"{hcol}_1" if f"{hcol}_1" in j else hcol, f"{hcol}_2" if f"{hcol}_2" in j else hcol

        c1 = hcol + "_1"
        c2 = hcol + "_2"
        if c1 not in j.columns or c2 not in j.columns:
            continue
        s1 = j[c1].astype(str).str.strip().str.lower().map(HUMAN_REL_MAP)
        s2 = j[c2].astype(str).str.strip().str.lower().map(HUMAN_REL_MAP)
        m = s1.notna() & s2.notna()
        ai = (s1[m] * 2).astype(int).values
        bi = (s2[m] * 2).astype(int).values
        iaa_rows.append({
            "target": short, "n": int(m.sum()),
            "exact_pct": exact_pct(ai, bi),
            "qwk": r3(quadratic_weighted_kappa(ai, bi, 3)),
            "cohen_kappa": r3(cohen_kappa(ai, bi, 3)),
            "spearman": r3(spearman(s1[m], s2[m])),
        })
        pooled_a += list(ai); pooled_b += list(bi)

    if pooled_a:
        pa = np.array(pooled_a); pb = np.array(pooled_b)
        iaa_rows.append({
            "target": "relations_pooled", "n": int(len(pa)),
            "exact_pct": exact_pct(pa, pb),
            "qwk": r3(quadratic_weighted_kappa(pa, pb, 3)),
            "cohen_kappa": r3(cohen_kappa(pa, pb, 3)),
            "spearman": r3(spearman(pa, pb)),
        })

    IAA = pd.DataFrame(iaa_rows)
    IAA.to_csv(out / "inter_annotator_agreement.csv", index=False)

    sev_rows = []
    for tag, hh in [("annotator_1", h1), ("annotator_2", h2)]:
        lvls = hh["level"].dropna().astype(int)
        row = {"annotator": tag, "n": int(len(lvls)), "mean_level": r3(lvls.mean())}
        for k in range(5):
            row[f"pct_level_{k}"] = round(100 * float((lvls == k).mean()), 2)
        sev_rows.append(row)
    SEV = pd.DataFrame(sev_rows)
    SEV.to_csv(out / "annotator_severity.csv", index=False)

    srp = load_scores(Path(args.srp))
    naive = load_scores(Path(args.naive))

    def model_vs_levels(model_df, human_levels):

        d = human_levels.merge(model_df, on="_id", how="inner").dropna(subset=["hl", "mlevel"])
        if len(d) < 3:
            return {"n": int(len(d)), "exact_pct": float("nan"), "qwk": float("nan"), "spearman": float("nan")}
        a = d["hl"].round().astype(int).clip(0, 4).values
        b = d["mlevel"].round().astype(int).clip(0, 4).values
        return {
            "n": int(len(d)),
            "exact_pct": exact_pct(a, b),
            "qwk": r3(quadratic_weighted_kappa(a, b, 5)),
            "spearman": r3(spearman(d["hl"], d["mlevel"])),
        }

    a1 = h1[["_id", "level"]].rename(columns={"level": "hl"}).dropna()
    a2 = h2[["_id", "level"]].rename(columns={"level": "hl"}).dropna()

    for tag, hf in [("annotator1", a1), ("annotator2", a2)]:
        rows = []
        for mname, mdf in [("SRP", srp), ("naive", naive)]:
            r = model_vs_levels(mdf, hf); r["method"] = mname; r["reference"] = tag
            rows.append(r)
        pd.DataFrame(rows)[["method", "reference", "n", "exact_pct", "qwk", "spearman"]].to_csv(
            out / f"model_vs_{tag}.csv", index=False)

    cons = j[["_id", "level_1", "level_2"]].dropna().copy()
    cons["hl"] = (cons["level_1"] + cons["level_2"]) / 2.0
    cons_mean = cons[["_id", "hl"]]
    agreed = cons[cons["level_1"] == cons["level_2"]][["_id", "level_1"]].rename(columns={"level_1": "hl"})

    cons_rows = []
    for mname, mdf in [("SRP", srp), ("naive", naive)]:
        r = model_vs_levels(mdf, cons_mean); r["method"] = mname; r["reference"] = "consensus_mean"
        cons_rows.append(r)
        r2 = model_vs_levels(mdf, agreed); r2["method"] = mname; r2["reference"] = "agreed_subset"
        cons_rows.append(r2)
    pd.DataFrame(cons_rows)[["method", "reference", "n", "exact_pct", "qwk", "spearman"]].to_csv(
        out / "model_vs_consensus.csv", index=False)

    def fmt(df):
        return df.to_string(index=False)

    lines = [
        "Task 1 — Inter-Annotator Agreement and Model-vs-Annotator Validation",
        "=" * 70, "",
        f"Annotator 1 labeled : {len(h1)} GAs",
        f"Annotator 2 labeled : {len(h2)} GAs",
        f"Double-labeled (overlap) : {n_overlap} GAs; level filled in both : {n_level}",
        "",
        "1. INTER-ANNOTATOR AGREEMENT (annotator 1 vs annotator 2)",
        "-" * 70,
        fmt(IAA), "",
        "2. ANNOTATOR SEVERITY (mean level and level distribution)",
        "-" * 70,
        fmt(SEV), "",
        "3. MODEL vs EACH ANNOTATOR (completeness level)",
        "-" * 70,
        "annotator 1:",
        fmt(pd.read_csv(out / "model_vs_annotator1.csv")), "",
        "annotator 2:",
        fmt(pd.read_csv(out / "model_vs_annotator2.csv")), "",
        "4. MODEL vs CONSENSUS (mean level; and exact-agreement subset)",
        "-" * 70,
        fmt(pd.read_csv(out / "model_vs_consensus.csv")), "",
        "Notes:",
        "- IAA 'qwk' for level is quadratic-weighted over 0..4; relations over 0..2.",
        "- Severity differences in section 2 indicate whether one annotator is",
        "  systematically stricter, the confound raised in review.",
        "- Model-vs-annotator agreement is reported against each labeler separately",
        "  and against their mean, so no single expert defines the reference.",
        f"- SRP source : {Path(args.srp).name}",
        f"- naive source: {Path(args.naive).name}",
    ]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved -> {out}")

if __name__ == "__main__":
    main()
