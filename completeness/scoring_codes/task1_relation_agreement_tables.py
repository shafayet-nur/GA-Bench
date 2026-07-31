#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import cohen_kappa_score as _skk
except Exception:
    _skk = None

PROJECT_BASE = Path("/path/to/user/sci_ga_paper1/task1_completeness_awq")
SCORES_DIR = PROJECT_BASE / "task1_scores"
DEFAULT_OUT = SCORES_DIR / "relation_agreement_tables"

HUMAN_DIR = Path(
    "/path/to/user/sci_ga_paper1/humman_annotation/humman_annotated_sheets"
)
DEFAULT_H1 = HUMAN_DIR / "annotation_sheet_1.xlsx"
DEFAULT_H2 = HUMAN_DIR / "annotation_sheet_2.xlsx"

MODELS = [
    ("Qwen-A", "qwen3_vl_32b", "A"),
    ("Qwen-B", "qwen3_vl_32b", "B"),
    ("Gemma-A", "gemma_3_27b", "A"),
    ("Gemma-B", "gemma_3_27b", "B"),
    ("Mistral-A", "mistral_small_24b", "A"),
    ("Mistral-B", "mistral_small_24b", "B"),
]

HUMAN_REL_COLS = {
    "Relation: Intro->Methods": "im",
    "Relation: Methods->Results": "mr",
    "Relation: Results->Discussion": "rd",
}
AUTO_REL_COLS = {
    "rel_intro_to_methods": "im",
    "rel_methods_to_results": "mr",
    "rel_results_to_discussion": "rd",
}
HUMAN_LEVEL_COL = "Completeness Level (0-4)"
REL_ORDER = ["im", "mr", "rd"]
REL_NAME = {"im": "intro_to_methods", "mr": "methods_to_results", "rd": "results_to_discussion"}

def norm_id(s) -> str:
    s = str(s or "").strip().lower().replace("https://doi.org/", "").replace("doi:", "")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")

def rel_to_score(v) -> float:

    s = str(v or "").strip().lower()
    if s.startswith("trace"):
        return 1.0
    if s.startswith("partial"):
        return 0.5
    if s.startswith("not"):
        return 0.0
    return float("nan")

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
    out = pd.DataFrame({"_id": h["doi_folder"].map(norm_id)})
    out["level"] = pd.to_numeric(h.get(HUMAN_LEVEL_COL), errors="coerce")
    for col, short in HUMAN_REL_COLS.items():
        out[short] = h[col].map(rel_to_score) if col in h.columns else np.nan
    return out.drop_duplicates("_id", keep="first").reset_index(drop=True)

def load_srp(path: Path) -> pd.DataFrame:

    a = pd.read_csv(path, dtype=str)
    if "parse_ok" in a.columns:
        a = a[a["parse_ok"].astype(str).str.lower() != "false"]
    idc = "doi_safe" if "doi_safe" in a.columns else ("doi" if "doi" in a.columns else a.columns[0])
    out = pd.DataFrame({"_id": a[idc].map(norm_id)})
    for col, short in AUTO_REL_COLS.items():
        out[short] = a[col].map(rel_to_score) if col in a.columns else np.nan
    return out.drop_duplicates("_id", keep="first").reset_index(drop=True)

def _confusion(a, b, K):
    O = np.zeros((K, K))
    for x, y in zip(a, b):
        O[int(x), int(y)] += 1
    return O

def qwk(a, b, K):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if len(a) == 0:
        return float("nan")
    if _skk is not None:
        return float(_skk(a, b, labels=list(range(K)), weights="quadratic"))
    O = _confusion(a, b, K)
    n = O.sum()
    w = np.array([[((i - j) ** 2) / ((K - 1) ** 2) for j in range(K)] for i in range(K)])
    r1 = O.sum(axis=1); r2 = O.sum(axis=0)
    E = np.outer(r1, r2) / n
    den = (w * E).sum()
    return float(1 - (w * O).sum() / den) if den else float("nan")

def spearman(x, y):
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y, dtype=float))
    m = x.notna() & y.notna()
    if m.sum() < 3 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return float("nan")
    return float(np.corrcoef(x[m].rank(), y[m].rank())[0, 1])

def exact_pct(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return round(100 * float((a == b).mean()), 2) if len(a) else float("nan")

def r3(x):
    return round(float(x), 3) if x == x else float("nan")

def _pts(df_scores: pd.DataFrame) -> pd.Series:

    return (df_scores[REL_ORDER] * 2).round().astype(int).sum(axis=1)

def _mean(df_scores: pd.DataFrame) -> pd.Series:
    return df_scores[REL_ORDER].mean(axis=1)

def r_metrics(H: pd.DataFrame, M: pd.DataFrame) -> dict:

    mask = H[REL_ORDER].notna().all(axis=1) & M[REL_ORDER].notna().all(axis=1)
    H = H[mask].reset_index(drop=True)
    M = M[mask].reset_index(drop=True)
    n = int(mask.sum())
    base = {"n": n, "R_spearman": float("nan"), "R_qwk": float("nan"),
            "R_exact_pct": float("nan"), "pooled_qwk": float("nan"),
            "pooled_exact_pct": float("nan"), "pooled_spearman": float("nan")}
    if n < 3:
        return base
    h_pts, m_pts = _pts(H), _pts(M)
    hp = (H[REL_ORDER] * 2).round().astype(int).values.ravel()
    mp = (M[REL_ORDER] * 2).round().astype(int).values.ravel()
    return {
        "n": n,
        "R_spearman": r3(spearman(_mean(H), _mean(M))),
        "R_qwk": r3(qwk(h_pts.values, m_pts.values, 7)),
        "R_exact_pct": exact_pct(h_pts.values, m_pts.values),
        "pooled_qwk": r3(qwk(hp, mp, 3)),
        "pooled_exact_pct": exact_pct(hp, mp),
        "pooled_spearman": r3(spearman(hp, mp)),
    }

def per_relation(H: pd.DataFrame, M: pd.DataFrame) -> list:
    rows = []
    for short in REL_ORDER:
        m = H[short].notna() & M[short].notna()
        hs = (H[short][m] * 2).round().astype(int)
        ms = (M[short][m] * 2).round().astype(int)
        rows.append({
            "relation": REL_NAME[short], "n": int(m.sum()),
            "exact_pct": exact_pct(hs.values, ms.values),
            "qwk": r3(qwk(hs.values, ms.values, 3)),
            "spearman": r3(spearman(H[short][m], M[short][m])),
        })
    return rows

def _align(srp: pd.DataFrame, human: pd.DataFrame):

    d = srp.merge(human[["_id"] + REL_ORDER], on="_id", how="inner", suffixes=("_m", "_h"))
    H = pd.DataFrame({s: d[f"{s}_h"] for s in REL_ORDER})
    M = pd.DataFrame({s: d[f"{s}_m"] for s in REL_ORDER})
    return H, M

def main() -> None:
    ap = argparse.ArgumentParser(description="Model-vs-human relation (R) agreement for Tables 2 & 3")
    ap.add_argument("--h1", default=str(DEFAULT_H1))
    ap.add_argument("--h2", default=str(DEFAULT_H2))
    ap.add_argument("--scores-dir", default=str(SCORES_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--primary", default="Qwen-B", help="config used for Table 2")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sdir = Path(args.scores_dir)

    h1 = load_human(Path(args.h1))
    h2 = load_human(Path(args.h2))

    t3_rows, pr_rows = [], []
    srp_cache = {}
    for label, tag, var in MODELS:
        fp = sdir / f"scores_{tag}_variant{var}.csv"
        if not fp.exists():
            print(f"[warn] missing {fp}")
            continue
        srp = load_srp(fp)
        srp_cache[label] = srp
        H, M = _align(srp, h1)
        m = r_metrics(H, M)
        m = {"model": label, **m}
        t3_rows.append(m)
        for r in per_relation(H, M):
            pr_rows.append({"model": label, **r})
        print(f"  {label:9s} vs set1 : R_qwk={m['R_qwk']}  R_exact={m['R_exact_pct']}%  "
              f"R_spearman={m['R_spearman']}  (pooled_qwk={m['pooled_qwk']})  n={m['n']}")

    t3_cols = ["model", "n", "R_spearman", "R_qwk", "R_exact_pct",
               "pooled_spearman", "pooled_qwk", "pooled_exact_pct"]
    pd.DataFrame(t3_rows)[t3_cols].to_csv(out / "table3_R_vs_set1.csv", index=False)
    pd.DataFrame(pr_rows)[["model", "relation", "n", "exact_pct", "qwk", "spearman"]].to_csv(
        out / "per_relation_by_model.csv", index=False)

    primary = args.primary
    if primary not in srp_cache:

        match = [m for m in MODELS if m[0] == primary]
        if not match:
            raise SystemExit(f"unknown --primary {primary}")
        _, tag, var = match[0]
        srp_cache[primary] = load_srp(sdir / f"scores_{tag}_variant{var}.csv")
    srp = srp_cache[primary]

    t2_rows = []

    for ref, hh in [("Annotation Set 1", h1), ("Annotation Set 2", h2)]:
        H, M = _align(srp, hh)
        m = r_metrics(H, M)
        t2_rows.append({"reference": ref, **m})

    hh = h1.merge(h2, on="_id", suffixes=("_1", "_2"))
    d = srp.merge(hh, on="_id", how="inner")
    H1 = pd.DataFrame({s: d[f"{s}_1"] for s in REL_ORDER})
    H2 = pd.DataFrame({s: d[f"{s}_2"] for s in REL_ORDER})
    M = pd.DataFrame({s: d[s] for s in REL_ORDER})

    complete = (H1[REL_ORDER].notna().all(axis=1)
                & H2[REL_ORDER].notna().all(axis=1)
                & M[REL_ORDER].notna().all(axis=1))

    def consensus_row(ref_label, extra_mask):
        msk = complete & extra_mask
        h1m, h2m, mm = _mean(H1[msk]), _mean(H2[msk]), _mean(M[msk])
        cons_mean = (h1m.values + h2m.values) / 2.0
        h1p, h2p, mp = _pts(H1[msk]), _pts(H2[msk]), _pts(M[msk])
        cons_pts = np.round((h1p.values + h2p.values) / 2.0).astype(int)
        n = int(msk.sum())
        if n < 3:
            return {"reference": ref_label, "n": n, "R_spearman": float("nan"),
                    "R_qwk": float("nan"), "R_exact_pct": float("nan"),
                    "pooled_qwk": float("nan"), "pooled_exact_pct": float("nan"),
                    "pooled_spearman": float("nan")}
        return {
            "reference": ref_label, "n": n,
            "R_spearman": r3(spearman(cons_mean, mm.values)),
            "R_qwk": r3(qwk(cons_pts, mp.values, 7)),
            "R_exact_pct": exact_pct(cons_pts, mp.values),
            "pooled_qwk": float("nan"),
            "pooled_exact_pct": float("nan"),
            "pooled_spearman": float("nan"),
        }

    all_true = pd.Series(True, index=d.index)
    agreed_mask = pd.to_numeric(d["level_1"], errors="coerce") == pd.to_numeric(d["level_2"], errors="coerce")
    t2_rows.append(consensus_row("Consensus (mean)", all_true))
    t2_rows.append(consensus_row("Agreed subset", agreed_mask))

    t2_cols = ["reference", "n", "R_spearman", "R_qwk", "R_exact_pct",
               "pooled_spearman", "pooled_qwk", "pooled_exact_pct"]
    pd.DataFrame(t2_rows)[t2_cols].to_csv(out / "table2_R_qwenB.csv", index=False)

    def fmt(df):
        return df.to_string(index=False)

    T3 = pd.read_csv(out / "table3_R_vs_set1.csv")
    T2 = pd.read_csv(out / "table2_R_qwenB.csv")
    PR = pd.read_csv(out / "per_relation_by_model.csv")

    lines = [
        "Task 1 — Model-vs-Human RELATION agreement (R) for Tables 2 and 3",
        "=" * 70, "",
        f"Primary config (Table 2) : {primary}",
        f"Annotation set 1 : {Path(args.h1).name}",
        f"Annotation set 2 : {Path(args.h2).name}", "",
        "R = mean of the 3 relation verdicts (traceable=1, partial=0.5, not=0).",
        "R_spearman uses continuous mean R; R_qwk / R_exact_pct use integer",
        "relation points 0..6 (K=7), the discrete analogue of R.",
        "pooled_* = 3-point per-link agreement (K=3), a cleaner alternative.", "",
        "NOTE: the Direct (naive) baseline has NO relation verdicts, so R is",
        "SRP-only; the Direct half of Tables 2/3 gets no R columns.", "",
        "TABLE 3 — SRP configs vs annotation set 1 (R columns):",
        "-" * 70, fmt(T3), "",
        "TABLE 2 — %s vs four references (R columns):" % primary,
        "-" * 70, fmt(T2), "",
        "Per-relation (each config vs set 1):",
        "-" * 70, fmt(PR), "",
        "How to use:",
        "- For Table 3, add three SRP R columns (R_spearman, R_qwk, R_exact_pct)",
        "  from table3_R_vs_set1.csv next to the existing SRP L columns.",
        "- For Table 2, do the same per reference from table2_R_qwenB.csv.",
        "- If a reviewer prefers a 3-point relation metric, swap in the pooled_*",
        "  columns (defined for the per-set rows; not for the mean references).",
    ]
    (out / "relation_R_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved -> {out}")

if __name__ == "__main__":
    main()
