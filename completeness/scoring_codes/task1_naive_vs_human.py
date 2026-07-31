#!/usr/bin/env python3

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

TEMP_DIR="/path/to/user/sci_ga_paper1/temp_statistics"
OUT_DIR=Path("/path/to/user/sci_ga_paper1/task1_completeness_awq/task1_scores")/"naive_vs_human"
HUMAN_XLSX=Path(TEMP_DIR)/"annotation_sheet.xlsx"
T1DIR=Path("/path/to/user/sci_ga_paper1/task1_completeness_awq/task1_scores")
MODELS=["qwen3_vl_32b","gemma_3_27b","mistral_small_24b"]; VARIANTS=["A","B"]
HEADER_RENAME={"Completeness Level (0-4)":"completeness_level"}

def norm_id(s):
    s=str(s or "").strip().lower().replace("https://doi.org/","").replace("doi:","")
    return re.sub(r"[^a-z0-9]+","_",s).strip("_")
def to_num(x):
    try: return float(x)
    except Exception: return np.nan
def spearman(a,b):
    d=pd.DataFrame({"a":a,"b":b}).apply(pd.to_numeric,errors="coerce").dropna()
    if len(d)<3 or d.a.nunique()<2 or d.b.nunique()<2: return float("nan"),0
    return round(float(np.corrcoef(d.a.rank(),d.b.rank())[0,1]),3),len(d)
def qwk(h,m,k=5):
    d=pd.DataFrame({"h":h,"m":m}).apply(pd.to_numeric,errors="coerce").dropna()
    if len(d)<3: return float("nan")
    h=d.h.round().astype(int).clip(0,k-1); m=d.m.round().astype(int).clip(0,k-1)
    O=np.zeros((k,k))
    for hi,mi in zip(h,m): O[hi,mi]+=1
    W=np.array([[((i-j)**2)/((k-1)**2) for j in range(k)] for i in range(k)])
    E=np.outer(O.sum(1),O.sum(0))/O.sum(); den=(W*E).sum()
    return round(float(1-(W*O).sum()/den),3) if den else float("nan")
def exact(h,m):
    d=pd.DataFrame({"h":h,"m":m}).apply(pd.to_numeric,errors="coerce").dropna()
    return round(100*(d.h.round()==d.m.round()).mean(),1) if len(d) else float("nan")

def load_human():
    sh=pd.read_excel(HUMAN_XLSX,sheet_name=None,dtype=str); fr=[]
    for name,df in sh.items():
        if name.lower().startswith("01") or "instruction" in name.lower(): continue
        df.columns=[str(c).strip() for c in df.columns]; df=df.rename(columns=HEADER_RENAME)
        df=df[df["doi_folder"].notna()&(df["doi_folder"].astype(str).str.strip()!="")]; fr.append(df)
    h=pd.concat(fr,ignore_index=True); h["_id"]=h["doi_folder"].map(norm_id)
    h["hlevel"]=h["completeness_level"].apply(lambda v: np.nan if str(v).strip().lower() in {"","n/a","na","none","nan"} else to_num(v))
    return h[["_id","hlevel"]].dropna()

def load_level(path):
    if not path.exists(): return None
    m=pd.read_csv(path,dtype=str)
    m=m[m.get("parse_ok",pd.Series(["True"]*len(m))).astype(str).str.lower()!="false"]
    idc="doi_safe" if "doi_safe" in m.columns else ("doi" if "doi" in m.columns else m.columns[0])
    m["_id"]=m[idc].map(norm_id); m["mlevel"]=m.get("level",pd.Series(dtype=str)).map(to_num)
    return m[["_id","mlevel"]].dropna()

def main():
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    H=load_human(); rows=[]
    def add(method,kind,path):
        M=load_level(path)
        if M is None: print(f"[skip] {path.name}"); return
        d=H.merge(M,on="_id",how="inner")
        sp,n=spearman(d.hlevel,d.mlevel)
        rows.append([method,kind,n,sp,qwk(d.hlevel,d.mlevel),exact(d.hlevel,d.mlevel)])
    for model in MODELS:
        for v in VARIANTS:
            add(f"{model}_{v}","SRP",T1DIR/f"scores_{model}_variant{v}.csv")
            add(f"{model}_naive_{v}","naive",T1DIR/f"scores_naive_{model}_variant{v}.csv")
    R=pd.DataFrame(rows,columns=["method","kind","n","level_spearman","level_qwk","exact_pct"]).sort_values("level_spearman",ascending=False)
    R.to_csv(OUT_DIR/"srp_vs_naive_vs_human.csv",index=False)

    deltas=[]
    for model in MODELS:
        srp=R[(R.method.str.startswith(model))&(R.kind=="SRP")]
        nai=R[(R.method.str.startswith(model))&(R.kind=="naive")]
        if len(srp) and len(nai):
            deltas.append([model,round(srp.level_spearman.max()-nai.level_spearman.max(),3),
                           round(srp.level_qwk.max()-nai.level_qwk.max(),3)])
    D=pd.DataFrame(deltas,columns=["model","spearman_SRPbest_minus_naive","qwk_SRPbest_minus_naive"])
    D.to_csv(OUT_DIR/"srp_minus_naive_delta.csv",index=False)

    L=["Task 1 — SRP vs Naive vs HUMAN (completeness level 0-4, n=500)","="*55,"",
       R.to_string(index=False),"",
       "SRP-best minus naive (per model):","-"*40,D.to_string(index=False),"",
       "Positive deltas => structured SRP scoring matches human levels better than the",
       "single-prompt naive VLM judge, confirming the core Task 1 claim against humans."]
    (OUT_DIR/"report.txt").write_text("\n".join(str(x) for x in L)+"\n")
    print("\n"+"\n".join(str(x) for x in L)+f"\n\nSaved -> {OUT_DIR}")

if __name__=="__main__":
    main()
