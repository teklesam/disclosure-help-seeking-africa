"""41c_add_hh_head.py - Pre-reg 37: sex of household head (v151) for every woman in the disclosure files, for the harmonised DHS-VACS pool.
Reads only caseid and v151 from the same IR files and the same survey naming as 41_build_disclosure_women.py; does not modify existing derived files.
v151: 1 = male, 2 = female head. Output: data/derived/women_hhhead.csv.gz (survey, caseid, hh_female); every woman must match exactly once.
"""
import zipfile, io, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path.home() / "Documents/OMS DHS Births Climate Policy"
pairs = pd.read_csv(ROOT / "data/derived/ir_ge_pairs.csv"); pairs = pairs[pairs.usable]
pairs["survey"] = pairs.country + "_" + pairs.years.astype(str).str[:4] + "_" + pairs.ir.str[4:6]
W = pd.concat([pd.read_csv(ROOT / f"data/derived/{f}", usecols=["survey", "caseid"], dtype={"caseid": str}) for f in ["women_disclosure.csv.gz", "women_disclosure_heldout.csv.gz"]])
out = []
for sv in sorted(W.survey.unique()):
    r = pairs[pairs.survey == sv]; assert len(r) == 1, sv; r = r.iloc[0]
    f = ROOT / "data/raw/dhs_ir" / r.ir
    if not f.exists(): f = Path.home() / "Downloads" / r.ir
    z = zipfile.ZipFile(f); raw = z.read([n for n in z.namelist() if n.lower().endswith(".dta")][0])
    d = pd.read_stata(io.BytesIO(raw), columns=["caseid", "v151"], convert_categoricals=False)
    d["v151"] = pd.to_numeric(d.v151, errors="coerce")
    d = d.assign(survey=sv, hh_female=np.where(d.v151 == 2, 1.0, np.where(d.v151 == 1, 0.0, np.nan)))[["survey", "caseid", "hh_female"]]
    out.append(d)
H = pd.concat(out); H["caseid"] = H.caseid.astype(str)
assert not H.duplicated(["survey", "caseid"]).any(), "duplicate caseid"
M = W.merge(H, on=["survey", "caseid"], how="left", indicator=True)
print("women:", len(W), "| matched:", int((M._merge == "both").sum()), "| hh_female missing:", int(M.hh_female.isna().sum()), "| share female head:", round(M.hh_female.mean(), 3))
assert (M._merge == "both").all(), "unmatched women"
M[["survey", "caseid", "hh_female"]].to_csv(ROOT / "data/derived/women_hhhead.csv.gz", index=False)
