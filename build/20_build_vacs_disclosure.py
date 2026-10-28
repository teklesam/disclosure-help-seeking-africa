"""20_build_vacs_disclosure.py - Pre-registration 33: harmonised VACS disclosure file, 8 countries (CDC-derived lifetime indicators; 1 = yes, 2 = no, missing = not a victim).
Output: OMS VACS Climate Violence/data/processed/vacs_disclosure_harmonised.csv
"""
import numpy as np, pandas as pd, re
from pathlib import Path
V = Path.home() / "Documents/OMS VACS Climate Violence/data/raw/vacs/OneDrive_VACS_8-18-2026"
OUT = Path.home() / "Documents/OMS VACS Climate Violence/data/processed/vacs_disclosure_harmonised.csv"
SV = ["tell", "knewplace", "sought", "recvdserv"]
M = {  # country: file, year, weight, psu, age, sex, hh-head sex, rural rule, SV vars, PV vars
 "KE": dict(f="Kenya VACS 2018-19/Kenya VACS 2018-19_Public Release/Kenya 2 VACS Public Use Data Package/Kenya_pubuse_092121.dta", yr=2019, w="SampleWeight", psu="strata", age="q2", sex="sex", hh="h2", rural=None,
            sv=["tell", "knewplace", "sought", "recvdserv"], pv=["pvtell", "PVknewplace", "pvsought", "PVrecvdserv"]),
 "LS": dict(f="Lesotho VACS/Lesotho VACS data_Most recent_From CDC direct/lesotho_pud_rurban.dta", yr=2018, w="individual_weight1", psu="psu", age="Q2", sex="sex", hh="H2", rural=("sett", lambda s: s == 3),
            sv=["tellEver", "knewplaceEver", "soughtEver", "recvdservEver"], pv=["PVTELLEver", "PVknewplaceEver", "PVSOUGHTEver", "PVrecvdservEver"]),
 "MWF": dict(f="_unzipped/Malawi VACS Public Use Data Package - STATA/Malawi VACS Public Use Data Package/Malawi Public Use Data Files/malawi_pubuse_Female_04072020.dta", yr=2013, w="Finalwgt", psu="psu", age="Q2", sex=None, hh="H2",
             rural=("Urbanization", None), sv=["tell", "knewplace", "sought", "recvdserv"], pv=["pvtell", "PVknewplace", "pvsought", "PVrecvdserv"], female=1),
 "MWM": dict(f="_unzipped/Malawi VACS Public Use Data Package - STATA/Malawi VACS Public Use Data Package/Malawi Public Use Data Files/malawi_pubuse_male_04072020.dta", yr=2013, w="Finalwgt", psu="psu", age="Q2", sex=None, hh="H2",
             rural=("Urbanization", None), sv=["tell", "knewplace", "sought", "recvdserv"], pv=["pvtell", "PVknewplace", "pvsought", "PVrecvdserv"], female=0),
 "MZ": dict(f="Mozambique/Mozambique VACS data_Most recent_From CDC direct_2024-07-23/mozambique_pud_rurban.dta", yr=2019, w="sampleweight", psu="Cluster", age="age", sex="Sex", hh="H2", rural=("Type_Name", lambda s: s == "Rural"),
            sv=["svtell", "svknewplace", "svsought", "svrecvdserv"], pv=["pvtell", "pvknewplace", "pvsought", "pvrecvdserv"]),
 "NM": dict(f="Namibia/Namibia VACS_Public Release/namibia_pubuse_05092022.dta", yr=2019, w="SampleWeight", psu="strata", age="Q2", sex="Sex", hh="H2", rural=("strata", lambda s: s.astype(str).str.contains("Rural")),
            sv=["tellEver", "knewplaceEver", "soughtEver", "recvdservEver"], pv=["PVTELLEver", "PVknewplaceEver", "PVSOUGHTEver", "PVrecvdservEver"]),
 "SZ": dict(f="Eswatini 2022/DATASET/eswatini_2024_09-26.dta", yr=2022, w="sampleweight", psu="psu", age="age", sex="sex", hh="H2", rural=("area", None),
            sv=["svtell", "svknewplace", "svsought", "svrecvdserv"], pv=["pvtell", "pvknewplace", "pvsought", "pvrecvdserv"]),
 "TZ": dict(f="Tanzania 2023/Tanzania 2023/TANZANIA_VACS_2024_PUD.dta", yr=2023, w="SAMPLEWEIGHT", psu="PSU", age="AGE", sex="SEX", hh="H2", rural=("AREA", lambda s: s.astype(str).str.upper() == "RURAL"),
            sv=["SVTELL", "SVKNEWPLACE", "SVSOUGHT", "SVRECVDSERV"], pv=["PVTELL", "PVKNEWPLACE", "PVSOUGHT", "PVRECVDSERV"]),
 "ZW": dict(f="Zimbabwe/Zimbabwe VACS Public Release/zimbabwe_pubuse_11212019.dta", yr=2017, w="finalwt", psu="psu", age="q2", sex="sex", hh="h2", rural=("sect", lambda s: s != 50),
            sv=["tell", "knewplace", "sought", "recvdserv"], pv=["pvtell", "PVknewplace", "pvsought", "PVrecvdserv"])}
def yn(s): s = pd.to_numeric(s, errors="coerce"); return np.where(s == 1, 1.0, np.where(s == 2, 0.0, np.nan))
def female(s): s = s.astype(str).str.strip().str.lower(); return np.where(s.isin(["2", "2.0", "female"]), 1.0, np.where(s.isin(["1", "1.0", "male"]), 0.0, np.nan))
def rural_generic(s):
    t = s.astype(str).str.strip().str.lower()
    if t.str.contains("rural").any(): return t.str.contains("rural").astype(float)
    n = pd.to_numeric(s, errors="coerce"); return np.where(n == 2, 1.0, np.where(n == 1, 0.0, np.nan))   # DHS-style 1 urban / 2 rural
rows, log = [], []
for k, m in M.items():
    with pd.io.stata.StataReader(V / m["f"]) as rd: cols = set(rd.variable_labels())
    need = [c for c in [m["w"], m["psu"], m["age"], m.get("sex"), m["hh"]] + m["sv"] + m["pv"] + ([m["rural"][0]] if m["rural"] else []) if c and c in cols]
    need = list(dict.fromkeys(need)); d = pd.read_stata(V / m["f"], columns=need, convert_categoricals=False)
    o = pd.DataFrame({"country": k[:2], "file": k, "year": m["yr"], "w": pd.to_numeric(d[m["w"]], errors="coerce"), "psu": k + "_" + d[m["psu"]].astype(str),
                      "age": pd.to_numeric(d[m["age"]], errors="coerce")})
    o["female"] = m["female"] if "female" in m else female(d[m["sex"]])
    o["hh_female"] = np.where(pd.to_numeric(d[m["hh"]], errors="coerce") == 2, 1.0, np.where(pd.to_numeric(d[m["hh"]], errors="coerce") == 1, 0.0, np.nan)) if m["hh"] in d else np.nan
    if m["rural"]: rc, fn = m["rural"]; o["rural"] = (fn(d[rc]).astype(float) if fn else rural_generic(d[rc])) if rc in d else np.nan
    else: o["rural"] = np.nan
    for pre, vs in (("sv", m["sv"]), ("pv", m["pv"])):
        for nm, v in zip(SV, vs): o[f"{pre}_{nm}"] = yn(d[v]) if v in d else np.nan
    o["sv_victim"] = o.sv_tell.notna().astype(int); o["pv_victim"] = o.pv_tell.notna().astype(int)
    rows.append(o)
    log.append(dict(country=k, n=len(o), sv_victims=int(o.sv_victim.sum()), pv_victims=int(o.pv_victim.sum()), rural_share=round(np.nanmean(o.rural), 2) if o.rural.notna().any() else None,
                    female_share=round(np.nanmean(o.female), 2), sv_tell=round(np.nanmean(o.sv_tell), 2), sv_sought=round(np.nanmean(o.sv_sought), 3), pv_tell=round(np.nanmean(o.pv_tell), 2), pv_sought=round(np.nanmean(o.pv_sought), 3)))
D = pd.concat(rows, ignore_index=True); D["age1824"] = (D.age >= 18).astype(int); D["poly"] = ((D.sv_victim == 1) & (D.pv_victim == 1)).astype(int)
D.to_csv(OUT, index=False); print(pd.DataFrame(log).to_string(index=False)); print("TOTAL", len(D), "| SV victims", D.sv_victim.sum(), "| PV victims", D.pv_victim.sum())
