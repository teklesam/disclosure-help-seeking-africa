"""41_build_disclosure_women.py - Climate x disclosure pilot (Pre-reg 25): one row per woman selected for the DV module, DISCOVERY countries, GPS rounds.
Violence (anyone): partner physical/sexual (d105*), non-partner physical (d115* / d117a), non-partner forced sex (d124/d125).
Recent = in the last 12 months: partner d105* codes 1/2 (often/sometimes), non-partner physical d117a 1/2, d124 = 1.
Disclosure cascade (asked of women who ever experienced physical or sexual violence):
  sought_any = 1 - d119y; formal = any d119 source whose LABEL names police / doctor / medical / health / lawyer / legal / social service / court (mapped per file);
  told = sought_any OR d128 (ever told anyone else). Also interview privacy d122a-c, interviewer v028.  No climate data read.
Output: data/derived/women_disclosure.csv.gz
"""
import sys, zipfile, io, re, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path.home() / "Documents/OMS DHS Births Climate Policy"
pairs = pd.read_csv(ROOT / "data/derived/ir_ge_pairs.csv"); pairs = pairs[pairs.usable]
sp = pd.read_csv(ROOT / "data/derived/split_countries.csv"); HELD = "--heldout" in sys.argv
DISC = set(sp[sp.set == ("held-out" if HELD else "discovery")].country)
PHYS = [f"d105{c}" for c in "abcdefgj"]; SEX = ["d105h", "d105i", "d105k"]
BASE = ["caseid", "v001", "v005", "v008", "v016", "v024", "v025", "v012", "v011", "v106", "v190", "v501", "v028", "v044", "d005",
        "d117a", "d124", "d125", "d128", "d119y", "d122a", "d122b", "d122c",
        "v714", "v218", "v157", "v158", "v159", "v743a", "v743b", "v743d", "v744a", "v744b", "v744c", "v744d", "v744e",   # Pre-reg 33 predictors
        "d107", "d110a", "d110b", "d110d", "d113", "d101a", "d101b", "d101c", "d101d", "d101e", "d101f",
        "v467b", "v467c", "v467d", "v467e"]   # Pre-reg 35: barriers to health care (1 = big problem, 2 = not a big problem)
FORMAL = re.compile(r"police|gendarm|doctor|medical|health|lawyer|legal|social serv|social work|court|judge|\bngo\b|helpline|hotline", re.I)   # AUDIT 2026-10-19: \bngo\b (not "Angola")
NOTFORMAL = re.compile(r"traditional|religious|healer", re.I)                                                          # AUDIT: traditional doctor/leader = informal
out, log = [], []
for _, r in pairs.iterrows():
    if r.country not in DISC: continue
    sv = f"{r.country}_{str(r.years)[:4]}_{r.ir[4:6]}"
    f = ROOT / "data/raw/dhs_ir" / r.ir
    if not f.exists(): f = Path.home() / "Downloads" / r.ir
    z = zipfile.ZipFile(f); raw = z.read([n for n in z.namelist() if n.lower().endswith(".dta")][0])
    with pd.io.stata.StataReader(io.BytesIO(raw)) as rd:
        lab = rd.variable_labels(); vl = rd.value_labels(); lbl = dict(zip(rd._varlist, rd._lbllist))
    codes = lambda v: {k: str(x).lower() for k, x in vl.get(lbl.get(v), {}).items()}
    if "d119y" not in lab or "d105a" not in lab: continue
    src = [v for v in lab if re.match(r"^d119[a-z]+$", v) and v != "d119y" and not (lab[v] or "").lower().startswith("na -")]
    formal = [v for v in src if FORMAL.search(lab[v] or "") and not NOTFORMAL.search(lab[v] or "")]
    d115 = [v for v in lab if re.match(r"^d115[a-z]+$", v) and v != "d115y" and not (lab[v] or "").lower().startswith("na -")]
    use = [c for c in BASE + PHYS + SEX + src + d115 + ["d115y"] if c in lab]
    d = pd.read_stata(io.BytesIO(raw), columns=use, convert_categoricals=False)
    for c in use:
        if c != "caseid": d[c] = pd.to_numeric(d[c], errors="coerce").astype("float64")
    d = d[(d.get("v044", 1) == 1) & d.d005.notna() & (d.d005 > 0)].copy()
    if r.country == "ET" and d.v008.max() < 1380: d["v008"] += 92; d["v011"] += 92
    P = [c for c in PHYS if c in d]; S = [c for c in SEX if c in d]
    recency_ok = any("often" in x for x in codes("d105a").values())   # AUDIT: DHS-4 files code d105 as ever (0/1) only -> no past-12m information
    part_ever = d[P + S].isin([1, 2, 3, 4]).any(axis=1); part_12 = d[P + S].isin([1, 2]).any(axis=1)
    np_phys_ever = (d["d115y"] == 0) if "d115y" in d else d[d115].eq(1).any(axis=1) if d115 else pd.Series(False, index=d.index)
    c117 = codes("d117a"); rec117 = [1, 2] + [k for k, x in c117.items() if re.search(r"once|one time|seldom", x)]   # AUDIT: AO "once", MZ 2011 "seldom"
    np_phys_12 = d["d117a"].isin(rec117) if "d117a" in d else pd.Series(False, index=d.index)
    np_sex_12 = d["d124"].eq(1) if "d124" in d else pd.Series(False, index=d.index)
    np_sex_ever = d["d125"].eq(1) if "d125" in d else pd.Series(False, index=d.index)
    d["victim_ever"] = (part_ever | np_phys_ever | np_sex_ever | np_sex_12).astype(int)
    d["victim_12m"] = (part_12 | np_phys_12 | np_sex_12).astype(float) if recency_ok else np.nan
    d["partner_12m"] = part_12.astype(float) if recency_ok else np.nan
    part_old = d[P + S].isin([3]).any(axis=1); np_old = np_phys_ever & ~np_phys_12                      # AUDIT: violence before the last 12 months
    d["recent_only"] = ((d["victim_12m"] == 1) & ~part_old & ~np_old).astype(float) if recency_ok else np.nan
    asked = d.d119y.notna()
    d["sought_any"] = np.where(asked, (d.d119y == 0).astype(float), np.nan)
    d["formal"] = np.where(asked, d[formal].eq(1).any(axis=1).astype(float) if formal else np.nan, np.nan)
    told_else = d["d128"].eq(1) if "d128" in d else pd.Series(False, index=d.index)
    d["told"] = np.where(asked, ((d.d119y == 0) | told_else).astype(float), np.nan)
    d["d128_present"] = int("d128" in d)
    d["private"] = np.where(d[[c for c in ["d122a", "d122b", "d122c"] if c in d]].isin([1, 2, 3]).any(axis=1), 0, 1) if any(c in d for c in ["d122a", "d122b", "d122c"]) else np.nan
    # ---- Pre-reg 33 predictors ----
    def anyof(cols, vals): c = [x for x in cols if x in d]; return d[c].isin(vals).any(axis=1).astype(float).where(d[c].notna().any(axis=1)) if c else pd.Series(np.nan, index=d.index)
    part_phys = d[[c for c in PHYS if c in d]].isin([1, 2, 3, 4]).any(axis=1); part_sex = d[[c for c in SEX if c in d]].isin([1, 2, 3, 4]).any(axis=1) if S else pd.Series(False, index=d.index)
    any_phys = part_phys | np_phys_ever; any_sex = part_sex | np_sex_ever | np_sex_12
    d["vtype"] = np.select([any_phys & any_sex, any_phys, any_sex], ["both", "physical", "sexual"], default="none")
    part_any = part_phys | part_sex; np_any = np_phys_ever | np_sex_ever | np_sex_12
    d["perp"] = np.select([part_any & np_any, part_any, np_any], ["both", "partner", "nonpartner"], default="none")
    d["severe"] = (d["d107"] == 1).astype(float).where(d["d107"].notna()) if "d107" in d else np.nan
    d["injury"] = anyof(["d110a", "d110b", "d110d"], [1])
    d["alcohol"] = (d["d113"] == 1).astype(float).where(d["d113"].notna()) if "d113" in d else np.nan
    ctl = [c for c in ["d101a", "d101b", "d101c", "d101d", "d101e", "d101f"] if c in d]
    d["controlling3"] = (d[ctl].eq(1).sum(axis=1) >= 3).astype(float).where(d[ctl].notna().any(axis=1)) if ctl else np.nan
    d["working"] = (d["v714"] == 1).astype(float).where(d["v714"].notna()) if "v714" in d else np.nan
    d["children"] = d["v218"] if "v218" in d else np.nan
    d["media"] = anyof(["v157", "v158", "v159"], [2, 3])           # at least once a week / almost every day
    dm = [c for c in ["v743a", "v743b", "v743d"] if c in d]
    d["decisions2"] = (d[dm].isin([1, 2]).sum(axis=1) >= 2).astype(float).where(d[dm].notna().any(axis=1)) if dm else np.nan   # alone or jointly
    d["justify"] = anyof(["v744a", "v744b", "v744c", "v744d", "v744e"], [1])
    d["survey"], d["country"] = sv, r.country
    keep = ["survey", "country", "caseid", "v001", "v005", "d005", "v008", "v016", "v024", "v025", "v012", "v011", "v106", "v190", "v501", "v028",
            "victim_ever", "victim_12m", "partner_12m", "recent_only", "sought_any", "formal", "told", "d128_present", "private",
            "vtype", "perp", "severe", "injury", "alcohol", "controlling3", "working", "children", "media", "decisions2", "justify",
            "v467b", "v467c", "v467d", "v467e"]
    out.append(d[[c for c in keep if c in d]])
    log.append(dict(survey=sv, women=len(d), victims=int(d.victim_ever.sum()), victims_12m=int(d.victim_12m.sum()), asked=int(asked.sum()),
                    recency_ok=recency_ok, rec117=str(rec117), formal_sources=" | ".join(f"{v}:{lab[v][:22]}" for v in formal), d128=int("d128" in d)))
    print(f"{sv}: {len(d):,} DV women; recency_ok={recency_ok}; ever victims {d.victim_ever.sum():,}; past-12m {d.victim_12m.sum():,}; asked help-seeking {asked.sum():,}; formal sources {len(formal)}", flush=True)
W = pd.concat(out, ignore_index=True); W.to_csv(ROOT / f"data/derived/women_disclosure{'_heldout' if HELD else ''}.csv.gz", index=False)
pd.DataFrame(log).to_csv(ROOT / f"data/derived/women_disclosure{'_heldout' if HELD else ''}_build_log.csv", index=False)
print(f"TOTAL {len(W):,} women, {W.survey.nunique()} surveys, {W.country.nunique()} countries; past-12m victims asked help-seeking: {int((W.victim_12m.eq(1) & W.sought_any.notna()).sum()):,}")
