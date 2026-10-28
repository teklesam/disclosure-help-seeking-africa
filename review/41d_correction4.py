"""41d_correction4.py - Pre-registration 38, Correction 4: formal help sources (A1) and perpetrator of forced sex (A2).
For every woman in the disclosure files it recomputes, from the raw IR file:
  formal (revised), formal_wide (sensitivity), help-source categories src_* (B8), perp (revised).
Safety check: the OLD formal and perp rules are recomputed first and must reproduce the existing columns exactly.
Then the derived files are backed up (*_pre_corr4.csv.gz) and updated: formal/perp replaced, formal_old/perp_old/formal_wide/src_* added.
"""
import zipfile, io, re, shutil, numpy as np, pandas as pd
from pathlib import Path
R = Path.home() / "Documents/OMS DHS Births Climate Policy"; D = R / "data/derived"
pairs = pd.read_csv(D / "ir_ge_pairs.csv"); pairs = pairs[pairs.usable]
pairs["survey"] = pairs.country + "_" + pairs.years.astype(str).str[:4] + "_" + pairs.ir.str[4:6]
PHYS = [f"d105{c}" for c in "abcdefgj"]; SEX = ["d105h", "d105i", "d105k"]
OLD_FORMAL = re.compile(r"police|gendarm|doctor|medical|health|lawyer|legal|social serv|social work|court|judge|\bngo\b|helpline|hotline", re.I)
NOTFORMAL = re.compile(r"traditional|religious|healer", re.I)
NEW_FORMAL = re.compile(OLD_FORMAL.pattern + r"|\bngos?\b|ngaos|non-?governmental|organi[sz]ation|\bcbo\b|ministry|social affairs|safe space|rescue cent|shelter", re.I)
WIDE_EXTRA = re.compile(r"chief|local authorit|local council|community leader|local leader|proximity", re.I)
CATS = [("src_police", r"police|gendarm"), ("src_health", r"doctor|medical|health|nurse"), ("src_legal", r"lawyer|legal|court|judge"),
        ("src_social_ngo", r"social serv|social work|\bngos?\b|ngaos|non-?governmental|organi[sz]ation|\bcbo\b|ministry|social affairs|safe space|rescue cent|shelter|helpline|hotline"),
        ("src_religious_traditional", r"religious|traditional|healer|priest|imam|pastor"),
        ("src_local_authority", r"chief|local authorit|local council|community leader|local leader|proximity"),
        ("src_partner_family", r"husband.{0,4}partner.{0,3}family|partner'?s? family|in-law|husband'?s? family"),
        ("src_family", r"own family|mother|father|sister|brother|daughter|son\b|relative|family"),
        ("src_friends_neighbours", r"friend|neighbo"),
        ("src_partner", r"current.{0,10}husband|former.{0,10}husband|boyfriend|husband/partner:")]

def norm_id(s): return s.astype(str).str.strip()

W = {f: pd.read_csv(D / f, dtype={"caseid": str}, low_memory=False) for f in ["women_disclosure.csv.gz", "women_disclosure_heldout.csv.gz"]}
rows, log = [], []
for sv in sorted(set(W["women_disclosure.csv.gz"].survey) | set(W["women_disclosure_heldout.csv.gz"].survey)):
    r = pairs[pairs.survey == sv].iloc[0]
    z = zipfile.ZipFile(R / "data/raw/dhs_ir" / r.ir); raw = z.read([n for n in z.namelist() if n.lower().endswith(".dta")][0])
    with pd.io.stata.StataReader(io.BytesIO(raw)) as rd:
        lab = rd.variable_labels(); vl = rd.value_labels(); ll = dict(zip(rd._varlist, rd._lbllist))
    asked = lambda v: v in lab and not (lab[v] or "").lower().startswith("na")
    src = [v for v in lab if re.match(r"^d119[a-z]+$", v) and v != "d119y" and not (lab[v] or "").lower().startswith("na -")]
    d115 = [v for v in lab if re.match(r"^d115[a-z]+$", v) and v != "d115y" and not (lab[v] or "").lower().startswith("na -")]
    cols = [c for c in ["caseid", "v044", "d005", "d119y", "d115y", "d124", "d125", "d127"] + PHYS + SEX + src + d115 if c in lab]
    d = pd.read_stata(io.BytesIO(raw), columns=list(dict.fromkeys(cols)), convert_categoricals=False)
    for c in d.columns:
        if c != "caseid": d[c] = pd.to_numeric(d[c], errors="coerce").astype("float64")
    d = d[(d.get("v044", 1) == 1) & d.d005.notna() & (d.d005 > 0)].copy(); d["caseid"] = norm_id(d.caseid)
    has = lambda c: c in d
    ask = d.d119y.notna()
    f_old = [v for v in src if OLD_FORMAL.search(lab[v] or "") and not NOTFORMAL.search(lab[v] or "")]
    f_new = [v for v in src if NEW_FORMAL.search(lab[v] or "") and not NOTFORMAL.search(lab[v] or "")]
    f_wide = f_new + [v for v in src if v not in f_new and WIDE_EXTRA.search(lab[v] or "")]
    anyeq = lambda vs: d[vs].eq(1).any(axis=1).astype(float) if vs else pd.Series(np.nan, index=d.index)
    out = pd.DataFrame({"survey": sv, "caseid": d.caseid})
    out["formal_old_check"] = np.where(ask, anyeq(f_old), np.nan) if f_old else np.nan
    out["formal"] = np.where(ask, anyeq(f_new), np.nan) if f_new else np.nan
    out["formal_wide"] = np.where(ask, anyeq(f_wide), np.nan) if f_wide else np.nan
    for name, pat in CATS:                                     # first matching category wins, in the order listed
        pass
    assigned = {}
    for v in src:
        l = (lab[v] or "").lower()
        for name, pat in CATS:
            if re.search(pat, l): assigned.setdefault(name, []).append(v); break
        else: assigned.setdefault("src_other", []).append(v)
    for name in [c for c, _ in CATS] + ["src_other"]:
        out[name] = np.where(ask, anyeq(assigned.get(name, [])), np.nan) if assigned.get(name) else np.nan
    # perpetrator
    P = [c for c in PHYS if has(c)]; S = [c for c in SEX if has(c)]
    part_any = d[P + S].isin([1, 2, 3, 4]).any(axis=1)
    np_phys = (d["d115y"] == 0) if has("d115y") else (d[d115].eq(1).any(axis=1) if d115 else pd.Series(False, index=d.index))
    np_sex12 = d["d124"].eq(1) if has("d124") else pd.Series(False, index=d.index)
    sex_ever = d["d125"].eq(1) if has("d125") else pd.Series(False, index=d.index)
    def perp(part, nonp): return np.select([part & nonp, part, nonp], ["both", "partner", "nonpartner"], default="none")
    out["perp_old_check"] = perp(part_any, np_phys | sex_ever | np_sex12)
    if has("d127") and asked("d127"):
        codes = {k: str(x).lower() for k, x in vl.get(ll.get("d127"), {}).items()}
        pcodes = [k for k, x in codes.items() if re.search(r"husband|partner|boyfriend", x)]
        by_partner = sex_ever & d["d127"].isin(pcodes); by_other = sex_ever & ~d["d127"].isin(pcodes)
        rule = "d127 " + ",".join(str(k) for k in sorted(pcodes))
    else:
        by_partner = pd.Series(False, index=d.index); by_other = sex_ever; rule = "d127 not asked: old rule"
    out["perp"] = perp(part_any | by_partner, np_phys | by_other | np_sex12)
    rows.append(out)
    log.append(dict(survey=sv, perp_rule=rule, formal_old=" | ".join(lab[v][:25] for v in f_old), formal_added=" | ".join(lab[v][:25] for v in f_new if v not in f_old),
                    wide_added=" | ".join(lab[v][:25] for v in f_wide if v not in f_new)))
C = pd.concat(rows, ignore_index=True)
assert not C.duplicated(["survey", "caseid"]).any()
pd.DataFrame(log).to_csv(D / "correction4_log.csv", index=False)

for f, w in W.items():
    w["caseid"] = norm_id(w.caseid)
    m = w.merge(C, on=["survey", "caseid"], how="left", suffixes=("", "_c4"), indicator=True)
    assert (m._merge == "both").all(), f"unmatched in {f}"
    # safety check: old rules reproduce the existing columns
    same_formal = ((m.formal.isna() & m.formal_old_check.isna()) | (m.formal == m.formal_old_check)).mean()
    same_perp = (m.perp == m.perp_old_check).mean()
    print(f"{f}: old formal reproduced {100*same_formal:.3f}% | old perp reproduced {100*same_perp:.3f}%")
    assert same_formal == 1 and same_perp == 1, "old rules not reproduced - stop"
    bak = D / f.replace(".csv.gz", "_pre_corr4.csv.gz")
    if not bak.exists(): shutil.copy(D / f, bak)
    m = m.rename(columns={"formal": "formal_old", "perp": "perp_old", "formal_c4": "formal", "perp_c4": "perp"})
    m = m.drop(columns=["_merge", "formal_old_check", "perp_old_check"])
    m.to_csv(D / f, index=False)
    y = m[(m.survey != "SN_2019_8B") & m.sought_any.notna() & (m.vtype != "none") & (m.v012 <= 24)]
    print(f"  young women {len(y):,}: formal {100*y.formal_old.mean():.2f}% -> {100*y.formal.mean():.2f}% (wide {100*y.formal_wide.mean():.2f}%);",
          "perp changed for", int((y.perp != y.perp_old).sum()), "|", y.groupby(["perp_old", "perp"]).size().to_dict())
