"""71_build_latex_manuscript.py - Lancet-style LaTeX manuscript (data, model, results; Figure 1, Table 1).
Every number is read from result files. Writes main.tex into the Overleaf working copy given as argv[1], and numbers_used.json for independent checking.
Hyperlinks: red boxes = internal links (figures, tables, sections, equations); green boxes = citations; blue = URLs.
"""
import json, sys, shutil, numpy as np, pandas as pd
from pathlib import Path
OL = Path(sys.argv[1]); R = Path.home() / "Documents/OMS DHS Births Climate Policy"; T = R / "outputs/tables"; V3 = T / "P33_youth_v3"
(OL / "figures").mkdir(exist_ok=True)
shutil.copy(R / "outputs/figures/disclosure_v3/fig1_map_help_seeking_lancet.png", OL / "figures/fig1_map_help_seeking.png")   # 61b_fig1_lancet_style.R
for _f in ["fig_cascade_forest_lancet.png", "fig2_forest_injury_severe_lancet.png", "fig3_forest_both_former_lancet.png"]:   # cascade: 61d_fig_cascade_forest_lancet.R                                          # 61c_fig2_3_factor_forests_lancet.R
    shutil.copy(R / "outputs/figures/disclosure_v3" / _f, OL / "figures" / _f)
N = {}                                                                     # every number used, for the independent check
def put(k, v): N[k] = v; return v
f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:.2f}"
def ci1(r): return f"{f1(r.pct)}\\% (95\\% CI {f1(r.lo)}--{f1(r.hi)})"
def orc(r, lab=None):
    if lab: put(lab, (round(r.OR, 2), round(r.OR_lo, 2), round(r.OR_hi, 2)))
    return f"{f2(r.OR)} (95\\% CrI {f2(r.OR_lo)}--{f2(r.OR_hi)})"
# ---------------- results ----------------
C = pd.read_csv(V3 / "A8_5_dhs_cascade_ci.csv"); VC = pd.read_csv(V3 / "A8_5_vacs_cascade_ci.csv")
cas = lambda o, reg=None, lvl="region": C[(C.outcome == o) & (C.level == lvl) & ((C.region == reg) if reg else True)].iloc[0]
At, As, Af = cas("told", lvl="all"), cas("sought_any", lvl="all"), cas("formal", lvl="all")
for k, r in [("told_all", At), ("sought_all", As), ("formal_all", Af)]: put(k, (round(r.pct, 1), round(r.lo, 1), round(r.hi, 1)))
REG = ["West", "Central", "East", "Southern"]
reg = {(o, g): cas(o, g) for o in ["told", "sought_any", "formal"] for g in REG}
for (o, g), r in reg.items(): put(f"{o}_{g}", (round(r.pct, 1), round(r.lo, 1), round(r.hi, 1)))
cty = C[C.level == "country"]
def rng(o):
    x = cty[cty.outcome == o]; a, b = x.loc[x.pct.idxmin()], x.loc[x.pct.idxmax()]; return a, b
NAMES = {"MR": "Mauritania", "NM": "Namibia", "SN": "Senegal", "BU": "Burundi"}
vac = lambda t, s, st: VC[(VC.level == "all VACS countries") & (VC.type == t) & (VC.sex == s) & (VC.step == st)].iloc[0]
P = pd.read_csv(V3 / "A3_pooled_numpyro.csv"); RP = pd.read_csv(V3 / "A3_replication_numpyro.csv")
PART = {"injury", "severe", "alcohol", "controlling3"}
def pp(t, o="sought_any", src="DHS", model=None):
    m = model or ("partner" if t in PART else "main"); return P[(P.source == src) & (P.model == m) & (P.outcome == o) & (P.term == t)].iloc[0]
S = pd.read_csv(V3 / "southern_africa_pooled_exploratory.csv")
TR = pd.read_csv(V3 / "A5_trends_numpyro.csv"); TRpp = pd.read_csv(V3 / "A5_trends_pp_numpyro.csv"); A82 = pd.read_csv(V3 / "A8_2_trends_numpyro.csv")
CT = pd.read_csv(V3 / "A5_country_trends.csv"); CT = CT[(CT.term == "yr10") & (CT.model == "adjusted") & (CT.outcome == "sought_any")]
tr = lambda o: TR[(TR.model == "adjusted") & (TR.outcome == o)].iloc[0]
cls = P[P.source == "DHS"].groupby("outcome")["class"].value_counts().unstack().fillna(0).astype(int)
rep = RP[RP.source == "DHS"]
# country weights for injury (sought) from the NumPyro posterior medians of tau and omega
CE = pd.read_csv(V3 / "A2_country_estimates.csv"); x = CE[(CE.source == "DHS") & (CE.model == "partner") & (CE.outcome == "sought_any") & (CE.term == "injury")].copy(); pi = pp("injury")
x["wc"] = 1 / (x.se**2 + pi.tau**2); Pr = x.groupby("region").wc.sum(); Wr = 1 / (1 / Pr + pi.omega**2); x["weight"] = 100 * x.region.map(Wr / Wr.sum()) * x.wc / x.region.map(Pr)
maxw = put("max_country_weight_injury", round(x.weight.max(), 1)); maxw_c = x.loc[x.weight.idxmax(), "country"]
n_div = int(P.divergences.sum()); n_divfits = int((P.divergences > 0).sum())
put("pooled_estimates", len(P)); put("rhat_max", round(P.rhat_max.max(), 3)); put("ess_min", int(P.ess_min.min())); put("divergences", n_div)
# sample sizes (from the analysis scripts' logs / harmonised files)
dhs = pd.concat([pd.read_csv(R / f"data/derived/{f}.csv.gz", usecols=["survey", "country", "sought_any", "vtype", "v012"]) for f in ["women_disclosure", "women_disclosure_heldout"]])
dhs = dhs[(dhs.survey != "SN_2019_8B") & dhs.sought_any.notna() & (dhs.vtype != "none") & (dhs.v012 <= 24)]
nD = put("n_dhs", len(dhs)); nS = put("n_surveys", dhs.survey.nunique()); nC = put("n_countries", dhs.country.nunique())
nmr = dhs.groupby("country").survey.nunique(); put("n_multiround_countries", int((nmr >= 2).sum()))
vac_df = pd.read_csv(Path.home() / "Documents/OMS VACS Climate Violence/data/processed/vacs_disclosure_harmonised.csv")
nV = put("n_vacs", len(vac_df)); nSV = put("n_vacs_sv", int(vac_df.sv_victim.sum())); nPV = put("n_vacs_pv", int(vac_df.pv_victim.sum()))
nVany = put("n_vacs_any", int(((vac_df.sv_victim == 1) | (vac_df.pv_victim == 1)).sum()))
fmt = lambda n: f"{n:,}".replace(",", "\\,")
# ---------------- key estimates ----------------
inj, sev, both, form = pp("injury"), pp("severe"), pp("vtypeboth"), pp("maritalformer")
injt, sevt, botht, formt = pp("injury", "told"), pp("severe", "told"), pp("vtypeboth", "told"), pp("maritalformer", "told")
alc, wrk, edu, wq, urb, age = pp("alcohol"), pp("working"), pp("educsecondary+"), pp("wealthq"), pp("urban"), pp("agegrp")
npt = pp("perpnonpartner"); injf = pp("injury", "formal")
_cv = CE[CE.source == "VACS"].groupby(["model", "outcome"]).country.nunique(); vac_lo = put("vacs_model_countries_min", int(_cv.min())); vac_hi = put("vacs_model_countries_max", int(_cv.max()))
n_dm = put("dhs_main_countries", CE[(CE.source == "DHS") & (CE.model == "main") & (CE.outcome == "sought_any")].country.nunique())
n_dp = put("dhs_partner_countries", CE[(CE.source == "DHS") & (CE.model == "partner") & (CE.outcome == "sought_any")].country.nunique())
VNAME = {"KE": "Kenya", "LS": "Lesotho", "MW": "Malawi", "MZ": "Mozambique", "NM": "Namibia", "SZ": "Eswatini", "TZ": "Tanzania", "ZW": "Zimbabwe"}
ov = sorted(set(vac_df.country) & set(dhs.country)); n_ov = put("vacs_dhs_overlap", len(ov)); not_ov = [VNAME[c] for c in sorted(set(vac_df.country) - set(dhs.country))]
P37 = pd.read_csv(R / "outputs/tables/P37_harmonised/P37_pooled.csv").set_index(["outcome", "term"])
_g = vac_df[(vac_df.female == 1) & (vac_df.age >= 15) & (vac_df.age <= 24) & ((vac_df.sv_victim == 1) | (vac_df.pv_victim == 1))]; nG = put("n_vacs_girls_1524", len(_g))
_c37 = pd.read_csv(R / "outputs/tables/P37_harmonised/P37_country_estimates.csv"); n37d = put("p37_dhs_countries", _c37[_c37.source == "DHS"].country.nunique()); n37v = put("p37_vacs_countries", _c37[_c37.source == "VACS"].country.nunique())
def p37(o, t, k): r = P37.loc[(o, t)]; put(f"p37_{o}_{t}_{k}", [round(r[f"OR_{k}"], 2), round(r[f"OR_{k}_lo"], 2), round(r[f"OR_{k}_hi"], 2)]); return f"{f2(r[f'OR_{k}'])} (95\\% CrI {f2(r[f'OR_{k}_lo'])}--{f2(r[f'OR_{k}_hi'])})"
def p37s(o, t, k): r = P37.loc[(o, t)]; return f"{f2(r[f'OR_{k}'])} ({f2(r[f'OR_{k}_lo'])}--{f2(r[f'OR_{k}_hi'])})"
mdx = lambda x: f"{x:.2f}".replace(".", "\\md{}")
TERM37 = [("both", "Both physical and sexual violence (vs one type)"), ("hh_female", "Female household head (vs male)"), ("rural", "Rural residence (vs urban)"), ("age", "Age (per year)")]
OUT37 = [("sought_any", "Sought any help"), ("told", "Told anyone")]
DEC = {"agree": "Agree", "differ": "Differ", "uncertain": "Uncertain"}
cellx = lambda r, k: f"{mdx(r[f'OR_{k}'])} ({mdx(r[f'OR_{k}_lo'])}--{mdx(r[f'OR_{k}_hi'])})"
s1rows = []
for o, olab in OUT37:
    s1rows.append(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{olab}}}}} \\\\")
    for t, tlab in TERM37:
        r = P37.loc[(o, t)]; s1rows.append(f"\\quad {tlab} & {cellx(r, 'DHS')} & {cellx(r, 'VACS')} & {cellx(r, 'ratio_VACS_DHS')} & {cellx(r, 'pooled')} & {DEC[r.agreement]} \\\\")
tableS1 = "\n".join(s1rows)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("manuscript_appendix", Path(__file__).with_name("manuscript_appendix.py")); APP = _ilu.module_from_spec(_spec); _spec.loader.exec_module(APP)
P38d = R / "outputs/tables/P38_review"
T38 = pd.read_csv(P38d / "P38_pooled.csv")
def p38(a, m, t, o="sought_any"): return T38[(T38.analysis == a) & (T38.model == m) & (T38.outcome == o) & (T38.term == t)].iloc[0]
def rdt(m, t):
    r = p38("B5_rd", m, t); v = [100 * np.log(x) for x in (r.OR, r.OR_lo, r.OR_hi)]; put(f"rd_{t}", [round(x, 1) for x in v]); return v
B1 = pd.read_csv(P38d / "P38_B1_prior_sensitivity.csv"); b1 = lambda m, t: B1[(B1.model == m) & (B1.term == t) & (B1.omega_prior_sd == 0.5)].iloc[0]
PW = pd.read_csv(P38d / "P38_B9_population_weighted_cascade.csv"); pw = lambda o: PW[(PW.outcome == o) & (PW.weighting == "population") & (PW.region == "All")].iloc[0]
for _o in ["told", "sought_any", "formal"]: put(f"popw_{_o}", round(pw(_o).pct, 1))
HS = pd.read_csv(P38d / "P38_B8_help_sources.csv"); hs = lambda k: put(f"src_{k}", round(HS[(HS.source == k) & (HS.region == "All")].pct.iloc[0], 1))
MC = pd.read_csv(P38d / "P38_B12_model_checks.csv"); OS = pd.read_csv(P38d / "P38_B6_onestage.csv").set_index("term")
PPd = pd.read_csv(P38d / "P38_B12_prior_predictive.csv").iloc[0]; TGx = pd.read_csv(P38d / "P38_B12_togo_exclusion.csv").set_index("outcome").loc["sought_any"]
_m2 = MC[MC.model == "M2"]; _m0 = MC[MC.model == "M0"]
n_m0_misfit = int((_m0.pB_chi2 < 0.05).sum()); n_terms = len(_m2)
_pb = _m2[["pB_chi2", "pB_sd", "pB_max", "pB_min"]].values.ravel(); _pb_ok = sorted(x for x in _pb if x >= 0.05)
n_m1_better = int(((MC.model == "M1") & (MC.elpd_diff_vs_M2 > 2 * MC.se_diff_vs_M2)).sum()); max_delpd = MC[(MC.model == "M1") & (MC.elpd_diff_vs_M2 > 2 * MC.se_diff_vs_M2)].elpd_diff_vs_M2.max()
TRp = pd.read_csv(V3 / "A5_pooled_trends.csv"); tr6 = TRp[(TRp.scope == "S2: DHS phase >= 6") & (TRp.model == "adjusted") & (TRp.outcome == "sought_any") & (TRp.term == "yr10")].iloc[0]
_inj = pd.concat([pd.read_csv(R / f"data/derived/{f}.csv.gz", usecols=["survey", "sought_any", "vtype", "v012", "perp", "injury"]) for f in ["women_disclosure", "women_disclosure_heldout"]])
_inj = _inj[(_inj.survey != "SN_2019_8B") & _inj.sought_any.notna() & (_inj.vtype != "none") & (_inj.v012 <= 24) & _inj.perp.isin(["partner", "both"])]
inj_miss = put("injury_missing_pct", round(100 * _inj.injury.isna().mean(), 1))
n_d127_old = int(pd.read_csv(R / "data/derived/correction4_log.csv").perp_rule.str.startswith("d127 not").sum())
girl = pp("female", "sought", src="VACS", model="SV"); girl_r = pp("female", "recvdserv", src="VACS", model="SV")
for k, r in [("or_injury", inj), ("or_severe", sev), ("or_both", both), ("or_former", form), ("or_injury_told", injt), ("or_severe_told", sevt),
             ("or_both_told", botht), ("or_former_told", formt), ("or_alcohol", alc), ("or_working", wrk), ("or_secondary", edu), ("or_wealth", wq), ("or_urban", urb),
             ("or_age", age), ("or_nonpartner", npt), ("or_injury_formal", injf), ("or_girls_sought", girl), ("or_girls_received", girl_r)]:
    put(k, (round(r.OR, 2), round(r.OR_lo, 2), round(r.OR_hi, 2)))
ts, tt_, tf = tr("sought_any"), tr("told"), tr("formal")
vt = A82[A82.model.str.startswith("A8.2a")].iloc[0]; sb = A82[A82.model.str.contains("base covariates")].iloc[0]; ss = A82[A82.model.str.contains("severity")].iloc[0]
tpp = TRpp[TRpp.outcome == "sought_any"].iloc[0]
for k, r in [("trend_sought", ts), ("trend_told", tt_), ("trend_formal", tf), ("trend_victim", vt), ("trend_partner_base", sb), ("trend_partner_sev", ss)]:
    put(k, (round(r.OR, 2), round(r.OR_lo, 2), round(r.OR_hi, 2)))
put("trend_sought_pp", (round(tpp.pp, 1), round(tpp.pp_lo, 1), round(tpp.pp_hi, 1)))
ndec = put("n_countries_decline", int((CT.est < 0).sum())); ntr = put("n_countries_trend", len(CT))
put("rep_same_sign", round(100 * rep.same_sign.mean())); put("rep_both_credible", round(100 * rep.both_credible.mean()))
for o in ["sought_any", "told", "formal"]: put(f"class_{o}", {k: int(v) for k, v in cls.loc[o].items()})
sv = lambda st, s: vac("SV", s, st)
# ---------------- Table 1 ----------------
KEYS = [("injury", "Injury from partner violence (vs none)\\textsuperscript{a}"), ("severe", "Severe physical violence (vs less severe or none)\\textsuperscript{a}"),
        ("vtypeboth", "Physical and sexual violence (vs physical only)"), ("maritalformer", "Formerly in union (vs currently in union)"),
        ("alcohol", "Partner drinks alcohol (vs does not)\\textsuperscript{a}"), ("working", "Currently working (vs not)"),
        ("educsecondary+", "Secondary or higher education (vs none)"), ("wealthq", "Household wealth (per quintile)"), ("urban", "Urban residence (vs rural)")]
cell = lambda o, l, h: ("\\begin{tabular}[t]{@{}c@{}}" + (f"\\textbf{{{f2(o)}}}" if (l > 1 or h < 1) else f2(o)) + f"\\\\ ({f2(l)}--{f2(h)})\\end{{tabular}}")
rows = []
for t, lab in KEYS:
    r = pp(t); s = S[(S.model == ("partner" if t in PART else "main")) & (S.term == t)].iloc[0]
    rows.append(f"{lab} & {cell(r.OR_West, r.OR_West_lo, r.OR_West_hi)} & {cell(r.OR_Central, r.OR_Central_lo, r.OR_Central_hi)} & {cell(r.OR_East, r.OR_East_lo, r.OR_East_hi)} & {cell(s.OR, s.lo, s.hi)} & {cell(r.OR, r.OR_lo, r.OR_hi)} \\\\")
    N[f"table1_{t}"] = [round(v, 2) for v in [r.OR_West, r.OR_Central, r.OR_East, s.OR, r.OR]]
table1 = "\n".join(rows)
ncw = CE[(CE.source == "DHS") & (CE.model == "main") & (CE.outcome == "sought_any")].groupby("region").country.nunique().to_dict()
S_n = int(S.n.max()); S_ev = int(S.events.max())
tlo, thi = rng("told"); slo, shi = rng("sought_any"); flo, fhi = rng("formal")
G = lambda r: f"{f1(r.pct)}\\% in {NAMES.get(r.country, r.country)}"

import re
_po = pd.read_csv(V3 / "A3_pooled_numpyro.csv"); _po = _po[(_po.source == "DHS") & (_po.outcome == "sought_any")]
cas = pd.read_csv(V3 / "A8_5_dhs_cascade_ci.csv").query("level == 'all'").set_index("outcome").n
PO2 = {(r.model, r.term): {"countries": int(r.countries), "tau": float(r.tau)} for r in _po.itertuples()}
md2 = lambda x: f"{x:.2f}"
_in_appendix = [False]
def lancet(tex):
    """Lancet house style for the body text (between \\begin{document} and the bibliography):
    (1) mid-dot decimals (32.9 -> 32\\md{}9), never inside URLs, labels, references, citations, file names or lengths;
    (2) parentheses nested inside parentheses become square brackets, outside mathematics;
    (3) equations are cited as 'equation n', without the automatic parentheses of \\eqref."""
    a = tex.index(r"\begin{document}"); b = tex.index(r"\begin{thebibliography}")
    head, body, tail = tex[:a], tex[a:b], tex[b:]
    body = body.replace(r"eq~\eqref{", r"equation~\ref{").replace(r"~\eqref{", r"~\ref{")
    prot = re.compile(r"\\(?:url|href|label|ref|eqref|cite|includegraphics(?:\[[^\]]*\])?|bibitem)\{[^}]*\}"   # command arguments
                      r"|\\\\\[[^\]]*\]"                                                                     # \\[0.3em]
                      r"|\\(?:setlength|hspace|vspace)\{[^}]*\}(?:\{[^}]*\})?")                               # lengths
    parts, last = [], 0
    for m in prot.finditer(body):
        parts.append((False, body[last:m.start()])); parts.append((True, m.group(0))); last = m.end()
    parts.append((False, body[last:]))
    num = re.compile(r"(?<![\w\\.])\d+(?:\.\d+)+(?!\d|em|cm|pt|ex|mm|in\b)")
    body = "".join(t if keep else num.sub(lambda m: m.group(0).replace(".", r"\md{}"), t) for keep, t in parts)
    math = re.compile(r"\$[^$]*\$|\\\[.*?\\\]|\\begin\{(equation|align)\*?\}.*?\\end\{\1\*?\}", re.S)
    depth = 0
    def scan(t):
        nonlocal depth
        r = []
        for ch in t:
            if ch == "(":
                depth += 1; r.append("[" if depth >= 2 else "(")
            elif ch == ")":
                r.append("]" if depth >= 2 else ")"); depth = max(depth - 1, 0)
            else:
                r.append(ch)
        return "".join(r)
    out, last = [], 0
    for m in math.finditer(body):
        out.append(scan(body[last:m.start()])); out.append(m.group(0)); last = m.end()
    out.append(scan(body[last:]))
    if depth != 0: raise ValueError(f"unbalanced parentheses in the text (depth {depth} at the end): check the last edited paragraph")
    if "\\section*{Supplementary appendix}" in tail and not _in_appendix[0]:
        k = tail.index("\\section*{Supplementary appendix}"); _in_appendix[0] = True
        app = lancet("\\begin{document}" + tail[k:].replace("\\end{document}", "\\begin{thebibliography}\\end{document}", 1))
        tail = tail[:k] + app[len("\\begin{document}"):].replace("\\begin{thebibliography}\\end{document}", "\\end{document}", 1)
        _in_appendix[0] = False
    return head + "".join(out) + tail

# author line kept in a local, untracked file (authors_local.tex); placeholder otherwise
AUTHORS = (Path(__file__).with_name("authors_local.tex").read_text().strip() if Path(__file__).with_name("authors_local.tex").exists() else r"\todo{Authors to confirm}")

SRC_TABLE = r"""\begin{table}[!htbp]
\centering
\begin{threeparttable}
\caption{\textbf{How the DHS and VACS data are used}}
\label{tab:sources}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{>{\raggedright\arraybackslash}p{2.7cm}>{\raggedright\arraybackslash}p{6.3cm}>{\raggedright\arraybackslash}p{6.3cm}}
\toprule
 & DHS & VACS \\
\midrule
Respondents & """ + fmt(nD) + r""" women aged 15--24 years who reported physical or sexual violence; """ + str(nS) + r""" surveys, """ + str(nC) + r""" countries in West, Central, East and Southern Africa, 2004--2025 (""" + str(N['n_multiround_countries']) + r""" with two to five rounds) & """ + fmt(nV) + r""" girls and boys aged 13--24 years, """ + fmt(nVany) + r""" of whom reported violence; eight countries in East and Southern Africa, one round each, 2013--2023 \\
\addlinespace
Questions & Physical violence since age 15 and sexual violence; told anyone, sought help and from whom (lifetime) & Sexual and physical violence asked separately; told anyone, knew where to get help, sought help, received services (lifetime), for each type \\
\addlinespace
Survey design & Domestic violence module weight; clustering by PSU & Survey weight, normalised within country; clustering by PSU (approximated by strata in Kenya and Namibia) \\
\addlinespace
Cascade (equation~\ref{eq:ci}) & Three steps by country, region and overall (figures~\ref{fig:map} and~\ref{fig:cascade}) & Four steps by sex and type of violence, all countries combined \\
\addlinespace
Country models (equations~\ref{eq:logit}--\ref{eq:sandwich}) & Main model (all women, """ + str(n_dm) + r""" countries) and partner-violence model (""" + str(n_dp) + r""" countries); survey-round fixed effects; at least 300 women and 30 events & Separate models for sexual and physical violence; sex, age, rural residence, female household head, both types of violence; one round, so no round effects; at least 150 respondents and 15 events (""" + str(vac_lo) + "--" + str(vac_hi) + r""" countries) \\
\addlinespace
Combining countries (equation~\ref{eq:hier}) & Countries within West, Central and East Africa; Southern Africa in a separate exploratory model & Countries within East and Southern Africa \\
\addlinespace
Replication and trends & Discovery and held-out countries; trends in """ + str(N['n_multiround_countries']) + r""" countries with repeated rounds & Not possible: too few countries per half, one round per country \\
\addlinespace
Harmonised pooled analysis (equation~\ref{eq:harm}) & Young women aged 15--24 years; told anyone and sought any help; age, rural residence, female household head, both types of violence (""" + str(n37d) + r""" countries) & Girls aged 15--24 years who reported violence (""" + fmt(nG) + r"""); same outcomes and factors (""" + str(n37v) + r""" countries) \\
\addlinespace
Findings informed & Cascade among young women; factors associated with help-seeking; agreement across regions; trends & Differences between girls and boys; sexual versus physical violence; ages 13--14, which DHS does not cover \\
\bottomrule
\end{tabular}
\begin{tablenotes}\footnotesize
\item DHS=Demographic and Health Surveys. VACS=Violence Against Children and Youth Surveys. PSU=primary sampling unit. Percentages were never pooled across surveys; associations for four common factors were pooled in the harmonised analysis (appendix table~\ref{tab:harm}).
\end{tablenotes}
\end{threeparttable}
\end{table}"""

tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,margin=2.4cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{newtxtext}
\usepackage{microtype}
\usepackage{amsmath}
\usepackage{newtxmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{threeparttable}
\usepackage{longtable}
\usepackage[switch]{lineno}% continuous line numbers for review
\renewcommand\linenumberfont{\normalfont\tiny\color{gray}}
\newcommand*\patchAmsMathEnvironmentForLineno[1]{%
  \expandafter\let\csname old#1\expandafter\endcsname\csname #1\endcsname
  \expandafter\let\csname oldend#1\expandafter\endcsname\csname end#1\endcsname
  \renewenvironment{#1}{\linenomath\csname old#1\endcsname}{\csname oldend#1\endcsname\endlinenomath}}
\newcommand*\patchBothAmsMathEnvironmentsForLineno[1]{\patchAmsMathEnvironmentForLineno{#1}\patchAmsMathEnvironmentForLineno{#1*}}
\AtBeginDocument{\patchBothAmsMathEnvironmentsForLineno{equation}\patchBothAmsMathEnvironmentsForLineno{align}\linenumbers}
\usepackage[font=small,labelfont=bf,labelsep=colon]{caption}
\usepackage{setspace}
\usepackage[super,sort&compress,comma]{natbib}
\usepackage{xcolor}
% Red boxes = internal links (figures, tables, sections, equations); green boxes = citations; blue boxes = web links.
\usepackage[colorlinks=false,pdfborder={0 0 1.2},linkbordercolor={1 0 0},citebordercolor={0 0.65 0},urlbordercolor={0 0.3 1},bookmarks=true]{hyperref}
\hypersetup{pdftitle={Help-seeking after violence among adolescents and young people in sub-Saharan Africa},pdfauthor={Samuel Tekle et al.}}
\setstretch{1.25}
\setlength{\parskip}{0.35em}
\newcommand{\todo}[1]{\textcolor{red}{[#1]}}
\newcommand{\md}{\ifmmode{\cdot}\else\textperiodcentered\fi}% Lancet decimal point
\title{\textbf{Who seeks help after violence among adolescents and young people in sub-Saharan Africa? A multi-country analysis of 64 Demographic and Health Surveys and eight Violence Against Children and Youth Surveys}}
\author{""" + AUTHORS + r"""\\[0.3em]
\small\parbox{0.9\textwidth}{\centering Oxford Martin Programme on Climate and Child Health; Accelerate Research Hub; University of Oxford \todo{affiliations to confirm}}}
\date{\small Draft for co-authors, October 2026. Correspondence: \todo{corresponding author}}
\begin{document}
\maketitle

\section*{Summary}
\noindent\todo{Summary to be written once the main text is final (Lancet structure: Background, Methods, Findings, Interpretation, Funding; up to 300 words). The previous version is saved in the archive folder of this Overleaf project.}

\section*{Research in context}
\noindent\textbf{Evidence before this study} We searched PubMed on 19 October 2026 with 12 structured queries combining disclosure and help-seeking, violence, DHS and VACS (appendix). Multi-country DHS analyses describe reporting among women of all ages,\cite{palermo2014} and VACS analyses describe a disclosure cascade among children and young people,\cite{pereira2020} mostly with country-by-country estimates. Pooled DHS analyses of 10 to 54 countries\cite{aboagye2023,ayebeng2024,goodson2021,stiller2025} combined all countries in one model, which assumes the same associations everywhere. We found no study that tested whether predictors of help-seeking among young people agree across African regions, replicated them in independent sets of countries, or described trends across survey rounds. \todo{Literature review to be completed.}

\noindent\textbf{Added value of this study} Combining two survey programmes in 33 countries, we show that the factors associated with help-seeking (injury, severity, poly-victimisation and being formerly in union) are shared across regions and replicate in independent countries, whereas socioeconomic resources are not associated, and that help-seeking among young women has not risen.

\noindent\textbf{Implications of all the available evidence} Services reach a small, severe part of the violence that young people experience. Programmes should create earlier, low-threshold routes to help, including for boys, rather than assume that help-seeking will rise with education or wealth.

\section{Introduction}\label{sec:intro}
\noindent\todo{Outline for drafting; literature led by DM. The previous prose is saved in the archive folder of this Overleaf project. Target about 450 words in four paragraphs.}

\noindent\textbf{Paragraph 1. The burden, and why help after violence matters}
\begin{itemize}\setlength{\itemsep}{1pt}
\item Violence against children and young people is common: an estimated one billion children experience violence each year;\cite{hillis2016} intimate partner violence is most prevalent among young women in sub-Saharan Africa;\cite{sardinha2022} adolescents in poor urban settings report high levels of partner and non-partner violence.\cite{decker2014}
\item Consequences for health: incident HIV infection,\cite{jewkes2010} depression and early pregnancy; violence prevention is framed by the INSPIRE strategies.\cite{who2016}
\item Help is time-critical: post-exposure prophylaxis and emergency contraception after sexual violence,\cite{who2013} and protection and justice depend on disclosure. Links to Sustainable Development Goal targets 5.2 and 16.2.\cite{un2015}
\end{itemize}

\noindent\textbf{Paragraph 2. What is known about disclosure and help-seeking}
\begin{itemize}\setlength{\itemsep}{1pt}
\item Most survivors never reach formal services: across 24 countries, 40\% of women disclosed violence and 7\% reported to a formal source;\cite{palermo2014} in sub-Saharan Africa, 35--39\% of women with partner violence sought help, mostly from family, and about 3\% from formal institutions.\cite{aboagye2023,goodson2021,stiller2025}
\item Among children and young people (VACS), formal help-seeking and service receipt are rare and lower among boys.\cite{sumner2015,pereira2020,velloza2022,denhard2024,okova2025}
\item Help-seeking as a process (recognition, decision, choice of source), shaped by individual, relational and structural factors;\cite{liang2005} models specific to young people.\cite{rickwood2005}
\end{itemize}

\noindent\textbf{Paragraph 3. Gaps}
\begin{itemize}\setlength{\itemsep}{1pt}
\item Pooled analyses combine countries in one model, which assumes the same associations everywhere;\cite{aboagye2023,goodson2021,ayebeng2024} whether the factors associated with help-seeking are shared across African regions is unknown, and findings are rarely replicated.
\item Young people (15--24 years) are rarely analysed separately, boys appear only in VACS, and trends across survey rounds have not been described.
\item Reported levels depend on survey methods, so cross-survey comparisons need care.\cite{cullen2023,palermo2014}
\end{itemize}

\noindent\textbf{Paragraph 4. Aim} (keep as drafted)
\begin{itemize}\setlength{\itemsep}{1pt}
\item We asked: (1) how many young survivors tell someone, seek help and reach formal services, and how this differs by region; (2) which factors are associated with help-seeking, and whether they are the same across regions; and (3) whether help-seeking is changing over time.
\end{itemize}

\section{Methods}\label{sec:methods}
\subsection{Data}\label{sec:data}
\textbf{DHS.} DHS are nationally representative household surveys. In the domestic violence module, one randomly selected woman per household is interviewed in private about physical violence since age 15 and sexual violence by partners and by anyone else.\cite{croft2018} We included all rounds from sub-Saharan African countries with the module and the help-seeking questions (""" + str(nS) + r""" surveys, """ + str(nC) + r""" countries, 2004--2025; """ + str(N['n_multiround_countries']) + r""" countries had two to five rounds) and analysed women aged 15--24 years who reported any physical or sexual violence (n=""" + fmt(nD) + r""").

\textbf{VACS.} VACS are nationally representative household surveys of girls and boys aged 13--24 years, with separate questions for sexual and physical violence.\cite{nguyen2019} We used eight countries with comparable indicators (Kenya 2019, Lesotho 2018, Malawi 2013, Mozambique 2019, Namibia 2019, Eswatini 2022, Tanzania 2023, Zimbabwe 2017; n=""" + fmt(nV) + r""", of whom """ + fmt(nVany) + r""" reported violence: """ + fmt(nSV) + r""" sexual and """ + fmt(nPV) + r""" physical).

\textbf{Role of each survey.} DHS is the primary source. It covers """ + str(nC) + r""" countries in all four regions over repeated rounds, and gives the cascade, the factors associated with help-seeking, the regional comparison, the replication and the trends. VACS adds what DHS cannot (boys, younger adolescents, and sexual and physical violence asked separately), in East and Southern Africa (appendix table~\ref{tab:sources}). We never pooled percentages across the two surveys, which sample different populations and word the questions differently; we pooled associations only in a pre-specified harmonised analysis (equation~\ref{eq:harm}). Seven of the eight VACS countries also have DHS surveys (all except """ + ", ".join(not_ov) + r""").

\textbf{Outcomes.} DHS asked women who reported violence whether they had ever sought help, from whom, and whether they had told anyone. We defined three outcomes: \emph{told anyone} (sought help or told someone, where asked); \emph{sought any help}; and \emph{sought formal help} (police, health workers, lawyers or courts, social services, non-governmental and women's organisations, government offices for women or social affairs, safe spaces or shelters, and helplines). Religious and traditional leaders were informal; chiefs and local authorities were formal only in a sensitivity analysis. We read and audited the source labels of every survey (appendix table~\ref{tab:coding}). In VACS we used the derived indicators \emph{told anyone}, \emph{knew where to get help}, \emph{sought help} and \emph{received services}, separately for sexual and physical violence. All help-seeking items refer to the respondent's lifetime.

\textbf{Predictors.} We pre-specified predictors from the help-seeking framework\cite{liang2005} and earlier survey studies:\cite{palermo2014,pereira2020} injury from partner violence (bruises; eye injuries, sprains, dislocations or burns; or wounds, broken bones, broken teeth or other serious injury), severe physical violence (DHS summary of being kicked or dragged, strangled or burnt, or threatened with a weapon), partner alcohol use and three or more controlling behaviours (asked about partners only, so modelled among women with partner violence); type of violence (physical, sexual, both); perpetrator (partner, non-partner, both; forced sex was attributed using the person who forced the first sexual act, where this was asked); violence in the past 12 months; age in years; education; household wealth quintile; urban residence; marital status; employment; three or more children; weekly media exposure; participation in household decisions; agreement that wife-beating is justified; and privacy of the interview (not interrupted by the husband or another adult). VACS models included sex, age in years, rural residence, a female household head and experiencing both sexual and physical violence.

\subsection{Statistical model}\label{sec:model}
\textbf{Prevalence.} For each outcome $y$ and group we estimated the survey-weighted proportion $\hat p=\sum_i w_i y_i/\sum_i w_i$, where $w_i$ is the domestic violence weight. Proportions near zero (formal help) have skewed sampling distributions, so 95\% CIs were computed on the logit scale,
\begin{equation}\label{eq:ci}
\operatorname{logit}(\hat p)\pm 1.96\,\frac{\widehat{\mathrm{SE}}(\hat p)}{\hat p(1-\hat p)},
\end{equation}
and back-transformed, with $\widehat{\mathrm{SE}}(\hat p)$ from Taylor linearisation allowing for clustering within primary sampling units (PSUs).\cite{lumley2004} Strata were not declared, which makes intervals slightly conservative. DHS weights sum to each survey's sample size, so pooled proportions weight surveys by sample size; we also report proportions weighted by each country's female population aged 15--24 years.\cite{worldpop2025} VACS weights were normalised within country.

\textbf{Country models.} Sampling, questionnaires and services differ between countries, so we used a two-stage individual participant data approach:\cite{riley2010,burke2017} we first fitted a separate model in each country $c$, which respects each survey's design.\cite{campos2023} For respondent $i$ in survey round $s(i)$,
\begin{equation}\label{eq:logit}
\operatorname{logit}\Pr(Y_{i}=1\mid \mathbf{x}_{i}) = \alpha_{s(i)} + \mathbf{x}_{i}^{\top}\boldsymbol\beta_{c},
\end{equation}
where $\alpha_{s(i)}$ are survey-round fixed effects, so that $\boldsymbol\beta_c$ compares women interviewed in the same round. Parameters maximised the weighted pseudo-log-likelihood
\begin{equation}\label{eq:pll}
\ell(\boldsymbol\beta_c)=\sum_i w_i\left[y_i\log\pi_i+(1-y_i)\log(1-\pi_i)\right],
\end{equation}
which yields estimates representative of the surveyed population.\cite{lumley2004} Standard errors used the cluster-robust sandwich estimator
\begin{equation}\label{eq:sandwich}
\widehat{\mathbf V}(\hat{\boldsymbol\beta}_c)=\frac{G}{G-1}\,\mathbf A^{-1}\Big(\sum_{g=1}^{G}\mathbf u_g\mathbf u_g^{\top}\Big)\mathbf A^{-1},\qquad \mathbf u_g=\sum_{i\in g}w_i(y_i-\pi_i)\mathbf x_i,
\end{equation}
where $g$ indexes the $G$ PSUs and $\mathbf A$ is the weighted information matrix, so that inference allows for correlation within communities. We fitted models with fixest.\cite{berge2018} Pre-specified rules required at least 300 respondents and 30 events per country and discarded estimates with standard errors above 5 (separation). Injury was missing for """ + f"{inj_miss}" + r"""\% of women with partner violence, who were kept with a missing-indicator term. All predictors entered together, so each odds ratio is mutually adjusted; because education, wealth, urban residence and age precede the other predictors, we also fitted a reduced model with these four only.\cite{westreich2013} VACS models were fitted separately for sexual and physical violence among those reporting each type, with one intercept per country (one round), the VACS predictors and an inclusion rule of 150 respondents and 15 events; rural residence, not recorded in Kenya, was omitted there.

\textbf{Combining countries.} We combined the country estimates $\hat\beta_{ck}$ for predictor $k$, with standard errors $s_{ck}$, in a Bayesian hierarchical model of countries within regions:
\begin{equation}\label{eq:hier}
\hat\beta_{ck}\sim\mathcal N(\theta_{ck},s_{ck}^2),\qquad \theta_{ck}=\mu_{r(c)k}+\tau_k z_{c},\qquad \mu_{rk}=\mu_k+\omega_k u_{r},\qquad z_c,u_r\sim\mathcal N(0,1),
\end{equation}
with priors $\mu_k\sim\mathcal N(0,1)$ and $\tau_k,\omega_k\sim\text{Half-Normal}(0,0.5)$ on the log-odds scale. $\mu_k$ is the pooled log odds ratio, $\mu_{rk}$ the regional means, and $\tau_k$ and $\omega_k$ the between-country and between-region standard deviations. The model shrinks each country towards its regional mean, and each regional mean towards the pooled mean, in proportion to their precision. We used a Bayesian estimator because between-region variance cannot be estimated reliably from three or four regions by moment-based methods; weakly informative priors stabilise it.\cite{gelman2006} The non-centred parametrisation avoids the funnel geometry that arises when $\omega_k$ is near zero. We sampled the posterior with the No-U-Turn sampler\cite{hoffman2014} in NumPyro\cite{phan2019} (four chains of 2000 warm-up and 2000 draws; target acceptance 0.995) and report posterior medians with 95\% credible intervals (CrIs) of $\exp(\mu_k)$ and $\exp(\mu_{rk})$. We fitted the model separately to DHS (regions: West, Central and East Africa) and VACS (East Africa: Kenya, Malawi, Mozambique, Tanzania, Zimbabwe; Southern Africa: Eswatini, Lesotho, Namibia).

Country $c$ in region $r$ carries approximately the weight
\begin{equation}\label{eq:weights}
\lambda_c=\frac{W_r}{\sum_{r'}W_{r'}}\cdot\frac{w_c}{\sum_{c'\in r}w_{c'}},\qquad w_c=\frac{1}{s_c^2+\tau^2},\qquad W_r=\Big(\frac{1}{\sum_{c\in r}w_c}+\omega^2\Big)^{-1},
\end{equation}
so large, precise countries count more, but the variance components stop any one country dominating.

\textbf{Bayesian inference.} We followed a Bayesian workflow.\cite{gelman2020,gabry2019} The priors are weakly informative: $\mu_k\sim\mathcal N(0,1)$ allows pooled odds ratios between 0.14 and 7.1, and the half-normal(0.5) priors allow regional or country odds ratios to differ up to about 2.7-fold. Prior predictive simulation confirmed that they admit all plausible values (appendix figure~\ref{fig:priorpred}). Convergence required $\hat R<1.01$,\cite{vehtari2021} bulk effective sample size above 400 and E-BFMI above 0.3. Posterior predictive checks compared observed and replicated discrepancies $T$ (a $\chi^2$ statistic and the standard deviation, maximum and minimum of the country estimates) through Bayesian $p$-values, $p_B=\Pr\{T(y^{\text{rep}},\theta)\ge T(y,\theta)\mid y\}$; values below 0.05 or above 0.95 indicate misfit.\cite{gelman1996} We compared the primary model (M2, countries within regions) with a common-effect model (M0) and a model without regions (M1) by the deviance information criterion (DIC),\cite{spiegelhalter2002} the widely applicable information criterion (WAIC)\cite{watanabe2010} and leave-one-country-out cross-validation (PSIS-LOO),\cite{vehtari2017} each computed from the likelihood marginal over the country effect. We also report 95\% prediction intervals for a new country\cite{riley2011} and sensitivity to the prior on $\omega_k$.

\textbf{Agreement across regions.} For each predictor we computed (i) the posterior probability that all regional means share the sign of the pooled mean, $\Pr\{\operatorname{sign}(\mu_{rk})=\operatorname{sign}(\tilde\mu_k)\ \forall r\}$, and (ii) a test of between-region heterogeneity, $Q_k=\sum_r \hat W_r(\hat\mu_{rk}-\bar\mu_k)^2\sim\chi^2_{R-1}$, from DerSimonian--Laird regional estimates\cite{dersimonian1986} with Holm correction.\cite{holm1979} A predictor was consistent across regions if (i) exceeded 0.9 and (ii) was not significant, region-specific if (ii) was significant, and uncertain otherwise.

\textbf{Replication.} Before any outcome analysis, DHS countries were randomly split (seed 20261015) into a discovery and a held-out set. A finding replicated if its credible interval excluded one, with the same sign, in both sets.

\textbf{Trends.} In DHS countries with at least two rounds, we added survey year to equation~\eqref{eq:logit} as $\gamma_c t_i$, with $t_i=(\text{year}_i-2015)/10$, so that $\exp(\gamma_c)$ is the odds ratio per decade, and pooled $\gamma_c$ with equation~\eqref{eq:hier}. To test whether trends reflected changes in who reports violence, we estimated the trend in past-year victimisation and the help-seeking trend among women with partner violence with further adjustment for severity and injury. A linear probability model gave the change in percentage points.

\textbf{Harmonised pooled analysis of DHS and VACS.} To test whether the surveys agree, and to pool them where they do, we restricted a pre-registered secondary analysis to what both measure: young women aged 15--24 years with violence (DHS) and girls of the same age (VACS); told anyone and sought any help (in VACS, about any type of violence); and age, rural residence, female household head and both types of violence. Equation~\ref{eq:logit} was fitted for each country and survey programme, and the estimates $\hat\beta_{jk}$ for country--programme $j$ were combined as
\begin{equation}\label{eq:harm}
\hat\beta_{jk}\sim\mathcal N(\theta_{jk},s_{jk}^2),\qquad \theta_{jk}=\mu_k+\delta_k x_j+\omega_k u_{r(j)}+\tau_k z_j,\qquad x_j=\begin{cases}-\tfrac12 & \text{DHS}\\ \phantom{-}\tfrac12 & \text{VACS},\end{cases}
\end{equation}
with $\delta_k\sim\mathcal N(0,1)$ and the other terms as in equation~\ref{eq:hier}, over all four regions. Because $x_j$ is centred, $\exp(\mu_k)$ is the pooled odds ratio with the two programmes weighted equally, $\exp(\mu_k\mp\delta_k/2)$ are the DHS and VACS odds ratios, and $\exp(\delta_k)$ is their ratio. The surveys agreed if the 95\% CrI of $\exp(\delta_k)$ included one and both programme-specific estimates had the same sign with posterior probability above 0\md{}9, differed if the CrI excluded one, and were otherwise uncertain. We interpreted the pooled odds ratio only where they agreed, and repeated the analysis with $\delta_k=0$ and in countries with both surveys.

\textbf{Southern Africa.} No Southern African country met the inclusion rule, so we fitted one exploratory model pooling Lesotho, Namibia and South Africa with survey fixed effects (table~\ref{tab:predictors}).

\textbf{Sensitivity analyses.} Added after an internal methodological review and pre-registered: crude odds ratios; the reduced socio-demographic model; women whose violence all occurred in the 12 months before the interview, so that lifetime help-seeking refers to recent violence; adjusted prevalence ratios (Poisson regression with robust standard errors)\cite{zou2004} and adjusted risk differences (linear probability models), pooled in the same way; one-stage models fitting all countries together, survey-weighted with fixed effects and as a mixed model with random slopes by country;\cite{bates2015} generalised variance inflation factors;\cite{fox1992} population-weighted proportions; and a wider definition of formal help (appendix).

\textbf{Pre-registration and software.} The analysis plan, all amendments (restriction to ages 15--24 to align with VACS; continuous age; the Bayesian implementation; four coding corrections; the sensitivity analyses) and their results were registered with time-stamped SHA-256 hashes before each analysis was run (appendix). Analyses used R 4.5 (survey, fixest, data.table, lme4, car) and Python 3 (NumPyro, JAX, ArviZ); the pooled estimates were reproduced independently with brms and Stan. Code is publicly available at \url{https://github.com/teklesam/disclosure-help-seeking-africa}.

\textbf{Role of the funding source.} The funders had no role in study design, data analysis, interpretation or writing.

\section{Results}\label{sec:results}
\subsection{How many young survivors seek help}\label{sec:cascade}
Of """ + fmt(nD) + r""" young women who experienced violence, """ + ci1(At) + r""" told anyone, """ + ci1(As) + r""" sought any help and """ + ci1(Af) + r""" sought formal help (figure~\ref{fig:cascade}). Weighted by population rather than sample size, the figures were """ + f"{pw('told').pct:.1f}" + r"""\%, """ + f"{pw('sought_any').pct:.1f}" + r"""\% and """ + f"{pw('formal').pct:.1f}" + r"""\% (appendix table~\ref{tab:popw}). Help came mostly from family (""" + f"{hs('src_family'):.0f}" + r"""\%), friends or neighbours (""" + f"{hs('src_friends_neighbours'):.0f}" + r"""\%) and the partner's family (""" + f"{hs('src_partner_family'):.0f}" + r"""\%); """ + f"{hs('src_police'):.1f}" + r"""\% went to the police and """ + f"{hs('src_health'):.1f}" + r"""\% to a health worker (appendix table~\ref{tab:helpsources}). Seeking any help ranged from """ + G(slo) + r""" to """ + G(shi) + r""", and formal help from """ + G(flo) + r""" to """ + G(fhi) + r""" (figure~\ref{fig:map}). Regions differed little in seeking any help (West """ + f"{f1(reg[('sought_any','West')].pct)}\\%" + r""", Central """ + f"{f1(reg[('sought_any','Central')].pct)}\\%" + r""", East """ + f"{f1(reg[('sought_any','East')].pct)}\\%" + r""", Southern """ + f"{f1(reg[('sought_any','Southern')].pct)}\\%" + r""") but substantially in formal help: """ + "; ".join(f"{g} " + f"{f1(reg[('formal',g)].pct)}\\% ({f1(reg[('formal',g)].lo)}--{f1(reg[('formal',g)].hi)})" for g in REG) + r""".

In VACS, """ + f"{f1(sv('tell','girls').pct)}\\% ({f1(sv('tell','girls').lo)}--{f1(sv('tell','girls').hi)})" + r""" of girls and """ + f"{f1(sv('tell','boys').pct)}\\% ({f1(sv('tell','boys').lo)}--{f1(sv('tell','boys').hi)})" + r""" of boys who experienced sexual violence told someone; """ + f"{f1(sv('sought','girls').pct)}\\%" + r""" and """ + f"{f1(sv('sought','boys').pct)}\\%" + r""" sought help, and """ + f"{f1(sv('recvdserv','girls').pct)}\\%" + r""" and """ + f"{f1(sv('recvdserv','boys').pct)}\\%" + r""" received services.

\begin{figure}[p]
\centering
\includegraphics[width=\textwidth,height=0.80\textheight,keepaspectratio]{figures/fig1_map_help_seeking.png}
\caption{\textbf{Help-seeking after violence among young women aged 15--24 years in """ + str(nC) + r""" sub-Saharan African countries}\\
(A) Survey-weighted percentage of young women who sought any help after physical or sexual violence, all DHS rounds pooled (DHS 2004--2025; """ + fmt(nD) + r""" women); numbers are DHS rounds per country and black outlines mark the eight VACS countries. (B) UN regions used for pooling: W=West, C=Central, E=East, S=Southern Africa. (C) Forest plot of country percentages: black markers are point estimates, horizontal lines are 95\% CIs from the survey design (logit method, equation~\ref{eq:ci}), and grey squares are proportional to each country's share of all women analysed (Women, \%). Hollow diamonds are regional subtotals and the overall estimate, pooling women within region with survey weights; their width spans the 95\% CI. The dashed line marks the overall estimate. Basemap: Natural Earth 1:50m.}
\label{fig:map}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=\textwidth,height=0.82\textheight,keepaspectratio]{figures/fig_cascade_forest_lancet.png}
\caption{\textbf{Disclosure cascade after violence among young women aged 15--24 years, by country}\\
(A) Told anyone, (B) sought any help and (C) sought formal help (police, health workers, lawyers or courts, social services, non-governmental and women's organisations, government offices for women, safe spaces and helplines), DHS 2004--2025, """ + str(nC) + r""" countries. Black markers are survey-weighted percentages, horizontal lines are 95\% CIs from the survey design (logit method, equation~\ref{eq:ci}), and grey squares are proportional to the number of women. Hollow diamonds are regional subtotals and the overall estimate, pooling women within region with survey weights; the dashed line marks the overall estimate. Denominators differ by step (""" + fmt(int(cas.loc['told'])) + ", " + fmt(int(cas.loc['sought_any'])) + " and " + fmt(int(cas.loc['formal'])) + r""" women) because in some surveys telling anyone could not be distinguished from seeking help, and in others no formal source was listed. Each panel has its own scale.}
\label{fig:cascade}
\end{figure}

\subsection{Which factors are associated with help-seeking}\label{sec:predictors}
Seeking any help was more likely after injury from partner violence (OR """ + orc(inj) + r"""), severe physical violence (""" + orc(sev) + r"""), both physical and sexual violence (""" + orc(both) + r""") and being formerly in union (""" + orc(form) + r"""; table~\ref{tab:predictors}). These factors raised the probability of seeking help by """ + f"{f1(rdt('partner', 'severe')[0])}" + r""" to """ + f"{f1(rdt('partner', 'injury')[0])}" + r""" percentage points (injury """ + "{:.1f} [95\\% CrI {:.1f}--{:.1f}]".format(*rdt('partner', 'injury')) + r"""; appendix table~\ref{tab:robust}) and were also associated with telling anyone (injury """ + orc(injt) + r"""; severe violence """ + orc(sevt) + r"""). Partner alcohol use (""" + orc(alc) + r"""; credible in West and Central Africa) and current employment (""" + orc(wrk) + r"""; credible in each region) showed smaller associations that were not credible overall. Secondary education (""" + orc(edu) + r"""), wealth (""" + orc(wq) + r""" per quintile), urban residence (""" + orc(urb) + r"""), age (""" + orc(age) + r""" per year) and violence by a non-partner only (""" + orc(npt) + r""") were not associated.

All four factors pointed the same way in West, Central and East Africa and in the exploratory Southern African model (figures~\ref{fig:forest_partner} and~\ref{fig:forest_main}; injury """ + f"{f2(S[(S.term=='injury')].iloc[0].OR)} (95\\% CI {f2(S[(S.term=='injury')].iloc[0].lo)}--{f2(S[(S.term=='injury')].iloc[0].hi)})" + r"""). Of """ + str(sum(N['class_sought_any'].values())) + r""" model terms for seeking any help, """ + str(N['class_sought_any'].get('universal', 0)) + r""" were consistent across regions, """ + str(N['class_sought_any'].get('region-specific', 0)) + r""" region-specific and """ + str(N['class_sought_any'].get('uncertain', 0)) + r""" uncertain. Injury, severe violence and both physical and sexual violence replicated in both independent country sets, and """ + str(N['rep_same_sign']) + r"""\% of all estimates had the same sign in the two sets. No country carried more than """ + f1(maxw) + r"""\% of the weight for injury (equation~\ref{eq:weights}).

The associations were robust (appendix table~\ref{tab:robust}): similar among women whose violence all occurred in the past 12 months (injury """ + orc(p38("B4_timing", "partner", "injury")) + r"""), as prevalence ratios (injury """ + orc(p38("B5_pr", "partner", "injury")) + r"""), in one-stage models (injury """ + f"{f2(OS.loc['injury', 'onestage_weighted_OR'])}" + r""" survey-weighted and """ + f"{f2(OS.loc['injury', 'glmm_OR'])}" + r""" in a mixed model) and without the most extreme country. Crude odds ratios were larger (injury """ + orc(p38("B2_crude", "partner", "injury")) + r"""), because severity, injury and type of violence overlap. Education, wealth and urban residence were null in crude, adjusted and reduced models (secondary education: crude """ + orc(p38("B2_crude", "main", "educsecondary+")) + r"""; reduced """ + orc(p38("B3_reduced", "main", "educsecondary+")) + r"""), and no predictor was collinear. Prediction intervals for a new country excluded one for injury (""" + f"{f2(b1('partner', 'injury').PI_lo)}--{f2(b1('partner', 'injury').PI_hi)}" + r""") but not for severe violence (""" + f"{f2(b1('partner', 'severe').PI_lo)}--{f2(b1('partner', 'severe').PI_hi)}" + r""") or being formerly in union (""" + f"{f2(b1('main', 'maritalformer').PI_lo)}--{f2(b1('main', 'maritalformer').PI_hi)}" + r"""): these associations may be weak in some countries.

Formal help could be modelled in only ten countries (six for partner-violence items, all in East Africa); estimates pointed the same way (injury """ + orc(injf) + r""").

In VACS, girls had higher odds than boys of seeking help after sexual violence (OR """ + orc(girl) + r""") and of receiving services (""" + orc(girl_r) + r"""); intervals were wide because few boys reported sexual violence.

\begin{figure}[p]
\centering
\includegraphics[width=\textwidth,height=0.80\textheight,keepaspectratio]{figures/fig2_forest_injury_severe_lancet.png}
\caption{\textbf{Injury and severe physical violence and seeking any help, by country}\\
Young women aged 15--24 years with partner violence in """ + str(PO2[('partner','injury')]['countries']) + r""" countries (DHS). Country rows are odds ratios from mutually adjusted survey-weighted logistic models in each country with 95\% CIs clustered by primary sampling unit (equations~\ref{eq:logit}--\ref{eq:sandwich}); grey squares are proportional to the number of women. Hollow navy diamonds are regional subtotals and the overall estimate from the Bayesian hierarchical model (posterior medians with 95\% credible intervals, equation~\ref{eq:hier}; between-country SD on the log-odds scale """ + "$\\tau$=" + md2(PO2[('partner','injury')]['tau']) + " for injury and " + md2(PO2[('partner','severe')]['tau']) + r""" for severe violence). The grey diamond is the exploratory pooled model of Lesotho, Namibia and South Africa (95\% CI), which is not part of the overall estimate. The solid line marks no association (odds ratio 1) and the dashed line the overall estimate; arrows mark intervals that extend beyond the axis.}
\label{fig:forest_partner}
\end{figure}

\begin{figure}[p]
\centering
\includegraphics[width=\textwidth,height=0.80\textheight,keepaspectratio]{figures/fig3_forest_both_former_lancet.png}
\caption{\textbf{Physical and sexual violence, being formerly in union and seeking any help, by country}\\
Young women aged 15--24 years who experienced violence in """ + str(PO2[('main','vtypeboth')]['countries']) + r""" countries (DHS). Physical and sexual violence is compared with physical violence only, and formerly in union with currently in union. Estimates and symbols as in figure~\ref{fig:forest_partner} (""" + "$\\tau$=" + md2(PO2[('main','vtypeboth')]['tau']) + " and " + md2(PO2[('main','maritalformer')]['tau']) + r""").}
\label{fig:forest_main}
\end{figure}

\begin{table}[t]
\centering
\begin{threeparttable}
\caption{\textbf{Factors associated with seeking any help after violence among young women aged 15--24 years, by UN region.} Odds ratios (95\% credible or confidence intervals).}
\label{tab:predictors}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{>{\raggedright\arraybackslash}p{4.4cm}ccccc}
\toprule
 & \multicolumn{4}{c}{UN region} & \\
\cmidrule(lr){2-5}
 & West Africa & Central Africa & East Africa & Southern Africa\textsuperscript{b} & All regions \\
 & (""" + str(ncw.get('West', '')) + r""" countries) & (""" + str(ncw.get('Central', '')) + r""") & (""" + str(ncw.get('East', '')) + r""") & (3) & \\
\midrule
""" + table1 + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}\footnotesize
\item Mutually adjusted survey-weighted logistic models in each country (equation~\ref{eq:logit}), combined in a Bayesian hierarchical model (equation~\ref{eq:hier}); regional and all-region estimates are posterior medians with 95\% credible intervals. All regions combines West, Central and East Africa. Bold: interval excludes 1. \textsuperscript{a}Among women with partner violence. \textsuperscript{b}No Southern African country met the inclusion rule; Lesotho, Namibia and South Africa were pooled in one exploratory model (""" + fmt(S_n) + r""" women, """ + str(S_ev) + r""" sought help; 95\% CIs). The all-region intervals are wider than the regional ones because they include between-region variation.
\end{tablenotes}
\end{threeparttable}
\end{table}

\subsection{Pooling DHS and VACS}\label{sec:harmres}
In the harmonised analysis (""" + fmt(nD) + r""" young women in DHS; """ + fmt(nG) + r""" girls in VACS; appendix table~\ref{tab:harm}), both types of violence were associated with seeking any help to a similar degree in both surveys (DHS OR """ + p37("sought_any", "both", "DHS") + r"""; VACS """ + p37("sought_any", "both", "VACS") + r"""; ratio """ + p37s("sought_any", "both", "ratio_VACS_DHS") + r"""; pooled """ + p37("sought_any", "both", "pooled") + r"""). For telling anyone the association was stronger in VACS (""" + p37s("told", "both", "VACS") + r""" vs """ + p37s("told", "both", "DHS") + r"""; ratio """ + p37s("told", "both", "ratio_VACS_DHS") + r"""), possibly because VACS asks about telling once for each type of violence. A female household head tended to be associated with telling anyone (pooled """ + p37s("told", "hh_female", "pooled") + r""") but not with seeking help (""" + p37s("sought_any", "hh_female", "pooled") + r"""). Age and rural residence were not associated in either survey (for age and telling anyone the surveys differed formally, but both estimates were close to one, DHS """ + p37s("told", "age", "DHS") + r""", VACS """ + p37s("told", "age", "VACS") + r""" per year). Results held with the programme difference fixed at zero and in the five countries with both surveys.

\subsection{Change over time}\label{sec:trends}
Across """ + str(ntr) + r""" countries with repeated rounds, seeking any help tended to decline (OR per decade """ + orc(ts) + r"""; """ + f"{f1(tpp.pp)}" + r""" percentage points per decade, 95\% CrI """ + f"{f1(tpp.pp_lo)} to {f1(tpp.pp_hi)}" + r"""), falling in """ + str(ndec) + r""" of """ + str(ntr) + r""" countries; the decline was clearer in DHS phase 6 or later, which removes the largest questionnaire changes (""" + orc(tr6) + r"""; appendix table~\ref{tab:trends}). Telling anyone (""" + orc(tt_) + r""") and formal help (""" + orc(tf) + r""") showed no clear trend, and past-year victimisation tended to decline (""" + orc(vt) + r"""). Among women with partner violence the decline persisted after adjusting for severity and injury (""" + orc(sb) + r""" before; """ + orc(ss) + r""" after).

\subsection{Model checking}\label{sec:diag}
All """ + str(len(P)) + r""" pooled estimates converged ($\hat R\le """ + f"{P.rhat_max.max():.3f}" + r"""$; minimum bulk effective sample size """ + str(int(P.ess_min.min())) + r"""); """ + str(n_div) + r""" divergent transitions occurred in """ + str(n_divfits) + r""" fits (at most 0.2\% of draws). The priors admitted odds ratios of """ + f"{f2(PPd['lo'])}--{f2(PPd['hi'])}" + r""" for a new country (appendix figure~\ref{fig:priorpred}). The common-effect model misfitted for """ + str(n_m0_misfit) + r""" of """ + str(n_terms) + r""" factors ($p_B<0.05$ for the $\chi^2$ discrepancy), confirming between-country heterogeneity; the primary model reproduced every discrepancy ($p_B$ """ + f"{f2(_pb_ok[0])}--{f2(_pb_ok[-1])}" + r""") except the largest estimate for both types of violence (Togo, OR """ + f"{f2(TGx.togo_OR)}" + r"""; excluding Togo, """ + f"{f2(TGx.OR_without_togo)} [{f2(TGx.lo)}--{f2(TGx.hi)}]" + r"""; appendix table~\ref{tab:ppc}, figure~\ref{fig:ppc}). DIC, WAIC and cross-validation did not favour the common-effect model. The model without regions predicted marginally better in """ + str(n_m1_better) + r""" of """ + str(2 * n_terms) + r""" comparisons (difference in expected log predictive density at most """ + f"{max_delpd:.1f}" + r""", below the threshold for a meaningful difference\cite{sivula2022}) with nearly identical odds ratios; we kept the regional model, which answers the study question and is more conservative (appendix table~\ref{tab:ic}). Pooled odds ratios were insensitive to the prior on between-region variation, although the width of the all-region intervals was not (appendix table~\ref{tab:priorsens}).

\section{Discussion}\label{sec:discussion}
\noindent\todo{Outline for drafting; literature led by DM. The previous prose is saved in the archive folder of this Overleaf project. Target about 1000 words. Numbers below are generated from the current results.}

\noindent\textbf{Paragraph 1. Principal findings}
\begin{itemize}\setlength{\itemsep}{1pt}
\item Across """ + str(nC) + r""" countries, about half of young women with violence told someone (""" + f"{f1(At.pct)}" + r"""\%), a third sought help (""" + f"{f1(As.pct)}" + r"""\%) and fewer than one in thirty reached formal services (""" + f"{f1(Af.pct)}" + r"""\%); population-weighted figures were slightly lower (""" + f"{pw('told').pct:.1f}, {pw('sought_any').pct:.1f} and {pw('formal').pct:.1f}" + r"""\%).
\item Help-seeking was associated with injury, severe violence, both physical and sexual violence and being formerly in union (""" + f"{f1(rdt('partner', 'severe')[0])}" + r""" to """ + f"{f1(rdt('partner', 'injury')[0])}" + r""" percentage points higher), shared across regions and replicated in independent countries; wealth, education and urban residence were not associated.
\item Help-seeking has not risen and may be falling (OR per decade """ + orc(ts) + r"""; """ + orc(tr6) + r""" in DHS phase 6 or later).
\end{itemize}

\noindent\textbf{Paragraph 2. Comparison with other studies}
\begin{itemize}\setlength{\itemsep}{1pt}
\item Levels: consistent with multi-country DHS analyses of women of all ages\cite{palermo2014,aboagye2023,goodson2021,stiller2025} and with VACS (lower among boys).\cite{sumner2015,pereira2020,velloza2022,denhard2024}
\item Determinants: severity associated with help-seeking, as in earlier work;\cite{goodson2021,ayebeng2024} unlike some pooled analyses, no association with wealth or education in crude, adjusted or reduced models.\cite{aboagye2023}
\item Method: partial pooling showed that associations are shared across regions, quantified between-country variation (prediction intervals) and replicated findings, which single pooled models cannot do.
\end{itemize}

\noindent\textbf{Paragraph 3. Interpretation}
\begin{itemize}\setlength{\itemsep}{1pt}
\item Help is sought at a crisis point (injury, severity, end of the relationship), consistent with help-seeking models.\cite{liang2005,rickwood2005}
\item Informal networks dominate (own family """ + f"{hs('src_family'):.0f}" + r"""\%; police """ + f"{hs('src_police'):.1f}" + r"""\%; health workers """ + f"{hs('src_health'):.1f}" + r"""\%), so families and community responders are the first point of contact.
\item The regional difference lies in the level of formal help (West """ + f"{f1(reg[('formal','West')].pct)}" + r"""\% to Southern """ + f"{f1(reg[('formal','Southern')].pct)}" + r"""\%), not in who seeks help; consistent with differences in service availability, which we could not measure.\cite{garciamoreno2015}
\end{itemize}

\noindent\textbf{Paragraph 4. Implications}
\begin{itemize}\setlength{\itemsep}{1pt}
\item Earlier, low-threshold routes to help (school and clinic identification, youth-friendly helplines, community responders), aligned with INSPIRE and health-system guidance.\cite{who2016,garciamoreno2015,ellsberg2015}
\item Services that include boys and unmarried young people (half of the sample had never been in a union).
\item Monitoring help-seeking alongside violence prevalence (SDG targets 5.2 and 16.2).\cite{un2015}
\end{itemize}

\noindent\textbf{Paragraph 5. Strengths}
\begin{itemize}\setlength{\itemsep}{1pt}
\item Two independent survey programmes; a pre-registered plan with time-stamped amendments; design-based uncertainty; partial pooling with a full Bayesian workflow (prior and posterior predictive checks, cross-validated model comparison); robustness to the effect measure, timing, one-stage modelling and population weighting; replication in independent countries.
\end{itemize}

\noindent\textbf{Paragraph 6. Limitations}
\begin{itemize}\setlength{\itemsep}{1pt}
\item Cross-sectional: associations may run both ways (seeking help may lead to leaving the partner).
\item Lifetime help-seeking, observed only among women who disclosed violence to the interviewer; reporting depends on survey methods.\cite{cullen2023,palermo2014}
\item Formal help too rare for regional comparison; VACS covers only East and Southern Africa and does not identify perpetrators; the perpetrator of forced sex could not be identified in """ + str(n_d127_old) + r""" DHS surveys.
\item Trends mix real change with questionnaire change; prediction intervals show that some associations may be weak in individual countries; the regional model was kept although a model without regions predicted marginally better for some factors.
\end{itemize}

\noindent\textbf{Paragraph 7. Conclusion}
\begin{itemize}\setlength{\itemsep}{1pt}
\item Young survivors seek help mainly once harm is severe, and few reach services; making help reachable earlier, and for boys, should be central to violence prevention and response for young people.
\end{itemize}

\section*{Contributors}
\todo{To be agreed. Proposed: ST conceived the study, built the data, did the analysis and wrote the first draft; DM led the literature review and the introduction and discussion; SF and LC contributed to design and interpretation. All authors had access to the data and approved the final version.}

\section*{Declaration of interests}
We declare no competing interests.

\section*{Data sharing}
DHS data are available from the DHS Program (\url{https://dhsprogram.com}) on registration; VACS public-use data from Together for Girls and the US Centers for Disease Control and Prevention. Data cannot be shared by the authors under these licences. Analysis code is available at \url{https://github.com/teklesam/disclosure-help-seeking-africa}.

\begin{thebibliography}{99}\setlength{\itemsep}{0pt}\small
\bibitem{who2016} World Health Organization. INSPIRE: seven strategies for ending violence against children. Geneva: World Health Organization, 2016.
\bibitem{sardinha2022} Sardinha L, Maheu-Giroux M, St\"ockl H, Meyer SR, Garc\'ia-Moreno C. Global, regional, and national prevalence estimates of physical or sexual, or both, intimate partner violence against women in 2018. \textit{Lancet} 2022; \textbf{399}: 803--13.
\bibitem{palermo2014} Palermo T, Bleck J, Peterman A. Tip of the iceberg: reporting and gender-based violence in developing countries. \textit{Am J Epidemiol} 2014; \textbf{179}: 602--12.
\bibitem{pereira2020} Pereira A, Peterman A, Neijhoft AN, et al. Disclosure, reporting and help seeking among child survivors of violence: a cross-country analysis. \textit{BMC Public Health} 2020; \textbf{20}: 1051.
\bibitem{liang2005} Liang B, Goodman L, Tummala-Narra P, Weintraub S. A theoretical framework for understanding help-seeking processes among survivors of intimate partner violence. \textit{Am J Community Psychol} 2005; \textbf{36}: 71--84.
\bibitem{croft2018} Croft TN, Marshall AMJ, Allen CK, et al. Guide to DHS statistics: DHS-7. Rockville, MD: ICF, 2018.
\bibitem{nguyen2019} Nguyen KH, Kress H, Villaveces A, Massetti GM. Sampling design and methodology of the Violence Against Children and Youth Surveys. \textit{Inj Prev} 2019; \textbf{25}: 321--27.
\bibitem{lumley2004} Lumley T. Analysis of complex survey samples. \textit{J Stat Softw} 2004; \textbf{9}: 1--19.
\bibitem{riley2010} Riley RD, Lambert PC, Abo-Zaid G. Meta-analysis of individual participant data: rationale, conduct, and reporting. \textit{BMJ} 2010; \textbf{340}: c221.
\bibitem{berge2018} Berg\'e L. Efficient estimation of maximum likelihood models with multiple fixed-effects: the R package FENmlm. CREA Discussion Papers 13. Luxembourg: University of Luxembourg, 2018.
\bibitem{gelman2006} Gelman A. Prior distributions for variance parameters in hierarchical models. \textit{Bayesian Anal} 2006; \textbf{1}: 515--34.
\bibitem{hoffman2014} Hoffman MD, Gelman A. The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. \textit{J Mach Learn Res} 2014; \textbf{15}: 1593--623.
\bibitem{phan2019} Phan D, Pradhan N, Jankowiak M. Composable effects for flexible and accelerated probabilistic programming in NumPyro. \textit{arXiv} 2019; 1912.11554.
\bibitem{vehtari2021} Vehtari A, Gelman A, Simpson D, Carpenter B, B\"urkner P-C. Rank-normalization, folding, and localization: an improved $\hat R$ for assessing convergence of MCMC. \textit{Bayesian Anal} 2021; \textbf{16}: 667--718.
\bibitem{dersimonian1986} DerSimonian R, Laird N. Meta-analysis in clinical trials. \textit{Control Clin Trials} 1986; \textbf{7}: 177--88.
\bibitem{holm1979} Holm S. A simple sequentially rejective multiple test procedure. \textit{Scand J Stat} 1979; \textbf{6}: 65--70.
\bibitem{burke2017} Burke DL, Ensor J, Riley RD. Meta-analysis using individual participant data: one-stage and two-stage approaches, and why they may differ. \textit{Stat Med} 2017; \textbf{36}: 855--75.
\bibitem{campos2023} Campos DG, Cheung MW-L, Scherer R. A primer on synthesizing individual participant data obtained from complex sampling surveys: a two-stage IPD meta-analysis approach. \textit{Psychol Methods} 2023; published online Jan 9. DOI:10.1037/met0000539.
\bibitem{westreich2013} Westreich D, Greenland S. The table 2 fallacy: presenting and interpreting confounder and modifier coefficients. \textit{Am J Epidemiol} 2013; \textbf{177}: 292--98.
\bibitem{gelman2020} Gelman A, Vehtari A, Simpson D, et al. Bayesian workflow. \textit{arXiv} 2020; 2011.01808.
\bibitem{gabry2019} Gabry J, Simpson D, Vehtari A, Betancourt M, Gelman A. Visualization in Bayesian workflow. \textit{J R Stat Soc Ser A} 2019; \textbf{182}: 389--402.
\bibitem{gelman1996} Gelman A, Meng X-L, Stern H. Posterior predictive assessment of model fitness via realized discrepancies. \textit{Stat Sin} 1996; \textbf{6}: 733--807.
\bibitem{spiegelhalter2002} Spiegelhalter DJ, Best NG, Carlin BP, van der Linde A. Bayesian measures of model complexity and fit. \textit{J R Stat Soc Ser B} 2002; \textbf{64}: 583--639.
\bibitem{watanabe2010} Watanabe S. Asymptotic equivalence of Bayes cross validation and widely applicable information criterion in singular learning theory. \textit{J Mach Learn Res} 2010; \textbf{11}: 3571--94.
\bibitem{vehtari2017} Vehtari A, Gelman A, Gabry J. Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. \textit{Stat Comput} 2017; \textbf{27}: 1413--32.
\bibitem{sivula2022} Sivula T, Magnusson M, Matamoros AA, Vehtari A. Uncertainty in Bayesian leave-one-out cross-validation based model comparison. \textit{arXiv} 2020; 2008.10296.
\bibitem{riley2011} Riley RD, Higgins JPT, Deeks JJ. Interpretation of random effects meta-analyses. \textit{BMJ} 2011; \textbf{342}: d549.
\bibitem{zou2004} Zou G. A modified Poisson regression approach to prospective studies with binary data. \textit{Am J Epidemiol} 2004; \textbf{159}: 702--06.
\bibitem{bates2015} Bates D, M\"achler M, Bolker B, Walker S. Fitting linear mixed-effects models using lme4. \textit{J Stat Softw} 2015; \textbf{67}: 1--48.
\bibitem{fox1992} Fox J, Monette G. Generalized collinearity diagnostics. \textit{J Am Stat Assoc} 1992; \textbf{87}: 178--83.
\bibitem{worldpop2025} WorldPop. Global2 age and sex structures, release R2025A. Southampton: University of Southampton, 2025.
\bibitem{aboagye2023} Aboagye RG, Seidu A-A, Cadri A, et al. Ending violence against women: help-seeking behaviour of women exposed to intimate partner violence in sub-Saharan Africa. \textit{PLoS One} 2023; \textbf{18}: e0291913.
\bibitem{ayebeng2024} Ayebeng C, Dickson KS, Ameyaw EK, et al. Influence of type of violence on women's help-seeking behaviour: evidence from 10 countries in sub-Saharan Africa. \textit{PLoS One} 2024; \textbf{19}: e0297308.
\bibitem{goodson2021} Goodson A, Hayes BE. Help-seeking behaviors of intimate partner violence victims: a cross-national analysis in developing nations. \textit{J Interpers Violence} 2021; published online Aug 22, 2018. DOI:10.1177/0886260518794508.
\bibitem{stiller2025} Stiller M, Wilson ML, B\"arnighausen T, Adedimeji A, Lewis E, Abio A. Help-seeking behaviors among survivors of intimate partner violence during pregnancy in 54 low- and middle-income countries: evidence from Demographic and Health Survey data. \textit{BMC Public Health} 2025. DOI:10.1186/s12889-025-21421-3.
\bibitem{hillis2016} Hillis S, Mercy J, Amobi A, Kress H. Global prevalence of past-year violence against children: a systematic review and minimum estimates. \textit{Pediatrics} 2016; \textbf{137}: e20154079.
\bibitem{decker2014} Decker MR, Peitzmeier S, Olumide A, et al. Prevalence and health impact of intimate partner violence and non-partner sexual violence among female adolescents aged 15--19 years in vulnerable urban environments: a multi-country study. \textit{J Adolesc Health} 2014; \textbf{55} (suppl): S58--67.
\bibitem{jewkes2010} Jewkes RK, Dunkle K, Nduna M, Shai N. Intimate partner violence, relationship power inequity, and incidence of HIV infection in young women in South Africa: a cohort study. \textit{Lancet} 2010; \textbf{376}: 41--48.
\bibitem{who2013} WHO. Responding to intimate partner violence and sexual violence against women: WHO clinical and policy guidelines. Geneva: World Health Organization, 2013.
\bibitem{un2015} UN General Assembly. Transforming our world: the 2030 Agenda for Sustainable Development. New York: United Nations, 2015.
\bibitem{sumner2015} Sumner SA, Mercy AA, Saul J, et al. Prevalence of sexual violence against children and use of social services---seven countries, 2007--2013. \textit{MMWR Morb Mortal Wkly Rep} 2015; \textbf{64}: 565--69.
\bibitem{velloza2022} Velloza J, Davies L, Ensminger A, et al. Disclosure and help-seeking behaviors related to sexual and physical violence in childhood and adolescence: results from the Namibia Violence Against Children and Youth Survey. \textit{Child Abuse Negl} 2022; \textbf{128}: 105624.
\bibitem{denhard2024} Denhard L, Brown C, Kanagasabai U, et al. Service-seeking behaviors among male victims of violence in five African countries: the effects of positive and adverse childhood experiences. \textit{Child Abuse Negl} 2024; \textbf{150}: 106452.
\bibitem{okova2025} Okova D, Lukwa AT, Alaba OA. Preventing sexual violence and strengthening post-victimization support among adolescents and young people in Kenya: an INSPIRE-aligned analysis of the 2019 Violence Against Children Survey (VACS). \textit{Int J Environ Res Public Health} 2025; \textbf{22}: 863.
\bibitem{rickwood2005} Rickwood D, Deane FP, Wilson CJ, Ciarrochi J. Young people's help-seeking for mental health problems. \textit{Aust e-J Adv Ment Health} 2005; \textbf{4}: 218--51. \todo{verify}
\bibitem{cullen2023} Cullen C. Method matters: the underreporting of intimate partner violence. \textit{World Bank Econ Rev} 2023; \textbf{37}: 49--73. \todo{verify}
\bibitem{garciamoreno2015} Garc\'ia-Moreno C, Hegarty K, d'Oliveira AF, Koziol-McLain J, Colombini M, Feder G. The health-systems response to violence against women. \textit{Lancet} 2015; \textbf{385}: 1567--79.
\bibitem{ellsberg2015} Ellsberg M, Arango DJ, Morton M, et al. Prevention of violence against women and girls: what does the evidence say? \textit{Lancet} 2015; \textbf{385}: 1555--66.
\end{thebibliography}

""" + APP.build(tableS1, n37d, n37v, SRC_TABLE) + r"""
\end{document}
"""
tex = lancet(tex)
(OL / "main.tex").write_text(tex)
(Path(sys.argv[2]) / "numbers_used.json").write_text(json.dumps(N, indent=1, default=str))
print("wrote", OL / "main.tex", "|", len(N), "numbers recorded")
