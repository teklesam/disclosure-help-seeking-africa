"""Independent check: recompute key numbers straight from result files (separate code path from the builder) and confirm they appear in the PDF text."""
import numpy as np, sys, re, numpy as np, pandas as pd, fitz
from pathlib import Path
R=__import__("pathlib").Path(__import__("os").environ["RR"]); V3=R/"outputs/tables/P33_youth_v3"
txt=" ".join(re.sub(r"(?m)^\s*\d{1,4}\s*$", "", p.get_text()) for p in fitz.open(sys.argv[1]));  # drop margin line numbers (lineno)
txt=re.sub(r"\s+"," ",txt).replace("–","-").replace("−","-").replace("·",".").replace("[","(").replace("]",")")  # mid-dot decimals and nested brackets
checks=[]
def chk(label, s): checks.append((label, s, s in txt))
c=pd.read_csv(V3/"A8_5_dhs_cascade_ci.csv")
for o,lab in [("told","told anyone"),("sought_any","sought any help"),("formal","sought formal help")]:
    r=c[(c.level=="all")&(c.outcome==o)].iloc[0]; chk(f"cascade {lab}", f"{r.pct:.1f}% (95% CI {r.lo:.1f}-{r.hi:.1f})")
for g in ["West","Central","East","Southern"]:
    r=c[(c.level=="region")&(c.region==g)&(c.outcome=="formal")].iloc[0]; chk(f"formal {g}", f"{g} {r.pct:.1f}% ({r.lo:.1f}-{r.hi:.1f})")
P=pd.read_csv(V3/"A3_pooled_numpyro.csv")
for m,t,o in [("partner","injury","sought_any"),("partner","severe","sought_any"),("main","vtypeboth","sought_any"),("main","maritalformer","sought_any"),("partner","injury","told"),("main","wealthq","sought_any")]:
    r=P[(P.source=="DHS")&(P.model==m)&(P.term==t)&(P.outcome==o)].iloc[0]; chk(f"OR {t} {o}", f"{r.OR:.2f} (95% CrI {r.OR_lo:.2f}-{r.OR_hi:.2f})")
r=P[(P.source=="VACS")&(P.model=="SV")&(P.term=="female")&(P.outcome=="sought")].iloc[0]; chk("VACS girls vs boys", f"{r.OR:.2f} (95% CrI {r.OR_lo:.2f}-{r.OR_hi:.2f})")
T=pd.read_csv(V3/"A5_trends_numpyro.csv"); r=T[(T.model=="adjusted")&(T.outcome=="sought_any")].iloc[0]; chk("trend sought", f"{r.OR:.2f} (95% CrI {r.OR_lo:.2f}-{r.OR_hi:.2f})")
A=pd.read_csv(V3/"A8_2_trends_numpyro.csv"); r=A[A.model.str.contains("severity")].iloc[0]; chk("trend severity-adjusted", f"{r.OR:.2f} (95% CrI {r.OR_lo:.2f}-{r.OR_hi:.2f})")
# Table 1 cells (West injury, Southern injury)
r=P[(P.source=="DHS")&(P.model=="partner")&(P.term=="injury")&(P.outcome=="sought_any")].iloc[0]; chk("Table1 West injury", f"{r.OR_West:.2f} ({r.OR_West_lo:.2f}-{r.OR_West_hi:.2f})")
S=pd.read_csv(V3/"southern_africa_pooled_exploratory.csv"); s=S[S.term=="injury"].iloc[0]; chk("Table1 Southern injury", f"{s.OR:.2f} ({s.lo:.2f}-{s.hi:.2f})")
# sample sizes
d=pd.concat([pd.read_csv(R/f"data/derived/{f}.csv.gz",usecols=["survey","sought_any","vtype","v012"]) for f in ["women_disclosure","women_disclosure_heldout"]])
d=d[(d.survey!="SN_2019_8B")&d.sought_any.notna()&(d.vtype!="none")&(d.v012<=24)]
chk("n DHS women", f"{len(d):,}".replace(","," ")); chk("n surveys", f"{d.survey.nunique()} Demographic and Health Surveys")
v=pd.read_csv(Path.home()/"Documents/OMS VACS Climate Violence/data/processed/vacs_disclosure_harmonised.csv"); chk("n VACS", f"{len(v):,}".replace(","," "))
# Pre-reg 37 harmonised pool: read the pooled table directly and recount VACS girls 15-24 with violence
P37=pd.read_csv(__import__("pathlib").Path(__import__("os").environ["RR"])/"outputs/tables/P37_harmonised/P37_pooled.csv").set_index(["outcome","term"])
g=lambda o,t,k: f"{P37.loc[(o,t)][f'OR_{k}']:.2f} (95% CrI {P37.loc[(o,t)][f'OR_{k}_lo']:.2f}-{P37.loc[(o,t)][f'OR_{k}_hi']:.2f})"
chk("P37 both sought pooled", g("sought_any","both","pooled")); chk("P37 both sought DHS", g("sought_any","both","DHS")); chk("P37 both sought VACS", g("sought_any","both","VACS"))
r=P37.loc[("told","both")]; chk("P37 both told ratio", f"{r.OR_ratio_VACS_DHS:.2f} ({r.OR_ratio_VACS_DHS_lo:.2f}-{r.OR_ratio_VACS_DHS_hi:.2f})")
gv=v[(v.female==1)&(v.age>=15)&(v.age<=24)&((v.sv_victim==1)|(v.pv_victim==1))]; chk("P37 VACS girls 15-24", f"{len(gv):,}".replace(","," ")+" girls in VACS")
# Pre-registration 38: read the review result files directly
Q=__import__("pathlib").Path(__import__("os").environ["RR"])/"outputs/tables/P38_review"
pw=pd.read_csv(Q/"P38_B9_population_weighted_cascade.csv"); r=pw[(pw.outcome=="sought_any")&(pw.weighting=="population")&(pw.region=="All")].iloc[0]; chk("P38 population-weighted sought", f"{r.pct:.1f}%")
t38=pd.read_csv(Q/"P38_pooled.csv"); r=t38[(t38.analysis=="B5_rd")&(t38.term=="injury")].iloc[0]; chk("P38 risk difference injury", f"{100*np.log(r.OR):.1f} (95% CrI {100*np.log(r.OR_lo):.1f}-{100*np.log(r.OR_hi):.1f}")
b1=pd.read_csv(Q/"P38_B1_prior_sensitivity.csv"); r=b1[(b1.term=="injury")&(b1.omega_prior_sd==0.5)].iloc[0]; chk("P38 prediction interval injury", f"({r.PI_lo:.2f}-{r.PI_hi:.2f})")
tg=pd.read_csv(Q/"P38_B12_togo_exclusion.csv").iloc[0]; chk("P38 Togo exclusion", f"{tg.OR_without_togo:.2f} ({tg.lo:.2f}-{tg.hi:.2f})")
mc=pd.read_csv(Q/"P38_B12_model_checks.csv"); n1=int(((mc.model=="M1")&(mc.elpd_diff_vs_M2>2*mc.se_diff_vs_M2)).sum()); chk("P38 M1 better count", f"marginally better in {n1} of")
hs=pd.read_csv(Q/"P38_B8_help_sources.csv"); chk("P38 help from family", f"from family ({hs[(hs.source=='src_family')&(hs.region=='All')].pct.iloc[0]:.0f}%)")
# Pre-registration 39 trends: read the trend tables directly
ts39=pd.read_csv(Q/"P39_slopes.csv"); r=ts39[(ts39.outcome=="sought_any")&(ts39.scope=="All")].iloc[0]
chk("P39 sought trend", f"{r.OR_decade:.2f} (95% CrI {r.OR_lo:.2f}-{r.OR_hi:.2f})"); chk("P39 fall 2005-2025", f"{-r.pp_2005_2025:.1f} percentage points (95% CrI {-r.pp_hi:.1f} to {-r.pp_lo:.1f})")
r=ts39[(ts39.outcome=="formal")&(ts39.scope=="East")].iloc[0]; chk("P39 formal East", f"{r.OR_decade:.2f} (95% CrI {r.OR_lo:.2f}-{r.OR_hi:.2f})")
# Pre-registration 42: recompute the single-type percentage from the derived files (point estimate) and read the pooled ratio directly
ty=pd.read_csv(R/"data/derived/women_perp_type.csv.gz",dtype={"caseid":str})
w42=pd.concat([pd.read_csv(R/f"data/derived/{f}.csv.gz",usecols=["survey","caseid","sought_any","formal","vtype","v012","d005"],dtype={"caseid":str}) for f in ["women_disclosure","women_disclosure_heldout"]])
w42=w42[(w42.survey!="SN_2019_8B")&w42.sought_any.notna()&(w42.vtype!="none")&(w42.v012<=24)]; w42["caseid"]=w42.caseid.str.strip()
w42=w42.merge(ty,on=["survey","caseid"]); ps=w42[(w42.PS==1)&(w42.PP==0)&(w42.NP==0)&(w42.NS==0)]
chk("P42 partner sexual only sought (recomputed)", f"{100*np.average(ps.sought_any,weights=ps.d005):.1f}% (")
o42=pd.read_csv(R/"outputs/tables/P42/P42_pooled.csv"); r=o42[(o42.analysis=="B_types")&(o42.term=="NS_minus_PS")&(o42.outcome=="sought_any")].iloc[0]
chk("P42 NS vs PS ratio", f"{r.OR:.2f} (95% CrI {r.OR_lo:.2f}-{r.OR_hi:.2f})")
# Correction 5 post-hoc addition 4: SDG 16.3.1 proxy, population-weighted cascade and coverage, read from the result files
s87=pd.read_csv(Q/"P87_sdg1631_pooled.csv"); r=s87[(s87.population=="victim_12m")&(s87.weighting=="population")&(s87.region=="All")].iloc[0]
chk("SDG proxy population-weighted", f"{r.pct:.1f}% (95% CI {r.lo:.1f}-{r.hi:.1f})")
t87=pd.read_csv(Q/"P87_sdg1631_slopes.csv"); r=t87[(t87.population=="victim_12m")&(t87.scope=="All")].iloc[0]; chk("SDG proxy trend", f"{r.OR_decade:.2f} (95% CrI {r.OR_lo:.2f}-{r.OR_hi:.2f})")
r=pw[(pw.outcome=="told")&(pw.weighting=="population")&(pw.region=="All")].iloc[0]; chk("population-weighted told", f"{r.pct:.1f}% (95% CI {r.lo:.1f}-{r.hi:.1f}) told anyone")
d87=pd.read_csv(Q/"P87_sdg1631_definitions.csv").set_index("definition").drop("police_legal"); chk("SDG proxy wider definitions", f"{d87.pct.min():.1f}% to {d87.pct.max():.1f}%")
r89=pd.read_csv(Q/"P89_results.csv"); r=r89[(r89.analysis=="prevalence trend")&(r89.measure=="victim_ever")].iloc[0]; chk("violence prevalence trend", f"{r.est:.2f} (95% CI {r.lo:.2f}-{r.hi:.2f})")   # post-hoc 6
cv=pd.read_csv(Q/"P88_coverage_table.csv").set_index("group"); chk("population coverage", f"{cv.loc['Sub-Saharan Africa','pop_coverage_pct']:.1f}% of the")
# Correction 5 post-hoc addition 5: population-weighted trend curves (read from 84f output)
pwt=pd.read_csv(Q/"P39pw_summary.csv").set_index(["outcome","scope"])
r=pwt.loc[("sought_any","All")]; chk("pw trend sought fall", f"{-r.change_pp:.1f} percentage points (95% CrI {-r.change_hi:.1f} to {-r.change_lo:.1f};")
chk("pw sought 2005 and 2025", f"from {r.pct_2005:.1f}% in 2005 to {r.pct_2025:.1f}% in 2025")
r=pwt.loc[("formal","All")]; chk("pw formal rise", f"{r.change_pp:.1f} percentage points (95% CrI {r.change_lo:.1f} to {r.change_hi:.1f};")
ok=sum(x[2] for x in checks)
for lab,s,f in checks: print(("OK  " if f else "MISS"), lab, "|", s)
print(f"\n{ok}/{len(checks)} numbers found in the PDF")
