"""82_make_readme.py - writes the public README.md of the code repository, with every number read from the result files.
Usage: python3 analysis/82_make_readme.py <repo_dir>
"""
import sys, re, numpy as np, pandas as pd
from pathlib import Path
R = Path.home() / "Documents/OMS DHS Births Climate Policy"; V3 = R / "outputs/tables/P33_youth_v3"; P38 = R / "outputs/tables/P38_review"; P37 = R / "outputs/tables/P37_harmonised"
REPO = Path(sys.argv[1])
f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:.2f}"; ci = lambda o, l, h: f"{f2(o)} ({f2(l)}–{f2(h)})"; pc = lambda r: f"{f1(r.pct)}% ({f1(r.lo)}–{f1(r.hi)})"

C = pd.read_csv(V3 / "A8_5_dhs_cascade_ci.csv"); PW = pd.read_csv(P38 / "P38_B9_population_weighted_cascade.csv")
A3 = pd.read_csv(V3 / "A3_pooled_numpyro.csv"); S = pd.read_csv(V3 / "southern_africa_pooled_exploratory.csv")
T38 = pd.read_csv(P38 / "P38_pooled.csv"); B1 = pd.read_csv(P38 / "P38_B1_prior_sensitivity.csv"); OS = pd.read_csv(P38 / "P38_B6_onestage.csv").set_index("term")
H = pd.read_csv(P37 / "P37_pooled.csv").set_index(["outcome", "term"]); MC = pd.read_csv(P38 / "P38_B12_model_checks.csv")
PP = pd.read_csv(P38 / "P38_B12_prior_predictive.csv").iloc[0]; TG = pd.read_csv(P38 / "P38_B12_togo_exclusion.csv").set_index("outcome").loc["sought_any"]
TR = pd.read_csv(V3 / "A5_trends_numpyro.csv"); TRp = pd.read_csv(V3 / "A5_pooled_trends.csv"); HS = pd.read_csv(P38 / "P38_B8_help_sources.csv")
VC = pd.read_csv(V3 / "A8_5_vacs_cascade_ci.csv"); TS = pd.read_csv(P38 / "P39_slopes.csv"); TMf = pd.read_csv(P38 / "P39_metafor_tests.csv").set_index("outcome"); CE = pd.read_csv(V3 / "A2_country_estimates.csv")
W = pd.concat([pd.read_csv(R / f"data/derived/{f}", usecols=["survey", "country", "v012", "sought_any", "vtype"]) for f in ["women_disclosure.csv.gz", "women_disclosure_heldout.csv.gz"]])
W = W[(W.survey != "SN_2019_8B") & W.sought_any.notna() & (W.vtype != "none") & (W.v012 <= 24)]
V = pd.read_csv(Path.home() / "Documents/OMS VACS Climate Violence/data/processed/vacs_disclosure_harmonised.csv")
nD, nS, nC, nV = len(W), W.survey.nunique(), W.country.nunique(), len(V)
nVany = int(((V.sv_victim == 1) | (V.pv_victim == 1)).sum())
pool = lambda m, t, o="sought_any": A3[(A3.source == "DHS") & (A3.model == m) & (A3.outcome == o) & (A3.term == t)].iloc[0]
p38 = lambda a, m, t: T38[(T38.analysis == a) & (T38.model == m) & (T38.term == t)].iloc[0]
cas = lambda o, lvl="all", reg=None: C[(C.outcome == o) & (C.level == lvl) & ((C.region == reg) if reg else True)].iloc[0]
pw = lambda o, reg="All": PW[(PW.outcome == o) & (PW.weighting == "population") & (PW.region == reg)].iloc[0]
vac = lambda st, sex: VC[(VC.level == "all VACS countries") & (VC.type == "SV") & (VC.sex == sex) & (VC.step == st)].iloc[0]
hs = lambda k: HS[(HS.source == k) & (HS.region == "All")].pct.iloc[0]

LAB = [("partner", "injury", "Injury from partner violence (vs none)ᵃ"), ("partner", "severe", "Severe physical violence ᵃ"), ("main", "vtypeboth", "Physical and sexual violence (vs physical only)"),
       ("main", "maritalformer", "Formerly in union (vs currently in union)"), ("partner", "alcohol", "Partner drinks alcohol ᵃ"), ("main", "working", "Currently working"),
       ("main", "educsecondary+", "Secondary or higher education (vs none)"), ("main", "wealthq", "Household wealth (per quintile)"), ("main", "urban", "Urban residence (vs rural)")]
def reg_cell(r, g): return ci(r[f"OR_{g}"], r[f"OR_{g}_lo"], r[f"OR_{g}_hi"])
rows_t1 = []
for m, t, lab in LAB:
    r = pool(m, t); s = S[S.term == t]; b = B1[(B1.model == m) & (B1.term == t) & (B1.omega_prior_sd == 0.5)].iloc[0]
    rows_t1.append(f"| {lab} | {reg_cell(r, 'West')} | {reg_cell(r, 'Central')} | {reg_cell(r, 'East')} | " + (ci(s.OR.iloc[0], s.lo.iloc[0], s.hi.iloc[0]) if len(s) else "–") +
                   f" | **{ci(r.OR, r.OR_lo, r.OR_hi)}** | {f2(b.PI_lo)}–{f2(b.PI_hi)} |")
F4 = [("partner", "injury", "Injury"), ("partner", "severe", "Severe violence"), ("main", "vtypeboth", "Both types"), ("main", "maritalformer", "Formerly in union")]
rd = lambda m, t: (lambda r: f"+{100*np.log(r.OR):.1f} ({100*np.log(r.OR_lo):.1f} to {100*np.log(r.OR_hi):.1f})")(p38("B5_rd", m, t))
rob = [("Primary (two-stage, Bayesian)", [ci(pool(m, t).OR, pool(m, t).OR_lo, pool(m, t).OR_hi) for m, t, _ in F4]),
       ("Risk difference, percentage points", [rd(m, t) for m, t, _ in F4]),
       ("Prevalence ratio", [(lambda r: ci(r.OR, r.OR_lo, r.OR_hi))(p38("B5_pr", m, t)) for m, t, _ in F4]),
       ("Violence only in past 12 months", [(lambda r: ci(r.OR, r.OR_lo, r.OR_hi))(p38("B4_timing", m, t)) for m, t, _ in F4]),
       ("Crude", [(lambda r: ci(r.OR, r.OR_lo, r.OR_hi))(p38("B2_crude", m, t)) for m, t, _ in F4]),
       ("One-stage, survey-weighted", [ci(OS.loc[t, "onestage_weighted_OR"], OS.loc[t, "lo1"], OS.loc[t, "hi1"]) for _, t, _ in F4]),
       ("One-stage mixed model", [ci(OS.loc[t, "glmm_OR"], OS.loc[t, "lo2"], OS.loc[t, "hi2"]) for _, t, _ in F4])]
hrow = lambda o, t, lab: f"| {lab} | {ci(*H.loc[(o, t)][['OR_DHS', 'OR_DHS_lo', 'OR_DHS_hi']])} | {ci(*H.loc[(o, t)][['OR_VACS', 'OR_VACS_lo', 'OR_VACS_hi']])} | {ci(*H.loc[(o, t)][['OR_ratio_VACS_DHS', 'OR_ratio_VACS_DHS_lo', 'OR_ratio_VACS_DHS_hi']])} | {ci(*H.loc[(o, t)][['OR_pooled', 'OR_pooled_lo', 'OR_pooled_hi']])} | {H.loc[(o, t)].agreement} |"
tr = lambda o: (lambda r: ci(r.OR, r.OR_lo, r.OR_hi))(TR[(TR.model == "adjusted") & (TR.outcome == o)].iloc[0])
tr6 = (lambda r: ci(r.OR, r.OR_lo, r.OR_hi))(TRp[(TRp.scope == "S2: DHS phase >= 6") & (TRp.model == "adjusted") & (TRp.outcome == "sought_any") & (TRp.term == "yr10")].iloc[0])
m2 = MC[MC.model == "M2"]; m0 = MC[MC.model == "M0"]
n_m1 = int(((MC.model == "M1") & (MC.elpd_diff_vs_M2 > 2 * MC.se_diff_vs_M2)).sum())
mx_m1 = MC[(MC.model == "M1") & (MC.elpd_diff_vs_M2 > 2 * MC.se_diff_vs_M2)].elpd_diff_vs_M2.max()
pb = sorted(x for x in m2[["pB_chi2", "pB_sd", "pB_max", "pB_min"]].values.ravel() if x >= 0.05)
hashes = [l.split("  ", 2) for l in (R / "docs/preregistration/PREREGISTRATION_hashes.log").read_text().splitlines() if re.search(r"PREREGISTRATION_3[378]", l) and not re.search(r"climate|A8\.1 ", l, re.I)]   # climate work is ongoing: not listed

md = f"""# Help-seeking after violence among adolescents and young people in sub-Saharan Africa

Code for a multi-country analysis of disclosure and help-seeking after physical and sexual violence among young people aged 13–24 years,
using **{nS} Demographic and Health Surveys (DHS) from {nC} countries** and **eight Violence Against Children and Youth Surveys (VACS)**.

> **Status:** manuscript in preparation (Oxford Martin Programme on Climate and Child Health / Accelerate Hub, University of Oxford).
> Results below are from the current pre-registered analysis and may change before publication. The survey data are **not** included (see [Data](#2-data)).

**Contents:** [1 Questions](#1-questions) · [2 Data](#2-data) · [3 Pipeline](#3-analysis-pipeline) · [4 Model](#4-model-construction) · [5 Results](#5-results) · [6 Reproduce](#6-repository-structure-and-how-to-reproduce) · [7 Pre-registration](#7-pre-registration) · [8 Software](#8-software)

---

## 1. Questions

1. How many young survivors tell someone, seek help and reach formal services, and how does this differ across African regions?
2. Which factors are associated with help-seeking, and are they the same across regions?
3. Is help-seeking among young people changing over time?
4. Do DHS and VACS agree, and can their associations be pooled?

## 2. Data

| | DHS | VACS |
|---|---|---|
| Respondents | {nD:,} women aged 15–24 years who reported physical or sexual violence | {nV:,} girls and boys aged 13–24 years ({nVany:,} reported violence) |
| Coverage | {nS} surveys, {nC} countries, 2004–2025, West, Central, East and Southern Africa | 8 countries, one round each, 2013–2023, East and Southern Africa |
| Violence | physical since age 15; sexual by partners and others | sexual and physical asked separately |
| Help-seeking | told anyone; sought help and from whom (lifetime) | told; knew where to get help; sought help; received services (lifetime) |
| Weights | domestic violence module weight `d005/1e6` (sums to sample size) | expansion weights, normalised within country |
| Role in the analysis | cascade, factors, regional comparison, replication, trends | girls vs boys; sexual vs physical violence; ages 13–14 |

Percentages are **never pooled across surveys**; associations are pooled only in a pre-specified harmonised analysis (§4.4).

**Data access.** DHS data are available from the [DHS Program](https://dhsprogram.com) on registration (Individual Recode files, `*IR*DT.zip`).
VACS public-use data are available from Together for Girls and the US CDC. The licences do not allow redistribution, so this repository contains code only.

## 3. Analysis pipeline

```mermaid
flowchart TD
    A1["DHS Individual Recode files<br/>(raw .dta in .zip)"] --> B1["41_build_disclosure_women.py<br/>violence, outcomes, predictors"]
    B1 --> B2["41c_add_hh_head.py<br/>sex of household head (v151)"]
    B1 --> B3["41d_correction4.py<br/>formal sources, perpetrator audit"]
    A2["VACS public-use files"] --> B4["20_build_vacs_disclosure.py<br/>harmonised indicators"]
    B2 --> D["Analysis sample<br/>{nD:,} young women, {nS} surveys"]
    B3 --> D
    D --> E1["59_A8_cascade_ci.R<br/>cascade, design-based CIs"]
    D --> E2["50_disclosure_predictors.R<br/>stage 1: one model per country"]
    B4 --> E2
    E2 --> F1["58_A8_pool_numpyro.py<br/>stage 2: Bayesian hierarchical pooling"]
    D --> E3["52_disclosure_trends.R<br/>trends across rounds"]
    D --> E5["83 / 84 / 85<br/>trend curves: survey-level<br/>meta-regression + formal tests"]
    E5 --> I
    E3 --> F1
    D --> E4["69_southern_africa_pooled.R<br/>exploratory Southern Africa"]
    D --> G1["73 / 74<br/>harmonised DHS–VACS pool"]
    B4 --> G1
    F1 --> H1["81_bayes_workflow.py<br/>prior/posterior predictive checks,<br/>DIC, WAIC, PSIS-LOO"]
    D --> H2["78 / 79 / 80<br/>sensitivity analyses"]
    F1 --> I["61b / 61c / 61d<br/>Lancet-style figures"]
    E1 --> I
    H1 --> J["71 + manuscript_appendix.py<br/>LaTeX manuscript and appendix"]
    H2 --> J
    G1 --> J
    I --> J
    J --> K["72_verify_manuscript_numbers.py<br/>independent check of every key number"]
```

## 4. Model construction

### 4.1 Two-stage individual participant data design

```mermaid
flowchart LR
    subgraph S1["Stage 1: within each country c"]
        direction TB
        a["Women i in survey rounds s"] --> b["Survey-weighted logistic regression<br/>round fixed effects α_s<br/>PSU-clustered sandwich SEs"]
        b --> c["Country log odds ratio β̂_c<br/>and standard error s_c"]
    end
    subgraph S2["Stage 2: Bayesian hierarchical pooling"]
        direction TB
        mu["μ: pooled log OR<br/>μ ~ N(0, 1)"] --> mur["μ_r: regional means<br/>μ_r = μ + ω·u_r, ω ~ HalfNormal(0.5)"]
        mur --> th["θ_c: true country effects<br/>θ_c = μ_r(c) + τ·z_c, τ ~ HalfNormal(0.5)"]
        th --> obs["β̂_c ~ N(θ_c, s_c²)"]
    end
    c --> obs
    obs --> out["Pooled and regional ORs, 95% CrIs,<br/>prediction intervals, regional agreement"]
```

### 4.2 Equations

Survey-weighted proportion with a logit-scale 95% CI (Taylor linearisation, PSU clustering):

$$\\hat p=\\frac{{\\sum_i w_i y_i}}{{\\sum_i w_i}},\\qquad \\mathrm{{logit}}(\\hat p)\\pm 1.96\\frac{{\\widehat{{\\mathrm{{SE}}}}(\\hat p)}}{{\\hat p(1-\\hat p)}}$$

Stage 1, country model for woman *i* in survey round *s(i)*, fitted by weighted pseudo-likelihood with cluster-robust standard errors:

$$\\mathrm{{logit}}\\Pr(Y_i=1\\mid\\mathbf x_i)=\\alpha_{{s(i)}}+\\mathbf x_i^{{\\top}}\\boldsymbol\\beta_c,\\qquad
\\widehat{{\\mathbf V}}(\\hat{{\\boldsymbol\\beta}}_c)=\\tfrac{{G}}{{G-1}}\\mathbf A^{{-1}}\\Big(\\sum_g\\mathbf u_g\\mathbf u_g^{{\\top}}\\Big)\\mathbf A^{{-1}}$$

Stage 2, countries within UN regions (non-centred):

$$\\hat\\beta_{{ck}}\\sim\\mathcal N(\\theta_{{ck}},s_{{ck}}^2),\\quad \\theta_{{ck}}=\\mu_{{r(c)k}}+\\tau_k z_c,\\quad \\mu_{{rk}}=\\mu_k+\\omega_k u_r,\\quad z_c,u_r\\sim\\mathcal N(0,1)$$

Approximate weight of country *c* in region *r* in the pooled estimate:

$$\\lambda_c=\\frac{{W_r}}{{\\sum_{{r'}}W_{{r'}}}}\\cdot\\frac{{w_c}}{{\\sum_{{c'\\in r}}w_{{c'}}}},\\qquad w_c=\\frac{{1}}{{s_c^2+\\tau^2}},\\qquad W_r=\\Big(\\frac{{1}}{{\\sum_{{c\\in r}}w_c}}+\\omega^2\\Big)^{{-1}}$$

### 4.3 Priors and Bayesian workflow

| Parameter | Prior | Implication |
|---|---|---|
| μ (pooled log OR) | Normal(0, 1) | pooled OR between 0.14 and 7.1 with 95% prior probability |
| ω (between-region SD) | Half-Normal(0.5) | regional ORs may differ up to about 2.7-fold |
| τ (between-country SD) | Half-Normal(0.5) | as for ω |
| δ (DHS–VACS difference) | Normal(0, 1) | ratio of ORs between surveys within 0.14–7.1 |

```mermaid
flowchart LR
    p1["Prior predictive<br/>simulation"] --> p2["NUTS sampling<br/>4 × (2000 + 2000)"] --> p3["Diagnostics<br/>R̂ < 1.01, ESS > 400,<br/>divergences, E-BFMI"] --> p4["Posterior predictive<br/>checks, Bayesian p-values"] --> p5["Model comparison<br/>M0 / M1 / M2:<br/>DIC, WAIC, PSIS-LOO"] --> p6["Sensitivity<br/>priors, effect scale,<br/>timing, one-stage"]
```

M0 = common effect; M1 = countries exchangeable, no regional level; M2 = countries within regions (primary).

### 4.4 Harmonised DHS–VACS pooling

$$\\theta_{{jk}}=\\mu_k+\\delta_k x_j+\\omega_k u_{{r(j)}}+\\tau_k z_j,\\qquad x_j=-\\tfrac12\\ (\\text{{DHS}}),\\ +\\tfrac12\\ (\\text{{VACS}})$$

exp(μ) is the pooled OR (programmes weighted equally), exp(μ ∓ δ/2) the DHS and VACS ORs, and exp(δ) their ratio; the surveys *agree* if the 95% CrI of exp(δ) includes 1 and both estimates share a sign with probability > 0.9.

## 5. Results

### 5.1 The disclosure cascade (DHS, young women 15–24 years)

| Step | All countries (sample-weighted) | Population-weighted | West | Central | East | Southern |
|---|---|---|---|---|---|---|
| Told anyone | {pc(cas('told'))} | {f1(pw('told').pct)}% | {f1(cas('told', 'region', 'West').pct)}% | {f1(cas('told', 'region', 'Central').pct)}% | {f1(cas('told', 'region', 'East').pct)}% | {f1(cas('told', 'region', 'Southern').pct)}% |
| Sought any help | {pc(cas('sought_any'))} | {f1(pw('sought_any').pct)}% | {f1(cas('sought_any', 'region', 'West').pct)}% | {f1(cas('sought_any', 'region', 'Central').pct)}% | {f1(cas('sought_any', 'region', 'East').pct)}% | {f1(cas('sought_any', 'region', 'Southern').pct)}% |
| Sought formal help | {pc(cas('formal'))} | {f1(pw('formal').pct)}% | {f1(cas('formal', 'region', 'West').pct)}% | {f1(cas('formal', 'region', 'Central').pct)}% | {f1(cas('formal', 'region', 'East').pct)}% | {f1(cas('formal', 'region', 'Southern').pct)}% |

Help came mostly from family ({hs('src_family'):.0f}%), friends or neighbours ({hs('src_friends_neighbours'):.0f}%) and the partner's family ({hs('src_partner_family'):.0f}%);
{hs('src_police'):.1f}% went to the police and {hs('src_health'):.1f}% to a health worker.
In VACS, after sexual violence, {f1(vac('tell', 'girls').pct)}% of girls and {f1(vac('tell', 'boys').pct)}% of boys told someone, and {f1(vac('sought', 'girls').pct)}% and {f1(vac('sought', 'boys').pct)}% sought help.

<p align="center"><img src="docs/figures/fig_cascade_forest_lancet.png" width="760" alt="Disclosure cascade by country"></p>

### 5.2 Factors associated with seeking any help

Odds ratios (95% credible intervals) from the two-stage model; the last column is the 95% prediction interval for a new country.

| Factor | West | Central | East | Southern (exploratory) | All regions | Prediction interval |
|---|---|---|---|---|---|---|
""" + "\n".join(rows_t1) + f"""

ᵃ Among women with partner violence. Southern Africa: one pooled model of Lesotho, Namibia and South Africa (95% CI), not part of "all regions".
Injury, severe violence and both types of violence **replicated in two independent random halves of countries**; wealth, education and urban residence were null
in crude, fully adjusted and reduced socio-demographic models.

<p align="center"><img src="docs/figures/fig2_forest_injury_severe_lancet.png" width="760" alt="Injury and severe violence by country"></p>

### 5.3 Robustness of the four main factors

| Analysis | """ + " | ".join(l for _, _, l in F4) + " |\n|---|---|---|---|---|\n" + "\n".join(f"| {a} | " + " | ".join(v) + " |" for a, v in rob) + f"""

### 5.4 Pooling DHS and VACS (young women and girls 15–24 years)

| Outcome · factor | DHS | VACS | Ratio VACS/DHS | Pooled | Surveys |
|---|---|---|---|---|---|
{hrow('sought_any', 'both', 'Sought help · both types of violence')}
{hrow('told', 'both', 'Told anyone · both types of violence')}
{hrow('told', 'hh_female', 'Told anyone · female household head')}
{hrow('sought_any', 'hh_female', 'Sought help · female household head')}

### 5.5 Help-seeking has fallen since 2005

Survey-level estimates for all {nS} surveys, pooled with a Bayesian hierarchical meta-regression (country intercepts and slopes nested in regional and overall means, survey-level residual), with a frequentist random-effects meta-regression as a formal test:

$$y_{{cs}}\\sim\\mathcal N(a_c+b_c t_{{cs}},\\ se_{{cs}}^2+\\sigma_e^2),\\quad b_c=B_{{r(c)}}+\\sigma_b z_c,\\quad B_r=B+\\omega_b u_r$$

| Outcome | All countries: OR per decade (95% CrI) | Pr(decline) | Change 2005–2025 | Meta-regression | West | Central | East |
|---|---|---|---|---|---|---|---|
""" + "\n".join(f"| {lab} | " + (lambda r: f"{ci(r.OR_decade, r.OR_lo, r.OR_hi)} | {f2(r.P_decline)} | {f1(r.pp_2005_2025)} pp ({f1(r.pp_lo)} to {f1(r.pp_hi)}) | ")(TS[(TS.outcome == o) & (TS.scope == 'All')].iloc[0]) +
                 f"{ci(TMf.loc[o, 'OR_decade'], TMf.loc[o, 'lo'], TMf.loc[o, 'hi'])}, " + ("p<0.001" if TMf.loc[o, 'p_slope'] < 0.001 else f"p={TMf.loc[o, 'p_slope']:.2f}") + " | " + " | ".join((lambda r: ci(r.OR_decade, r.OR_lo, r.OR_hi))(TS[(TS.outcome == o) & (TS.scope == g)].iloc[0]) for g in ["West", "Central", "East"]) + " |"
                 for o, lab in [("told", "Told anyone"), ("sought_any", "Sought any help"), ("formal", "Sought formal help")]) + f"""

Within countries, adjusted for the women's characteristics, the odds of seeking any help fell by {100 * (1 - TR[(TR.model == "adjusted") & (TR.outcome == "sought_any")].OR.iloc[0]):.0f}% per decade ({tr('sought_any')}; {tr6} in DHS phase 6 or later).
Southern Africa has no repeated surveys, so its trend is not estimated.

<p align="center"><img src="docs/figures/fig_trends_lancet.png" width="760" alt="Trends in disclosure and help-seeking 2004-2025"></p>

### 5.6 Model checking

* **Prior predictive:** OR for a new country 95% range {f2(PP['lo'])}–{f2(PP['hi'])}.
* **Posterior predictive:** the common-effect model M0 misfitted for {int((m0.pB_chi2 < 0.05).sum())} of {len(m0)} factors (Bayesian p < 0.05); the primary model M2 reproduced every discrepancy (p_B {f2(pb[0])}–{f2(pb[-1])}) except the largest estimate for both types of violence (Togo; excluding it, {ci(TG.OR_without_togo, TG.lo, TG.hi)}).
* **Model comparison:** M1 (no regions) was marginally better than M2 in {n_m1} of {2 * len(m2)} comparisons (Δelpd ≤ {f1(mx_m1)}, below the threshold of 4 for a meaningful difference); M2 was kept because regional comparison is a study question.
* **Computation:** R̂ ≤ {MC.rhat_max.max():.3f}; E-BFMI ≥ {f2(MC.ebfmi_min.min())}. Pooled estimates were reproduced independently with **brms/Stan** (`methods_walkthrough/75_disclosure_walkthrough_brms.Rmd`).

<p align="center"><img src="docs/figures/figS_ppc.png" width="700" alt="Posterior predictive checks"></p>

## 6. Repository structure and how to reproduce

| Folder | Scripts | Purpose |
|---|---|---|
| `build/` | `41_build_disclosure_women.py` (run with and without `--heldout`), `41c_add_hh_head.py`, `20_build_vacs_disclosure.py` | raw DHS/VACS files → analysis files |
| `review/` | `41d_correction4.py` | coding audit of help sources and perpetrator (run after `build/`) |
| `analysis/` | `50_disclosure_predictors.R` (env `P33_SCOPE=youth AGE_CONT=1`), `52_disclosure_trends.R`, `59_A8_cascade_ci.R`, `69_southern_africa_pooled.R`, `57_A8_victimisation_trend.R` | stage 1 models, cascade, trends |
| `pooling/` | `pool_numpyro.py`, `58_A8_pool_numpyro.py` | stage 2 Bayesian pooling (NumPyro) |
| `harmonised/` | `73_p37_country_models.R`, `74_p37_pool.py` | DHS–VACS pooling |
| `review/` | `78_review_country_models.R`, `79_review_pool.py`, `80_population_weights.R`, `81_bayes_workflow.py` | sensitivity analyses and Bayesian workflow |
| `analysis/` | `83_trend_survey_estimates.R`, `84_trend_model.py`, `85_trend_metafor.R` | trend curves, regional and overall pooling, formal tests |
| `figures/` | `61b`, `61c`, `61d`, `61e` | figures |
| `manuscript/` | `71_build_latex_manuscript.py`, `manuscript_appendix.py`, `72_verify_manuscript_numbers.py` | manuscript, appendix, independent number check |
| `methods_walkthrough/` | `76_step_by_step_guide_codes_weights.Rmd`, `75_disclosure_walkthrough_brms.Rmd` | teaching versions: survey codes and weights; the full analysis in tidyverse + brms |

Paths assume the project folder `~/Documents/OMS DHS Births Climate Policy` with the DHS files in `data/raw/dhs_ir/`; edit `ROOT` at the top of each script to use another location.

## 7. Pre-registration

Every analysis plan, amendment and result was written and hashed (SHA-256) before the analysis it describes was run.
The documents will be released with the paper; the time stamps and hashes below allow them to be checked then.

| Time stamp | SHA-256 (first 16) | Stage |
|---|---|---|
""" + "\n".join(f"| {h[0][:16]} | `{h[1][:16]}` | {h[2].replace('|', '/').replace('_', ' ')} |" for h in hashes if len(h) == 3) + f"""

## 8. Software

R 4.5 (survey, fixest, data.table, lme4, car, brms, cmdstanr, tidyverse) · Python 3 (pandas, NumPyro, JAX, ArviZ, PyMuPDF).
Bayesian sampling: No-U-Turn sampler, 4 chains × (2000 warm-up + 2000 draws), seed 20261023, target acceptance 0.995.

---
*Contact: Samuel Tekle, University of Oxford. Please cite the paper once published.*
"""
(REPO / "README.md").write_text(md); print("README written:", len(md.split()), "words")
