"""manuscript_appendix.py - builds the full supplementary appendix (LaTeX) for 71_build_latex_manuscript.py.
Every number is read from the result files of the pre-registered analyses (Pre-registrations 33, 37 and 38).
Decimal points are written as ordinary dots; the manuscript's lancet() post-processor converts them to mid-dots.
"""
import re, numpy as np, pandas as pd
from pathlib import Path

R = Path.home() / "Documents/OMS DHS Births Climate Policy"; V3 = R / "outputs/tables/P33_youth_v3"; P38 = R / "outputs/tables/P38_review"
f2 = lambda x: f"{x:.2f}"; f1 = lambda x: f"{x:.1f}"
ci = lambda o, l, h: f"{f2(o)} ({f2(l)}--{f2(h)})"
fmtn = lambda n: f"{int(n):,}".replace(",", "\\,")
LAB = {"injury": "Injury from partner violence (vs none)", "severe": "Severe physical violence", "alcohol": "Partner drinks alcohol",
       "vtypeboth": "Physical and sexual violence (vs physical only)", "maritalformer": "Formerly in union (vs currently in union)",
       "working": "Currently working", "educsecondary+": "Secondary or higher education (vs none)", "educprimary": "Primary education (vs none)",
       "wealthq": "Household wealth (per quintile)", "urban": "Urban residence (vs rural)", "agegrp": "Age (per year)"}
COUNTRY = {"AO": "Angola", "BF": "Burkina Faso", "BJ": "Benin", "BU": "Burundi", "CD": "DR Congo", "CI": "C\\^ote d'Ivoire", "CM": "Cameroon", "ET": "Ethiopia",
           "GA": "Gabon", "GH": "Ghana", "GM": "Gambia", "GN": "Guinea", "KE": "Kenya", "KM": "Comoros", "LB": "Liberia", "LS": "Lesotho", "MD": "Madagascar",
           "ML": "Mali", "MR": "Mauritania", "MW": "Malawi", "MZ": "Mozambique", "NG": "Nigeria", "NM": "Namibia", "RW": "Rwanda", "SL": "Sierra Leone",
           "SN": "Senegal", "TD": "Chad", "TG": "Togo", "TZ": "Tanzania", "UG": "Uganda", "ZA": "South Africa", "ZM": "Zambia", "ZW": "Zimbabwe"}
REGION = {**{c: "West" for c in ["BF", "BJ", "CI", "GH", "GM", "GN", "LB", "ML", "MR", "NG", "SL", "SN", "TG"]}, **{c: "Central" for c in ["AO", "CD", "CM", "GA", "TD"]},
          **{c: "East" for c in ["BU", "ET", "KE", "KM", "MD", "MW", "MZ", "RW", "TZ", "UG", "ZM", "ZW"]}, **{c: "Southern" for c in ["LS", "NM", "ZA"]}}

def texesc(t):
    t = t.replace("_", " ").replace("&", "and").replace("%", "\\%").replace("#", "\\#").replace("~", "about ")
    return t.replace(">=", "$\\ge$").replace("<=", "$\\le$").replace("<", "$<$").replace(">", "$>$").replace("^", "")

def table(label, caption, colspec, header, body, note, size="\\footnotesize", long=False, sep="3.5pt"):
    if long:
        return (f"{{{size}\\setlength{{\\tabcolsep}}{{{sep}}}\n\\begin{{longtable}}{{{colspec}}}\n\\caption{{{caption}}}\\label{{{label}}}\\\\\n\\toprule\n{header}\\\\\n\\midrule\n\\endfirsthead\n"
                f"\\toprule\n{header}\\\\\n\\midrule\n\\endhead\n\\bottomrule\n\\endlastfoot\n{body}\n\\end{{longtable}}\n\\vspace{{-0.6em}}\\noindent{{\\footnotesize {note}}}\\par}}\n")
    return (f"\\begin{{table}}[!htbp]\n\\centering\n\\begin{{threeparttable}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n{size}\n\\setlength{{\\tabcolsep}}{{{sep}}}\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header}\\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
            f"\\begin{{tablenotes}}\\footnotesize\n\\item {note}\n\\end{{tablenotes}}\n\\end{{threeparttable}}\n\\end{{table}}\n")

def build(harm_rows, n37d, n37v, src_table="", forest_figs=""):
    A2 = pd.read_csv(V3 / "A2_country_estimates.csv"); A3 = pd.read_csv(V3 / "A3_pooled_numpyro.csv")
    pool = lambda m, t, o="sought_any": A3[(A3.source == "DHS") & (A3.model == m) & (A3.outcome == o) & (A3.term == t)].iloc[0]
    T38 = pd.read_csv(P38 / "P38_pooled.csv")
    p38 = lambda a, m, t, o="sought_any": T38[(T38.analysis == a) & (T38.model == m) & (T38.outcome == o) & (T38.term == t)]
    out = ["\\clearpage\n\\section*{Supplementary appendix}\\label{sec:appendix}\n\\setcounter{table}{0}\\renewcommand{\\thetable}{S\\arabic{table}}\n"
           "\\setcounter{figure}{0}\\renewcommand{\\thefigure}{S\\arabic{figure}}\n"
           "This appendix reports the data, the coding audit, the Bayesian inference framework and its checks, and all pre-registered sensitivity analyses "
           "(Pre-registrations 33, 37 and 38). Unless stated, estimates are odds ratios (ORs) for seeking any help among young women aged 15--24 years who "
           "experienced violence, pooled with the hierarchical model of equation~\\ref{eq:hier}, as posterior medians with 95\\% credible intervals (CrIs).\n"]

    # ---------------- A. Data ----------------
    out.append("\\subsection*{A. Data and coding}\n"); out.append(src_table + "\n")
    W = pd.concat([pd.read_csv(R / f"data/derived/{f}", usecols=["survey", "country", "v012", "sought_any", "vtype", "d005"]) for f in ["women_disclosure.csv.gz", "women_disclosure_heldout.csv.gz"]])
    W = W[(W.survey != "SN_2019_8B") & W.sought_any.notna() & (W.vtype != "none") & (W.v012 <= 24)]
    pairs = pd.read_csv(R / "data/derived/ir_ge_pairs.csv"); pairs = pairs[pairs.usable]
    pairs["survey"] = pairs.country + "_" + pairs.years.astype(str).str[:4] + "_" + pairs.ir.str[4:6]
    g = W.groupby("survey").apply(lambda d: pd.Series({"n": len(d), "pct": 100 * np.average(d.sought_any, weights=d.d005)}), include_groups=False).reset_index()
    g = g.merge(pairs[["survey", "years"]], on="survey", how="left"); g["country"] = g.survey.str[:2]; g["region"] = g.country.map(REGION)
    g["phase"] = g.survey.str.split("_").str[2].str[0]
    g = g.sort_values(["region", "country", "years"], key=lambda s: s.map({"West": 0, "Central": 1, "East": 2, "Southern": 3}) if s.name == "region" else s)
    body = "\n".join(f"{COUNTRY[r.country]} & {str(r.years).replace('-', '--')} & {r.phase} & {r.region} & {fmtn(r.n)} & {f1(r.pct)} \\\\" for r in g.itertuples())
    out.append(table("tab:surveys", "\\textbf{DHS surveys included}", "llcccc", "Country & Years & Phase & UN region & Young women with violence & Sought any help, \\%",
                     body, f"{len(g)} surveys, {g.country.nunique()} countries, {fmtn(g.n.sum())} women aged 15--24 years who reported physical or sexual violence and were asked the help-seeking questions. Percentages are survey-weighted (DHS domestic violence weight).", long=True))

    CH = pd.read_csv(P38 / "P38_sample_characteristics.csv").set_index("region")
    rows = [("Women", "women", "n"), ("Surveys", "surveys", "n"), ("Countries", "countries", "n"), ("Mean age, years", "age_mean", "x"),
            ("Secondary or higher education", "secondary", "p"), ("Urban residence", "urban", "p"), ("Currently in union", "in_union", "p"),
            ("Formerly in union", "formerly_in_union", "p"), ("Never in union", "never_in_union", "p"), ("Currently working", "working", "p"),
            ("Physical violence only", "physical_only", "p"), ("Sexual violence only", "sexual_only", "p"), ("Both physical and sexual violence", "both_types", "p"),
            ("Violence by a partner", "partner_violence", "p"), ("Violence in the past 12 months", "past_12_months", "p"),
            ("Injury (among women with partner violence)", "injury_among_partner", "p"), ("Severe violence (among women with partner violence)", "severe_among_partner", "p")]
    cols = ["All", "West", "Central", "East", "Southern"]
    val = lambda v, k: fmtn(v) if k == "n" else f1(v)
    body = "\n".join(f"{lab} & " + " & ".join(val(CH.loc[c, col], k) for c in cols) + " \\\\" for lab, col, k in rows)
    out.append(table("tab:characteristics", "\\textbf{Characteristics of the young women in the DHS analysis, by UN region}", ">{\\raggedright\\arraybackslash}p{6cm}ccccc",
                     "& All & West & Central & East & Southern", body,
                     f"Survey-weighted percentages unless stated. Missing values: employment {fmtn(CH.loc['All', 'missing_working'])}, past-12-month violence {fmtn(CH.loc['All', 'missing_recent'])} (not recorded in phase 4 surveys), severe violence {fmtn(CH.loc['All', 'missing_severe'])}."))

    C4 = pd.read_csv(R / "data/derived/correction4_log.csv")
    clean = lambda s: re.sub(r":.*", "", str(s)).replace("&", "\\&")
    add = C4[C4.formal_added.notna()]; wide = C4[C4.wide_added.notna()]
    body = "\n".join(f"{COUNTRY[r.survey[:2]]} {r.survey[3:7]} & " + "; ".join(sorted({clean(x) for x in str(r.formal_added).split(' | ')})) + " & formal \\\\" for r in add.itertuples())
    body += "\n\\midrule\n" + "\n".join(f"{COUNTRY[r.survey[:2]]} {r.survey[3:7]} & " + "; ".join(sorted({clean(x) for x in str(r.wide_added).split(' | ')})) + " & wide definition only \\\\" for r in wide.itertuples())
    n_old = int(C4.perp_rule.str.startswith("d127 not").sum())
    out.append(table("tab:coding", "\\textbf{Coding audit (Correction 4): help sources reclassified as formal}", ">{\\raggedright\\arraybackslash}p{3.2cm}>{\\raggedright\\arraybackslash}p{8.6cm}l",
                     "Survey & Source label in the DHS file & Classified as", body,
                     "Labels of every help source (d119*) in every survey were read from the files. Sources meeting the paper's definition of formal help that the original pattern missed were reclassified (upper panel); "
                     "chiefs, local authorities and community leaders are formal only in the wide sensitivity definition (lower panel). "
                     f"Perpetrator of forced sex: d125 (forced sexual acts by anyone) was assigned to a partner or a non-partner using d127 (person who forced the first sexual act) where it was asked; in {n_old} of {len(C4)} surveys d127 was not asked and the original coding was kept. "
                     "In all surveys the original coding was first reproduced exactly from the raw files (100\\%) before the correction was applied."))

    HS = pd.read_csv(P38 / "P38_B8_help_sources.csv")
    SRC = [("src_family", "Own family"), ("src_friends_neighbours", "Friends or neighbours"), ("src_partner_family", "Partner's family"), ("src_partner", "Current or former partner"),
           ("src_police", "Police"), ("src_local_authority", "Chief or local authority"), ("src_religious_traditional", "Religious or traditional leader"),
           ("src_health", "Doctor or health worker"), ("src_social_ngo", "Social services, NGO, women's organisation or helpline"), ("src_legal", "Lawyer or court"), ("src_other", "Other")]
    piv = HS.pivot(index="source", columns="region", values="pct")
    cell = lambda s, r: "--" if pd.isna(piv.loc[s, r]) else f1(piv.loc[s, r])
    body = "\n".join(f"{lab} & " + " & ".join(cell(s, r) for r in cols) + " \\\\" for s, lab in SRC if s in piv.index)
    out.append(table("tab:helpsources", "\\textbf{Sources of help among young women who sought help, by UN region}", ">{\\raggedright\\arraybackslash}p{6cm}ccccc",
                     "& All & West & Central & East & Southern", body,
                     "Survey-weighted percentages of young women who sought any help; a woman may report several sources. Each category's denominator is restricted to surveys that listed that category; -- = not listed in that region's surveys."))

    # ---------------- B. Bayesian inference ----------------
    PP = pd.read_csv(P38 / "P38_B12_prior_predictive.csv"); MC = pd.read_csv(P38 / "P38_B12_model_checks.csv")
    new = PP.iloc[0]; sdp = PP.iloc[2]
    out.append("\\subsection*{B. Bayesian inference: priors, computation, model checking and comparison}\n"
               "We followed a Bayesian workflow:\\cite{gelman2020,gabry2019} (1) specify weakly informative priors and check their implications by prior predictive simulation; "
               "(2) fit candidate models with Hamiltonian Monte Carlo and check computation; (3) check fit with posterior predictive checks and Bayesian $p$-values;\\cite{gelman1996} "
               "(4) compare models by out-of-sample predictive accuracy;\\cite{spiegelhalter2002,vehtari2017,watanabe2010} and (5) test sensitivity to the priors.\n\n")
    body = ("$\\mu_k$ (pooled log OR) & Normal(0, 1) & 95\\% prior range of the pooled OR 0.14--7.1; centred on no association \\\\\n"
            "$\\omega_k$ (between-region SD) & Half-Normal(0.5) & 95\\% prior mass below 0.98 on the log-odds scale: regional ORs may differ up to about 2.7-fold \\\\\n"
            "$\\tau_k$ (between-country SD) & Half-Normal(0.5) & As for $\\omega_k$; allows large between-country differences while ruling out implausible ones \\\\\n"
            "$\\delta_k$ (DHS--VACS difference, equation~\\ref{eq:harm}) & Normal(0, 1) & Ratio of ORs between surveys within 0.14--7.1 with 95\\% prior probability \\\\")
    out.append(table("tab:priors", "\\textbf{Prior distributions and their justification}", ">{\\raggedright\\arraybackslash}p{4.2cm}l>{\\raggedright\\arraybackslash}p{8.2cm}",
                     "Parameter & Prior & Justification", body,
                     f"Prior predictive simulation of the primary model (injury design: {MC[(MC.term == 'injury') & (MC.outcome == 'sought_any') & (MC.model == 'M2')].countries.iloc[0]} countries with their standard errors): "
                     f"the OR for a new country had a 95\\% prior predictive range of {f2(new['lo'])}--{f2(new['hi'])}, with {f1(100 * new['share_outside_0p1_10'])}\\% outside 0.1--10, "
                     f"and the SD of the country log ORs a range of {f2(sdp['lo'])}--{f2(sdp['hi'])}; the observed SD for injury ({f2(new['observed_injury_SD_of_log_ORs'])}) lies within it (figure~\\ref{{fig:priorpred}}). "
                     "The priors therefore allow every plausible association and between-country spread while excluding implausibly extreme values."))
    out.append("\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=0.95\\textwidth]{figures/figS_prior_predictive.png}\n"
               "\\caption{\\textbf{Prior predictive checks of the hierarchical model}\\\\ (A) Prior predictive distribution of the odds ratio for a new country (log scale); "
               "ticks mark the observed country estimates for injury. (B) Prior predictive distribution of the standard deviation of 21 country log odds ratios; "
               "the dashed line marks the observed value for injury. Simulated from the priors only, using the 21 countries' standard errors and regions.}\n\\label{fig:priorpred}\n\\end{figure}\n")
    TL = {"sought_any": "Sought", "told": "Told"}
    M = MC.pivot_table(index=["outcome", "term"], columns="model", values=["DIC", "WAIC", "elpd_loo", "elpd_diff_vs_M2", "se_diff_vs_M2"], sort=False)
    order = MC[["outcome", "term"]].drop_duplicates().itertuples(index=False)
    def dif(o, t, m):
        d, s = M.loc[(o, t), ("elpd_diff_vs_M2", m)], M.loc[(o, t), ("se_diff_vs_M2", m)]
        mark = "$^{*}$" if d > 2 * s else ""
        return f"{f1(d)} ({f1(s)}){mark}"
    body = "\n".join(f"{LAB[t].split(' (')[0]} & {TL[o]} & " + " & ".join(f1(M.loc[(o, t), ('DIC', m)]) for m in ["M0", "M1", "M2"]) + " & " +
                     " & ".join(f1(M.loc[(o, t), ('WAIC', m)]) for m in ["M0", "M1", "M2"]) + f" & {dif(o, t, 'M0')} & {dif(o, t, 'M1')} \\\\" for o, t in order)
    n_star = int(((MC.model != "M2") & (MC.elpd_diff_vs_M2 > 2 * MC.se_diff_vs_M2)).sum())
    out.append(table("tab:ic", "\\textbf{Model comparison: DIC, WAIC and leave-one-country-out cross-validation}", ">{\\raggedright\\arraybackslash}p{3.6cm}cccccccccc",
                     "& & \\multicolumn{3}{c}{DIC} & \\multicolumn{3}{c}{WAIC} & \\multicolumn{2}{c}{$\\Delta$elpd vs M2 (SE)}\\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}\nFactor & Outcome & M0 & M1 & M2 & M0 & M1 & M2 & M0 & M1",
                     body,
                     "M0 = common effect; M1 = countries exchangeable (no regional level); M2 = countries within regions (primary). Lower DIC and WAIC and higher expected log predictive density (elpd, PSIS-LOO) indicate better predictive accuracy; "
                     "all computed from the likelihood marginal over the country effect. $\\Delta$elpd $>0$ favours the alternative over M2. "
                     f"$^{{*}}$Alternative better than M2 by more than two standard errors (the pre-specified rule): {n_star} of {2 * len(MC.drop_duplicates(['outcome', 'term']))} comparisons, all for M1 and all with $|\\Delta\\text{{elpd}}|\\le{f1(MC[(MC.model == 'M1') & (MC.elpd_diff_vs_M2 > 2 * MC.se_diff_vs_M2)].elpd_diff_vs_M2.max())}$, "
                     "below the threshold of 4 at which differences are considered meaningful;\\cite{sivula2022} M2 was retained because regional comparison is a study question, and its pooled ORs were nearly identical to M1 with wider, more conservative intervals.", sep="3pt"))
    M2 = MC[MC.model == "M2"]; M0 = MC[MC.model == "M0"].set_index(["outcome", "term"])
    TG = pd.read_csv(P38 / "P38_B12_togo_exclusion.csv").set_index("outcome").loc["sought_any"]
    body = "\n".join(f"{LAB[r.term].split(' (')[0]} & {TL[r.outcome]} & {f2(M0.loc[(r.outcome, r.term), 'pB_chi2'])} & {f2(r.pB_chi2)} & {f2(r.pB_sd)} & {f2(r.pB_max)} & {f2(r.pB_min)} & "
                     f"{f2(r.pareto_k_max)} & {r.rhat_max:.3f} & {int(r.ess_bulk_min)} & {int(r.divergences)} & {f2(r.ebfmi_min)} \\\\" for r in M2.itertuples())
    out.append(table("tab:ppc", "\\textbf{Posterior predictive checks and computational diagnostics}", ">{\\raggedright\\arraybackslash}p{3.4cm}ccccccccccc",
                     "& & \\multicolumn{5}{c}{Bayesian $p$-value} & & & & & \\\\\n\\cmidrule(lr){3-7}\nFactor & Outcome & $\\chi^2$ (M0) & $\\chi^2$ & SD & Max & Min & $\\hat k$ max & $\\hat R$ max & ESS min & Div. & E-BFMI",
                     body,
                     "Bayesian $p$-values $p_B=\\Pr\\{T(y^{\\text{rep}},\\theta)\\ge T(y,\\theta)\\}$ for discrepancies $T$: $\\chi^2=\\sum_c(\\hat\\beta_c-\\theta_c)^2/s_c^2$, the SD, maximum and minimum of the country estimates; values below 0.05 or above 0.95 indicate misfit. "
                     "Columns from $\\chi^2$ onwards refer to the primary model M2; the first column is the common-effect model M0, whose misfit for several factors shows that between-country heterogeneity is needed. "
                     "$\\hat k$ = largest Pareto shape in PSIS-LOO (above 0.7: that country is influential and its LOO estimate less reliable). Div. = divergent transitions out of 8000 draws; E-BFMI above 0.3 indicates adequate exploration. "
                     f"For both types of violence the most extreme country (Togo, OR {f2(TG.togo_OR)}) exceeded most replicated maxima ($p_B$ for the maximum {f2(M2[(M2.outcome == 'sought_any') & (M2.term == 'vtypeboth')].pB_max.iloc[0])}); excluding Togo changed the pooled OR from "
                     f"{f2(MC[(MC.model == 'M2') & (MC.outcome == 'sought_any') & (MC.term == 'vtypeboth')].OR.iloc[0])} to {ci(TG.OR_without_togo, TG.lo, TG.hi)}.", sep="2.6pt"))
    out.append("\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=0.95\\textwidth]{figures/figS_ppc.png}\n"
               "\\caption{\\textbf{Posterior predictive checks of the primary model for the four main factors}\\\\ Upper row: realised $\\chi^2$ discrepancy against its replicated value for each posterior draw; "
               "points above the line are replicates more discrepant than the data, and their share is the Bayesian $p$-value shown above each panel. Lower row: replicated SD of the country log ORs (grey) against the observed SD (dashed line).}\n\\label{fig:ppc}\n\\end{figure}\n")
    B1 = pd.read_csv(P38 / "P38_B1_prior_sensitivity.csv")
    def b1(m, t, s): r = B1[(B1.model == m) & (B1.term == t) & (B1.omega_prior_sd == s)].iloc[0]; return r
    body = "\n".join(f"{LAB[t]} & {ci(b1(m, t, .25).OR, b1(m, t, .25).OR_lo, b1(m, t, .25).OR_hi)} & {ci(b1(m, t, .5).OR, b1(m, t, .5).OR_lo, b1(m, t, .5).OR_hi)} & "
                     f"{ci(b1(m, t, 1.).OR, b1(m, t, 1.).OR_lo, b1(m, t, 1.).OR_hi)} & {f2(b1(m, t, .5).PI_lo)}--{f2(b1(m, t, .5).PI_hi)} \\\\"
                     for m, t in B1[["model", "term"]].drop_duplicates().itertuples(index=False))
    out.append(table("tab:priorsens", "\\textbf{Sensitivity to the prior for between-region variation, and 95\\% prediction intervals for a new country}", ">{\\raggedright\\arraybackslash}p{5.2cm}cccc",
                     "& \\multicolumn{3}{c}{Pooled OR (95\\% CrI) with $\\omega\\sim$ Half-Normal($s$)} & \\\\\n\\cmidrule(lr){2-4}\nFactor & $s=0.25$ & $s=0.5$ (primary) & $s=1$ & 95\\% prediction interval",
                     body, "Point estimates do not depend on the prior; with three regions, the width of the all-region interval does. The prediction interval, $\\exp(\\mu+\\omega u_{\\text{new}}+\\tau z_{\\text{new}})$ under the primary prior, is the range of ORs expected in a country not in the data.\\cite{riley2011}"))

    # ---------------- B2. Country forest plots and trend model ----------------
    out.append("\\subsection*{Factors associated with help-seeking, by country}\n" + forest_figs + "\n")
    TS = pd.read_csv(P38 / "P39_slopes.csv"); TN = pd.read_csv(P38 / "P39_slopes_noresid.csv"); TM = pd.read_csv(P38 / "P39_metafor_tests.csv").set_index("outcome")
    OLt = {"told": "Told anyone", "sought_any": "Sought any help", "formal": "Sought formal help"}
    body = []
    for o in OLt:
        for sc in ["All", "West", "Central", "East", "Southern"]:
            r = TS[(TS.outcome == o) & (TS.scope == sc)].iloc[0]; r0 = TN[(TN.outcome == o) & (TN.scope == sc)].iloc[0]
            lab = (OLt[o] if sc == "All" else "") ; scl = "All countries" if sc == "All" else sc + (" (not estimable)" if sc == "Southern" else "")
            pv = TM.loc[o, 'p_slope']; pt = "p$<$0.001" if pv < 0.001 else f"p={pv:.2f}"
            body.append(f"{lab} & {scl} & {int(r.surveys)} & {ci(r.OR_decade, r.OR_lo, r.OR_hi)} & {f2(r.P_decline)}" +
                        f" & {ci(r0.OR_decade, r0.OR_lo, r0.OR_hi)} & " + (f"{ci(TM.loc[o, 'OR_decade'], TM.loc[o, 'lo'], TM.loc[o, 'hi'])}, {pt}" if sc == "All" else "") + " \\\\")
        body.append("\\addlinespace")
    out.append(table("tab:trendmodel", "\\textbf{Trends in the cascade, 2004--2025: survey-level hierarchical meta-regression}", ">{\\raggedright\\arraybackslash}p{1.9cm}lccccc",
                     "Outcome & Scope & Surveys & OR per decade (95\\% CrI) & Pr(decline) & Without residual & Meta-regression", "\n".join(body[:-1]),
                     "Primary model: $y_{cs}\\sim\\mathcal N(a_c+b_c t_{cs}, se_{cs}^2+\\sigma_e^2)$, $a_c=A_{r(c)}+\\sigma_a z^a_c$, $b_c=B_{r(c)}+\\sigma_b z^b_c$, $A_r=A+\\omega_a u^a_r$, $B_r=B+\\omega_b u^b_r$; $y$ = logit of the survey-weighted proportion with its design-based standard error; $t$ = decades from 2015. "
                     "Priors: $A\\sim\\mathcal N(\\text{logit }0.3, 1.5)$ (0.03 for formal help), $B\\sim\\mathcal N(0,1)$, $\\sigma_a,\\omega_a\\sim$ Half-Normal(1), $\\sigma_b,\\omega_b,\\sigma_e\\sim$ Half-Normal(0.5). "
                     "The registered model had no survey residual ($\\sigma_e$); it was added before interpretation because the residual variation between a country's surveys would otherwise be treated as sampling error (Pre-registration 39, deviation 1). "
                     "Random-effects meta-regression: metafor, REML, random intercepts and slopes by country and a survey-level residual; Wald test of the slope. Southern Africa has no repeated surveys, so its slope is informed only by the other regions. "
                     "One survey with no formal-help events (Mozambique 2022) used a 0.5 continuity correction. Change from 2005 to 2025 (overall curve): " + "; ".join(f"{OLt[o].lower()} {f1(TS[(TS.outcome == o) & (TS.scope == 'All')].pp_2005_2025.iloc[0])} percentage points ({f1(TS[(TS.outcome == o) & (TS.scope == 'All')].pp_lo.iloc[0])} to {f1(TS[(TS.outcome == o) & (TS.scope == 'All')].pp_hi.iloc[0])})" for o in OLt) + ".", size="\\scriptsize", sep="3pt"))

    # ---------------- B3. Bayesian workflow for the trend model (Pre-registration 39, addendum B) ----------------
    PPt = pd.read_csv(P38 / "P39B_prior_predictive.csv").set_index("outcome"); MCt = pd.read_csv(P38 / "P39B_model_checks.csv"); SNt = pd.read_csv(P38 / "P39B_prior_sensitivity.csv")
    OLt2 = {"told": "Told anyone", "sought_any": "Sought any help", "formal": "Sought formal help"}
    MN = {"T0": "T0 no trend", "T1": "T1 common trend", "T2": "T2 country trends", "T3": "T3 primary"}
    out.append("\\subsection*{Bayesian workflow for the trend model}\n"
               "The trend meta-regression was checked with the same workflow as the pooling model: prior predictive simulation with the actual survey design; "
               "computational diagnostics; posterior predictive checks with Bayesian $p$-values, including a trend discrepancy (the least-squares slope of the survey logits on time); "
               "comparison of four models by DIC, WAIC and leave-one-survey-out cross-validation; and prior sensitivity (Pre-registration 39, addendum B).\n")
    body = "\n".join(f"{OLt2[o]} & {f1(r.prev_lo)}--{f1(r.prev_hi)} & {f2(r.OR_decade_lo)}--{f2(r.OR_decade_hi)} & {f2(r.sd_logit_lo)}--{f2(r.sd_logit_hi)} & {f2(r.sd_logit_obs)} & {f2(r.slope_lo)} to {f2(r.slope_hi)} & {f2(r.slope_obs)} \\\\" for o, r in PPt.iterrows())
    out.append(table("tab:trendprior", "\\textbf{Prior predictive checks of the trend model}", ">{\\raggedright\\arraybackslash}p{2.6cm}cccccc",
                     "& \\multicolumn{3}{c}{95\\% prior predictive range} & & & \\\\\n\\cmidrule(lr){2-4}\nOutcome & Prevalence, \\% & OR per decade & SD of logits & Observed SD & Slope per decade & Observed slope",
                     body, "Simulated from the priors only, using each outcome's survey times, standard errors, countries and regions (8000 draws). The priors admit every plausible prevalence and trend. "
                     "For telling and seeking help the observed between-survey spread is smaller than the priors anticipate, so the priors are conservative rather than in conflict with the data; estimates were unchanged under narrower slope priors (table~\\ref{tab:trendsens}).", sep="3pt"))
    rows = []
    for o in OLt2:
        for m in ["T0", "T1", "T2", "T3"]:
            r = MCt[(MCt.outcome == o) & (MCt.model == m)].iloc[0]
            orr = ci(r.OR_decade, r.OR_lo, r.OR_hi) if pd.notna(r.get("OR_decade", np.nan)) else "--"
            dl = "ref." if m == "T3" else f"{f1(r.delta_elpd_vs_T3)} ({f1(r.se_delta)})"
            rows.append(f"{OLt2[o] if m == 'T0' else ''} & {MN[m]} & {orr} & {f1(r.DIC)} & {f1(r.WAIC)} & {dl} & {f2(r.pB_chi2)} & {f2(r.pB_sd)} & {f2(r.pB_max)} & {f2(r.pB_min)} & {f2(r.pB_slope)} & {r.rhat_max:.3f} & {int(r.ess_bulk_min)}/{int(r.ess_tail_min)} & {int(r.divergences)} \\\\")
        rows.append("\\addlinespace")
    out.append(table("tab:trendchecks", "\\textbf{Trend model: comparison, posterior predictive checks and diagnostics}", ">{\\raggedright\\arraybackslash}p{1.5cm}lcccccccccccc",
                     "& & & & & & \\multicolumn{5}{c}{Bayesian $p$-value} & & & \\\\\n\\cmidrule(lr){7-11}\nOutcome & Model & OR per decade & DIC & WAIC & $\\Delta$elpd (SE) & $\\chi^2$ & SD & Max & Min & Slope & $\\hat R$ & ESS bulk/tail & Div.", "\n".join(rows[:-1]),
                     "T0: no trend; T1: one trend for all countries; T2: country trends without regions; T3: country trends within regions (primary, figure~\\ref{fig:trends}). "
                     "$\\Delta$elpd: difference in expected log predictive density from leave-one-survey-out cross-validation (PSIS-LOO) relative to T3; positive values favour the alternative. "
                     "The pre-specified rule retains T3 unless an alternative is better by more than two standard errors and by more than 4; no alternative met it. "
                     "For seeking any help the no-trend model predicted worse and could not reproduce the observed decline (Bayesian $p$ for the slope 1.00), supporting the trend; "
                     "the common-trend model was marginally better with a narrower interval, so T3 is conservative. Tail ESS: minimum effective sample size of the 5\\% and 95\\% quantile indicators.", size="\\scriptsize", sep="2pt"))
    body = "\n".join(f"{OLt2[r.outcome]} & $\\mathcal N(0,{r.b_sd:g})$ & Half-Normal({r.slope_sd:g}) & {ci(r.OR_decade, r.OR_lo, r.OR_hi)} & {f2(r.P_decline)} \\\\" for r in SNt.itertuples())
    out.append(table("tab:trendsens", "\\textbf{Trend model: sensitivity to the priors}", ">{\\raggedright\\arraybackslash}p{3cm}cccc",
                     "Outcome & Prior on $B$ & Prior on $\\sigma_b,\\omega_b$ & OR per decade (95\\% CrI) & Pr(decline)", body,
                     "Primary priors: $B\\sim\\mathcal N(0,1)$ and $\\sigma_b,\\omega_b\\sim$ Half-Normal(0.5) (appendix table~\\ref{tab:trendmodel})."))
    out.append("\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=0.92\\textwidth]{figures/figS_trend_checks.png}\n"
               "\\caption{\\textbf{Prior and posterior predictive checks of the trend model}\\\\ Top row: prior predictive distribution of the standard deviation of the survey logits; "
               "middle row: prior predictive distribution of the slope of the survey logits per decade; bottom row: posterior predictive distribution of the slope under the primary model, with the Bayesian $p$-value. "
               "Dashed lines mark the observed values.}\n\\label{fig:trendchecks}\n\\end{figure}\n")

    # ---------------- C. Sensitivity analyses ----------------
    out.append("\\subsection*{C. Sensitivity analyses}\n")
    rows = []
    for m, t in [("partner", "injury"), ("partner", "severe"), ("partner", "alcohol"), ("main", "vtypeboth"), ("main", "maritalformer"), ("main", "working"),
                 ("main", "educsecondary+"), ("main", "wealthq"), ("main", "urban")]:
        cr = p38("B2_crude", m, t); ad = pool(m, t); rd = p38("B3_reduced", "main", t)
        rows.append(f"{LAB[t]} & " + (ci(cr.OR.iloc[0], cr.OR_lo.iloc[0], cr.OR_hi.iloc[0]) if len(cr) else "--") + f" & {ci(ad.OR, ad.OR_lo, ad.OR_hi)} & " +
                    (ci(rd.OR.iloc[0], rd.OR_lo.iloc[0], rd.OR_hi.iloc[0]) if len(rd) else "--") + " \\\\")
    out.append(table("tab:crude", "\\textbf{Crude, fully adjusted and reduced socio-demographic models}", ">{\\raggedright\\arraybackslash}p{5.6cm}ccc",
                     "Factor & Crude & Fully adjusted (Table 2) & Socio-demographic only", "\n".join(rows),
                     "Crude: each factor alone with survey-round fixed effects. Fully adjusted: all pre-specified predictors (equation~\\ref{eq:logit}). Socio-demographic only: age, education, wealth and urban residence with each other only, "
                     "because they precede employment, media exposure, decision-making, attitudes and the characteristics of the violence, and adjusting for those would remove part of their effect.\\cite{westreich2013} All pooled with equation~\\ref{eq:hier}."))
    OS = pd.read_csv(P38 / "P38_B6_onestage.csv").set_index("term")
    MCo = MC[MC.outcome == "sought_any"].set_index(["term", "model"])
    F4 = [("partner", "injury"), ("partner", "severe"), ("main", "vtypeboth"), ("main", "maritalformer")]
    def col(a): r = [p38(a, m, t) for m, t in F4]; return [ci(x.OR.iloc[0], x.OR_lo.iloc[0], x.OR_hi.iloc[0]) if len(x) else "--" for x in r]
    rdpp = [p38("B5_rd", m, t) for m, t in F4]; rdpp = [f"{f1(100 * np.log(x.OR.iloc[0]))} ({f1(100 * np.log(x.OR_lo.iloc[0]))} to {f1(100 * np.log(x.OR_hi.iloc[0]))})" for x in rdpp]
    lines = [("Primary (Table 2)", [ci(pool(m, t).OR, pool(m, t).OR_lo, pool(m, t).OR_hi) for m, t in F4]),
             ("Violence only in the past 12 months", col("B4_timing")),
             ("Adjusted prevalence ratio", col("B5_pr")),
             ("Adjusted risk difference, percentage points", rdpp),
             ("One-stage, survey-weighted logit (common effect)", [ci(OS.loc[t, "onestage_weighted_OR"], OS.loc[t, "lo1"], OS.loc[t, "hi1"]) for _, t in F4]),
             ("One-stage mixed model, random slope by country", [ci(OS.loc[t, "glmm_OR"], OS.loc[t, "lo2"], OS.loc[t, "hi2"]) for _, t in F4]),
             ("No regional level (M1)", [ci(MCo.loc[(t, "M1"), "OR"], MCo.loc[(t, "M1"), "OR_lo"], MCo.loc[(t, "M1"), "OR_hi"]) for _, t in F4])]
    body = "\n".join(f"{lab} & " + " & ".join(v) + " \\\\" for lab, v in lines)
    out.append(table("tab:robust", "\\textbf{The four main factors under alternative analyses}", ">{\\raggedright\\arraybackslash}p{4.6cm}cccc",
                     "Analysis & Injury & Severe violence & Both types & Formerly in union", body,
                     "Odds ratios (95\\% CrI or CI) unless stated. Past 12 months: women whose violence all occurred in the 12 months before the interview, so that lifetime help-seeking refers to recent violence. "
                     "Prevalence ratios from survey-weighted Poisson models with robust standard errors,\\cite{zou2004} risk differences from survey-weighted linear probability models, each pooled with equation~\\ref{eq:hier} (for risk differences with priors divided by 4 for the probability scale, $\\mu\\sim\\mathcal N(0,0.25)$ and $\\tau,\\omega\\sim$ Half-Normal(0.125), because the log-odds priors did not converge on this scale). "
                     "One-stage models: all countries in one survey-weighted logistic model with survey-round fixed effects and PSU-clustered standard errors, and an unweighted mixed logistic model with a random intercept for survey and a random slope for the factor by country (lme4).\\cite{bates2015}", sep="3pt"))
    PW = pd.read_csv(P38 / "P38_B9_population_weighted_cascade.csv")
    OL = {"told": "Told anyone", "sought_any": "Sought any help", "formal": "Sought formal help", "formal_wide": "Formal help, wide definition"}
    body = "\n".join(f"{OL[o]} & {w.split(' ')[0]} & " + " & ".join(
        (lambda x: f"{f1(x.pct)} ({f1(x.lo)}--{f1(x.hi)})")(PW[(PW.outcome == o) & (PW.weighting == w) & (PW.region == r)].iloc[0]) for r in cols) + " \\\\"
        for o in OL for w in ["sample-size (primary)", "population"])
    out.append(table("tab:popw", "\\textbf{Disclosure cascade weighted by sample size and by population}", ">{\\raggedright\\arraybackslash}p{2.5cm}lccccc",
                     "Step & Weighting & All & West & Central & East & Southern", body,
                     "Sample-size weighting (primary): DHS domestic violence weights, which sum to each survey's sample size, so surveys count in proportion to the women interviewed. "
                     "Population weighting: each country counts in proportion to its female population aged 15--24 years (WorldPop R2025A at the survey year; 2015 for earlier surveys), with a country's rounds sharing its weight equally. "
                     "Wide definition of formal help adds chiefs, local authorities and community leaders.", sep="2.2pt"))
    TRp = pd.read_csv(V3 / "A5_pooled_trends.csv")
    def trend(scope, o): x = TRp[(TRp.scope == scope) & (TRp.model == "adjusted") & (TRp.outcome == o) & (TRp.term == "yr10")]; return ci(x.OR.iloc[0], x.OR_lo.iloc[0], x.OR_hi.iloc[0]) + f" [{int(x.countries.iloc[0])}]" if len(x) else "--"
    body = "\n".join(f"{OL[o]} & {trend('primary: ever-victims, all rounds', o)} & {trend('S2: DHS phase >= 6', o)} & {trend('S1: recent victims', o)} \\\\" for o in ["sought_any", "told", "formal"])
    out.append(table("tab:trends", "\\textbf{Change per decade in help-seeking: all rounds and DHS phase 6 or later}", ">{\\raggedright\\arraybackslash}p{3.4cm}ccc",
                     "Outcome & All rounds & Phase 6 or later & Violence in past 12 months", body,
                     "Odds ratio per decade (95\\% CrI) [number of countries], adjusted for the main-model predictors, pooled with the registered Gibbs implementation of equation~\\ref{eq:hier}. "
                     "Restricting to phase 6 or later removes the largest questionnaire changes; restricting to recent violence aligns timing."))
    F11 = [p38("B11_formal_primary", m, t, "formal") for m, t in F4]; F11w = [p38("B11_formal_wide", m, t, "formal_wide") for m, t in F4]
    fx = lambda x: (ci(x.OR.iloc[0], x.OR_lo.iloc[0], x.OR_hi.iloc[0]) + f" [{int(x.countries.iloc[0])}]") if len(x) else "--"
    body = f"Primary definition & " + " & ".join(fx(x) for x in F11) + " \\\\\nWide definition & " + " & ".join(fx(x) for x in F11w) + " \\\\"
    out.append(table("tab:formal", "\\textbf{Formal help-seeking: primary and wide definitions}", ">{\\raggedright\\arraybackslash}p{3.4cm}cccc",
                     "Definition & Injury & Severe violence & Both types & Formerly in union", body,
                     "Odds ratios (95\\% CrI) [countries]. Formal help could be modelled only in countries with at least 30 events, so intervals are wide and regions are not compared."))
    VIF = pd.read_csv(P38 / "P38_B7_vif.csv")
    out.append(f"\\paragraph{{Multicollinearity.}} Generalised variance inflation factors\\cite{{fox1992}} were computed on the pooled data. The largest GVIF$^{{1/(2\\,df)}}$ was {f2(VIF.GVIF_1_2df.max())} "
               f"({VIF.loc[VIF.GVIF_1_2df.idxmax(), 'term']}, {VIF.loc[VIF.GVIF_1_2df.idxmax(), 'model']} model), well below the conventional threshold of 2, so collinearity does not affect the estimates.\n\n")
    out.append(table("tab:harm", "\\textbf{Harmonised pooled analysis of DHS and VACS: young women and girls aged 15--24 years who reported violence}", ">{\\raggedright\\arraybackslash}p{4.3cm}ccccc",
                     f" & DHS & VACS & Ratio VACS/DHS & Pooled & Surveys \\\\\n & ({n37d} countries) & ({n37v} countries) & & & ", harm_rows,
                     "Odds ratios (95\\% credible intervals) from equation~\\ref{eq:harm}: country models for each survey programme (survey-weighted logistic regression with the four factors entered together; DHS with survey-round fixed effects), combined in a Bayesian hierarchical model with a programme term. "
                     "Pooled odds ratios give the two programmes equal weight and are interpreted only where the surveys agree. Agree: CrI of the ratio includes one and posterior probability of the same sign above 0.9. Differ: CrI of the ratio excludes one. VACS rural residence excludes Kenya (not recorded). Pre-registration 37."))

    # ---------------- D. Pre-registration ----------------
    log = [l.split("  ", 2) for l in (R / "docs/preregistration/PREREGISTRATION_hashes.log").read_text().splitlines() if l.strip()]
    keep = [x for x in log if len(x) == 3 and re.search(r"PREREGISTRATION_3[3-8]", x[2])]
    body = "\n".join(f"{x[0][:16]} & \\texttt{{{x[1][:16]}\\ldots}} & {texesc(x[2])} \\\\" for x in keep)
    out.append(table("tab:prereg", "\\textbf{Pre-registration record}", "l>{\\ttfamily}l>{\\raggedright\\arraybackslash}p{8.2cm}", "Time stamp & SHA-256 (first 16 characters) & Document and stage", body,
                     "Each analysis plan, amendment and result was hashed when written; a plan was always hashed before the analysis it describes was run. Full documents and hashes are in the code repository.", long=True))
    return "\n".join(out)
