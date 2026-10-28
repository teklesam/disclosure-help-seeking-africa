"""62_figures_v3.py - Figures 2-5 and supplementary weights for manuscript v3 (youth; continuous age; NumPyro pooling; survey-design CIs).
No in-figure titles; captions carry them. Okabe-Ito colours: West blue, Central green, East orange, Southern pink.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
R = Path.home() / "Documents/OMS DHS Births Climate Policy"; T = R / "outputs/tables"; V3 = T / "P33_youth_v3"; F = R / "outputs/figures/disclosure_v3"; F.mkdir(exist_ok=True)
OK = {"West": "#0072B2", "Central": "#009E73", "East": "#E69F00", "Southern": "#CC79A7", "girls": "#D55E00", "boys": "#56B4E9"}
REGS = ["West", "Central", "East", "Southern"]
NAMES = {"AO": "Angola", "BF": "Burkina Faso", "BJ": "Benin", "BU": "Burundi", "CD": "DR Congo", "CI": "Côte d'Ivoire", "CM": "Cameroon", "ET": "Ethiopia", "GA": "Gabon", "GH": "Ghana",
         "GM": "Gambia", "GN": "Guinea", "KE": "Kenya", "KM": "Comoros", "LB": "Liberia", "LS": "Lesotho", "MD": "Madagascar", "ML": "Mali", "MR": "Mauritania", "MW": "Malawi", "MZ": "Mozambique",
         "NG": "Nigeria", "NM": "Namibia", "RW": "Rwanda", "SL": "Sierra Leone", "SN": "Senegal", "TD": "Chad", "TG": "Togo", "TZ": "Tanzania", "UG": "Uganda", "ZA": "South Africa", "ZM": "Zambia", "ZW": "Zimbabwe", "SZ": "Eswatini"}

# ---------- Figure 2: cascade with 95% CIs ----------
D = pd.read_csv(V3 / "A8_5_dhs_cascade_ci.csv"); Vc = pd.read_csv(V3 / "A8_5_vacs_cascade_ci.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[1.1, 1]))
outs = [("told", "Told anyone"), ("sought_any", "Sought any help"), ("formal", "Sought formal help")]
for j, (o, lab) in enumerate(outs):
    for k, r in enumerate(REGS + ["All countries"]):
        x = D[(D.outcome == o) & (((D.level == "region") & (D.region == r)) | ((D.level == "all") & (r == "All countries")))].iloc[0]
        xx = j + (k - 2) * 0.15; col = OK.get(r, "black")
        a.errorbar(xx, x.pct, yerr=[[x.pct - x.lo], [x.hi - x.pct]], fmt="D" if r == "All countries" else "o", color=col, ms=5, capsize=2, lw=1.2, label=(r if j == 0 else None))
a.set_xticks(range(3)); a.set_xticklabels([l for _, l in outs]); a.set_ylim(0, 65)
a.set_ylabel("Young women aged 15–24 who experienced violence (%)\nsurvey-weighted, 95% CI", fontsize=9); a.spines[["top", "right"]].set_visible(False)
a.legend(frameon=False, fontsize=8, loc="upper right", title="DHS, by UN region", title_fontsize=8); a.text(-0.08, 1.02, "a", transform=a.transAxes, fontsize=13, fontweight="bold")
steps = [("tell", "Told\nanyone"), ("knewplace", "Knew where\nto get help"), ("sought", "Sought\nhelp"), ("recvdserv", "Received\nservices")]
Va = Vc[Vc.level == "all VACS countries"]
for t, mk, off in [("SV", "o", -0.22), ("PV", "s", 0.08)]:
    for s, so in [("girls", 0.0), ("boys", 0.12)]:
        for i, (st, _) in enumerate(steps):
            x = Va[(Va.type == t) & (Va.step == st) & (Va.sex == s)].iloc[0]
            b.errorbar(i + off + so, x.pct, yerr=[[x.pct - x.lo], [x.hi - x.pct]], fmt=mk, color=OK[s], ms=5, capsize=2, lw=1.2,
                       mfc=OK[s] if t == "SV" else "white", label=(f"{'Sexual' if t == 'SV' else 'Physical'} violence, {s}" if i == 0 else None))
b.set_xticks(range(4)); b.set_xticklabels([l for _, l in steps], fontsize=8.5); b.set_ylim(0, 65)
b.set_ylabel("Young people aged 13–24 who experienced violence (%)\nsurvey-weighted, 95% CI", fontsize=9); b.spines[["top", "right"]].set_visible(False)
b.legend(frameon=False, fontsize=7.8, loc="upper right", title="VACS, 8 countries", title_fontsize=8); b.text(-0.1, 1.02, "b", transform=b.transAxes, fontsize=13, fontweight="bold")
fig.text(0.01, -0.04, "95% confidence intervals account for survey weights and clustering (logit scale). (a) DHS 2004–2025, women 15–24 (n = 49 313; told anyone in surveys where asked). "
         "(b) VACS 2013–2023, filled = sexual, open = physical violence;\ncountries weighted by sample size.", fontsize=7.5)
fig.tight_layout(); fig.savefig(F / "fig2_cascade_ci.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Figure 3: predictors with regional credible intervals ----------
P = pd.read_csv(V3 / "A3_pooled_numpyro.csv"); P = P[P.source == "DHS"]
KEYS = [("partner", "injury", "Injury from partner violence (vs none)"), ("partner", "severe", "Severe physical violence (vs less severe/none)"),
        ("main", "vtypeboth", "Both physical and sexual (vs physical only)"), ("main", "maritalformer", "Formerly in union (vs currently in union)"),
        ("partner", "alcohol", "Partner drinks alcohol (vs not)"), ("partner", "controlling3", "≥3 controlling behaviours (vs 0–2)"),
        ("main", "perpnonpartner", "Non-partner perpetrator only (vs partner only)"), ("main", "working", "Working (vs not working)"),
        ("main", "educsecondary+", "Secondary+ education (vs none)"), ("main", "wealthq", "Wealth (per quintile, 1–5)"), ("main", "urban", "Urban (vs rural)"),
        ("main", "media", "Weekly media exposure (vs less)"), ("main", "decisions2", "Joint/own decisions (vs none; women in union)"),
        ("main", "justify", "Justifies wife-beating (vs never)"), ("main", "agegrp", "Age (per year, 15–24)")]
fig, axes = plt.subplots(1, 2, figsize=(12, 7.6), sharey=True)
for ax, (o, lab, tag) in zip(axes, [("sought_any", "Sought any help", "a"), ("told", "Told anyone", "b")]):
    for i, (mdl, term, nm) in enumerate(KEYS):
        r = P[(P.model == mdl) & (P.outcome == o) & (P.term == term)]
        if r.empty: continue
        r = r.iloc[0]; yy = len(KEYS) - i
        for k, reg in enumerate(REGS):
            if pd.notna(r.get(f"OR_{reg}")):
                y2 = yy + 0.3 - k * 0.12
                ax.plot([r[f"OR_{reg}_lo"], r[f"OR_{reg}_hi"]], [y2, y2], color=OK[reg], lw=0.9, alpha=0.8); ax.scatter(r[f"OR_{reg}"], y2, color=OK[reg], s=8, zorder=3)
        ax.plot([r.OR_lo, r.OR_hi], [yy - 0.22, yy - 0.22], color="black", lw=2); ax.scatter(r.OR, yy - 0.22, color="black", s=34, marker="D", zorder=4)
    ax.axvline(1, color="grey", ls="--", lw=0.8); ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"]); ax.set_xlim(0.35, 5)
    ax.set_xlabel(f"{lab}: odds ratio (log scale)\nblack = pooled, 95% CrI; colours = regional means, 95% CrI", fontsize=8.5); ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.02, 1.01, tag, transform=ax.transAxes, fontsize=13, fontweight="bold")
axes[0].set_yticks(range(len(KEYS), 0, -1)); axes[0].set_yticklabels([n for *_, n in KEYS], fontsize=8.3)
for reg in [r for r in REGS if f"OR_{r}" in P.columns and P[f"OR_{r}"].notna().any()]: axes[1].plot([], [], color=OK[reg], lw=2, label=f"{reg} Africa")
axes[1].plot([], [], color="black", lw=2, marker="D", label="All regions (pooled)")
fig.legend(*axes[1].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper center", ncol=5, bbox_to_anchor=(0.6, 1.02))
fig.text(0.01, -0.02, "DHS women 15–24 who experienced violence. Mutually adjusted country logistic models, pooled in a Bayesian hierarchical model (NumPyro, 4 chains). "
         "Partner items (injury, severity, alcohol, control)\namong women with partner violence. Dashed line = no association. Southern African countries (Lesotho, Namibia, South Africa) had too few young women for country models (<300 women or <30 events).", fontsize=7.3)
fig.tight_layout(rect=(0, 0.02, 1, 0.97)); fig.savefig(F / "fig3_predictors_regional_cri.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Figure 4: trends in any help-seeking (NumPyro pooled) ----------
rr = pd.read_csv(V3 / "A5_rates_by_round.csv"); ct = pd.read_csv(V3 / "A5_country_trends.csv")
ct = ct[(ct.term == "yr10") & (ct.model == "adjusted") & (ct.outcome == "sought_any")].copy()
pt = pd.read_csv(V3 / "A5_trends_numpyro.csv"); pt = pt[(pt.model == "adjusted") & (pt.outcome == "sought_any")].iloc[0]
vt = pd.read_csv(V3 / "A8_2_trends_numpyro.csv")
ct["OR"] = np.exp(ct.est); ct["lo"] = np.exp(ct.est - 1.96 * ct.se); ct["hi"] = np.exp(ct.est + 1.96 * ct.se)
ct["ro"] = ct.region.map({r: i for i, r in enumerate(REGS)}); ct = ct.sort_values(["ro", "OR"]).reset_index(drop=True)
fig = plt.figure(figsize=(13, 9)); outer = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)
NR = int(np.ceil(len(ct) / 4)); inner = outer[0].subgridspec(NR, 4, hspace=0.6, wspace=0.12); ymax = 100 * rr.sought_any.max() * 1.1
for i, c in enumerate(ct.country):
    ax = fig.add_subplot(inner[i // 4, i % 4]); g = rr[rr.country == c].sort_values("syear"); reg = g.region.iloc[0]
    ax.plot(g.syear, 100 * g.sought_any, "-o", color=OK[reg], ms=3.5, lw=1.4); ax.set_ylim(0, ymax); ax.set_xlim(2003, 2027); ax.set_xticks([2005, 2015, 2025]); ax.tick_params(labelsize=7)
    ax.set_title(f"{NAMES[c]} ({reg[0]})", fontsize=8, pad=2); ax.spines[["top", "right"]].set_visible(False)
    if i % 4: ax.set_yticklabels([])
    if i + 4 < len(ct): ax.set_xticklabels([])
    if i == 0: ax.text(-0.35, 1.12, "a", transform=ax.transAxes, fontsize=13, fontweight="bold")
fig.text(0.075, 0.5, "Young women 15–24 who sought any help after violence (%), survey-weighted, by DHS round", rotation=90, va="center", fontsize=8.5)
ax = fig.add_subplot(outer[1]); yy = np.arange(len(ct))[::-1] + 4
for y0, r in zip(yy, ct.itertuples()):
    ax.plot([r.lo, r.hi], [y0, y0], color=OK[r.region], lw=1.3); ax.scatter(r.OR, y0, color=OK[r.region], s=20, zorder=3)
rows = [(2.2, pt, "Help-seeking, pooled (95% CrI)", "black"),
        (1.2, vt[vt.model.str.startswith("A8.2b sought, partner victims, + sev")].iloc[0], "…adjusted for severity & injury", "#555555"),
        (0.2, vt[vt.model.str.startswith("A8.2a")].iloc[0], "Victimisation (all women 15–24)", "#CC79A7")]
for y0, r, lab, col in rows:
    ax.plot([r.OR_lo, r.OR_hi], [y0, y0], color=col, lw=2); ax.scatter(r.OR, y0, marker="D", color=col, s=30, zorder=3)
ax.set_yticks([r[0] for r in rows] + list(yy)); ax.set_yticklabels([r[2] for r in rows] + [f"{NAMES[c]} ({g[0]})" for c, g in zip(ct.country, ct.region)], fontsize=7.5)
ax.axvline(1, color="grey", ls="--", lw=0.8); ax.set_xscale("log"); ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"]); ax.set_xlim(0.2, 6)
ax.set_xlabel("Odds ratio per decade (adjusted; log scale)\ncountries: 95% CI; diamonds: pooled 95% CrI", fontsize=8.5); ax.spines[["top", "right"]].set_visible(False)
ax.text(-0.02, 1.01, "b", transform=ax.transAxes, fontsize=13, fontweight="bold")
fig.text(0.08, 0.02, f"DHS 2004–2025; {len(ct)} countries with ≥2 rounds (no Southern African country has ≥2 rounds). Blue = West, green = Central, orange = East Africa. Country logistic models adjusted for Table 2 covariates;\n"
         "rounds without recency coding drop out (e.g. Mali 2006). Pink diamond: trend in past-12-month victimisation, to check whether falling help-seeking reflects changing victimisation.", fontsize=7.3)
fig.savefig(F / "fig4_trends_any_help.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Figure 5: climate, with placebo ----------
c1 = pd.read_csv(T / "P27_three_set_youth.csv"); c2 = pd.read_csv(T / "P28_spei12_violence_speibase_youth.csv"); c3 = pd.read_csv(T / "P31_known_disasters_youth.csv")
ck = pd.read_csv(T / "A8_1_climate_checks_youth.csv")
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
for ax, (o1, o2, o3, lab, eq, tag) in zip(axes, [("formal_k", "formal help-seeking", "formal_k", "Sought formal help", 5, "a"), ("sought_k", "sought any help", "sought_any_k", "Sought any help", 20, "b")]):
    rows = []
    for st, nm in [("discovery", "discovery countries"), ("held-out", "held-out countries"), ("fresh", "fresh countries"), ("pooled (DL, 3 sets)", "pooled (3 sets)")]:
        r = c1[(c1.outcome == o1) & (c1.set == st)].iloc[0]; rows.append((f"Regional hazard, {nm}", r.est, r.lo, r.hi, "#D55E00"))
    if o1 == "sought_k":
        r = ck[(ck.check.str.startswith("(a)")) & (ck.set == "pooled (3 sets)")].iloc[0]; rows.append(("PLACEBO: hazard in the 12 months AFTER interview", r.est, r.lo, r.hi, "#CC79A7"))
    r = c2[(c2.design == "D3") & (c2["sample"] == f"all | {o2}, recent victims")].iloc[0]; rows.append(("SPEIbase drought (same place over time)", r.est, r.lo, r.hi, "#D55E00"))
    for sc, nm in [("POOLED (7 events, stacked)", "Seven documented disasters (pooled)"), ("pooled flood", "   floods (5 events)"), ("pooled drought", "   droughts (2 events)")]:
        r = c3[(c3.outcome == o3) & (c3.scope == sc)].iloc[0]; rows.append((nm, r.est, r.lo, r.hi, "#D55E00"))
    if o1 == "formal_k": rows.insert(4, ("(placebo shown for any help only)", np.nan, np.nan, np.nan, "white"))
    yy = np.arange(len(rows))[::-1]; ax.axvspan(-eq, eq, color="#cfe8cf", alpha=0.6, label=f"Equivalence band (±{eq} per 1,000)")
    for y0, (nm, e, lo, hi, col) in zip(yy, rows):
        if np.isfinite(e): ax.plot([lo, hi], [y0, y0], color="#333333", lw=1.4); ax.scatter(e, y0, color=col, s=34, zorder=3)
    ax.axvline(0, color="grey", ls="--", lw=0.8); ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=8.3)
    ax.set_xlabel(f"{lab}: change per 1,000 recent victims, 95% CI\nper SD of hazard / per SPEI unit / before–after disaster", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2)); ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.02, 1.02, tag, transform=ax.transAxes, fontsize=13, fontweight="bold")
fig.text(0.01, -0.07, "Women 15–24 with violence in the past 12 months (DHS). Regional hazard = the region's main hazard (hot days, heavy-rain days or drought months) in the 12 months before interview, "
         "per SD within region.\nThe placebo (pink) uses hazard AFTER the interview, which cannot cause earlier help-seeking: its similar size shows the any-help association is not a climate effect.", fontsize=7.4)
fig.tight_layout(); fig.savefig(F / "fig5_climate_with_placebo.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Supplementary: country weights in pooled estimates (NumPyro tau, omega) ----------
CE = pd.read_csv(V3 / "A2_country_estimates.csv")
SK = [("partner", "injury", "Injury from partner violence"), ("partner", "severe", "Severe physical violence"), ("main", "maritalformer", "Formerly in union")]
fig, axes = plt.subplots(1, 3, figsize=(15, 9.5))
for ax, (mdl, term, lab), tag in zip(axes, SK, "abc"):
    x = CE[(CE.source == "DHS") & (CE.model == mdl) & (CE.outcome == "sought_any") & (CE.term == term)].copy()
    p = P[(P.model == mdl) & (P.outcome == "sought_any") & (P.term == term)].iloc[0]
    x["wc"] = 1 / (x.se**2 + p.tau**2); Pr = x.groupby("region").wc.sum(); Wr = 1 / (1 / Pr + p.omega**2)
    x["weight"] = 100 * x.region.map(Wr / Wr.sum()) * x.wc / x.region.map(Pr)
    y = 0; ticks = []; labs = []
    for reg in REGS[::-1]:
        if reg not in Pr.index: continue
        y += 1; ax.plot([p[f"OR_{reg}_lo"], p[f"OR_{reg}_hi"]], [y, y], color=OK[reg], lw=3); ax.scatter(p[f"OR_{reg}"], y, marker="D", color=OK[reg], s=45, zorder=3)
        ticks.append(y); labs.append(f"{reg} Africa ({100 * Wr[reg] / Wr.sum():.0f}%)")
        for r in x[x.region == reg].sort_values("est", ascending=False).itertuples():
            y += 1; lo, hi = np.exp(r.est - 1.96 * r.se), np.exp(r.est + 1.96 * r.se)
            ax.plot([max(lo, 0.1), min(hi, 30)], [y, y], color=OK[reg], lw=0.8); ax.scatter(np.exp(r.est), y, marker="s", color=OK[reg], s=8 + 9 * r.weight, zorder=3)
            ticks.append(y); labs.append(f"{NAMES[r.country]} ({r.weight:.1f}%)")
        y += 0.6
    ax.plot([p.OR_lo, p.OR_hi], [-0.6, -0.6], color="black", lw=3); ax.scatter(p.OR, -0.6, marker="D", color="black", s=50, zorder=3)
    ticks.append(-0.6); labs.append(f"Pooled: {p.OR:.2f} ({p.OR_lo:.2f}–{p.OR_hi:.2f})")
    ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=7); ax.axvline(1, color="grey", ls="--", lw=0.8); ax.set_xscale("log"); ax.set_xlim(0.1, 30)
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 16]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8", "16"])
    ax.set_xlabel(f"{lab}: odds ratio for seeking any help\n(log scale; countries 95% CI, diamonds 95% CrI)", fontsize=8.5); ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.02, 1.01, tag, transform=ax.transAxes, fontsize=13, fontweight="bold")
fig.text(0.01, 0.0, "DHS women 15–24. Square size and % = approximate weight of each country in the pooled estimate (from the posterior median between-country SD τ and between-region SD ω, NumPyro); "
         "diamonds = regional and pooled means.", fontsize=7.3)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(F / "figS1_country_weights.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("v3 figures written to", F)
