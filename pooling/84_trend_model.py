"""84_trend_model.py - Pre-registration 39: Bayesian hierarchical meta-regression of survey-level cascade estimates over time.
    y_cs ~ N(a_c + b_c t_cs, se_cs^2);  a_c = A_r + sigma_a za_c;  b_c = B_r + sigma_b zb_c;  A_r = A + omega_a ua_r;  B_r = B + omega_b ub_r
Outputs (outputs/tables/P38_review/): P39_curves.csv, P39_slopes.csv
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
import numpy as np, pandas as pd, jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary
from scipy.special import expit, logit
numpyro.set_host_device_count(4)
T = "outputs/tables/P38_review/"; REG = ["West", "Central", "East", "Southern"]
E = pd.read_csv(T + "P39_survey_estimates.csv")

def model(y, se, t, ci, cr, n_c, n_r, a0):
    A = numpyro.sample("A", dist.Normal(a0, 1.5)); B = numpyro.sample("B", dist.Normal(0, 1))
    om_a = numpyro.sample("omega_a", dist.HalfNormal(1.0)); om_b = numpyro.sample("omega_b", dist.HalfNormal(0.5))
    sg_a = numpyro.sample("sigma_a", dist.HalfNormal(1.0)); sg_b = numpyro.sample("sigma_b", dist.HalfNormal(0.5))
    with numpyro.plate("regions", n_r):
        A_r = numpyro.deterministic("A_r", A + om_a * numpyro.sample("ua", dist.Normal(0, 1)))
        B_r = numpyro.deterministic("B_r", B + om_b * numpyro.sample("ub", dist.Normal(0, 1)))
    with numpyro.plate("countries", n_c):
        a_c = A_r[cr] + sg_a * numpyro.sample("za", dist.Normal(0, 1))
        b_c = B_r[cr] + sg_b * numpyro.sample("zb", dist.Normal(0, 1))
    sg_e = numpyro.sample("sigma_e", dist.HalfNormal(0.5))          # Deviation 1: survey-level residual heterogeneity
    numpyro.sample("y", dist.Normal(a_c[ci] + b_c[ci] * t, jnp.sqrt(se**2 + sg_e**2)), obs=y)

curves, slopes = [], []
for o in ["told", "sought_any", "formal"]:
    d = E[E.outcome == o].reset_index(drop=True)
    cs = sorted(d.country.unique()); ci = np.array([cs.index(c) for c in d.country])
    creg = d.groupby("country").region.first().reindex(cs); cr = np.array([REG.index(r) for r in creg])
    a0 = float(logit(0.03 if o == "formal" else 0.3))
    mc = MCMC(NUTS(model, target_accept_prob=0.995, max_tree_depth=12), num_warmup=2000, num_samples=2000, num_chains=4, chain_method="parallel", progress_bar=False)
    mc.run(jax.random.PRNGKey(20261023), y=jnp.array(d.logit.values), se=jnp.array(d.se.values), t=jnp.array(d.t.values), ci=jnp.array(ci), cr=jnp.array(cr),
           n_c=len(cs), n_r=len(REG), a0=a0)
    s = {k: np.asarray(v) for k, v in mc.get_samples().items()}; sg = mc.get_samples(group_by_chain=True)
    dg = summary({k: sg[k] for k in ["A", "B", "omega_a", "omega_b", "sigma_a", "sigma_b", "sigma_e", "A_r", "B_r"]}, group_by_chain=True)
    rhat = max(float(np.max(dg[k]["r_hat"])) for k in dg); ess = min(float(np.min(dg[k]["n_eff"])) for k in dg)
    div = int(mc.get_extra_fields()["diverging"].sum())
    q = lambda v: np.quantile(v, [0.5, 0.025, 0.975])
    # curves: overall over 2004-2025; regions over their observed years
    grid_all = np.linspace((2004 - 2015) / 10, (2025.6 - 2015) / 10, 60)
    for g in grid_all:
        m, lo, hi = q(expit(s["A"] + s["B"] * g)); curves.append(dict(outcome=o, scope="All", year=2015 + 10 * g, pct=100 * m, lo=100 * lo, hi=100 * hi))
    for j, r in enumerate(REG):
        dr = d[d.region == r]
        if dr.empty: continue
        gr = np.linspace(dr.t.min(), dr.t.max(), 40) if dr.t.max() > dr.t.min() else np.array([dr.t.min()])
        estimable = r != "Southern"
        for g in gr:
            m, lo, hi = q(expit(s["A_r"][:, j] + s["B_r"][:, j] * g))
            curves.append(dict(outcome=o, scope=r, year=2015 + 10 * g, pct=100 * m, lo=100 * lo, hi=100 * hi, trend_estimable=estimable))
    # slopes and tests
    ch = lambda A, B: 100 * (expit(A + B * 1.0) - expit(A + B * -1.0))             # change 2005 -> 2025, percentage points
    row = dict(outcome=o, scope="All", surveys=len(d), countries=len(cs), multi_round=int((d.country.value_counts() >= 2).sum()))
    m, lo, hi = q(np.exp(s["B"])); row.update(OR_decade=m, OR_lo=lo, OR_hi=hi, P_decline=float((s["B"] < 0).mean()))
    m, lo, hi = q(ch(s["A"], s["B"])); row.update(pp_2005_2025=m, pp_lo=lo, pp_hi=hi)
    reg_ok = [REG.index(r) for r in ["West", "Central", "East"]]
    row.update(P_all_regions_same_sign=float((np.sign(s["B_r"][:, reg_ok]) == np.sign(np.median(s["B"]))[None]).all(1).mean()),
               omega_b=float(np.median(s["omega_b"])), sigma_b=float(np.median(s["sigma_b"])), sigma_e=float(np.median(s["sigma_e"])), rhat_max=rhat, ess_min=ess, divergences=div)
    slopes.append(row)
    for j, r in enumerate(REG):
        dr = d[d.region == r]
        m, lo, hi = q(np.exp(s["B_r"][:, j]))
        slopes.append(dict(outcome=o, scope=r, surveys=len(dr), countries=dr.country.nunique(), multi_round=int((dr.country.value_counts() >= 2).sum()),
                           OR_decade=m, OR_lo=lo, OR_hi=hi, P_decline=float((s["B_r"][:, j] < 0).mean()), trend_estimable=r != "Southern"))
    a = slopes[-5]; print(f"{o:10s} overall OR/decade {a['OR_decade']:.2f} ({a['OR_lo']:.2f}-{a['OR_hi']:.2f}) P(decline) {a['P_decline']:.3f} | change 2005-25 {a['pp_2005_2025']:.1f} pp ({a['pp_lo']:.1f} to {a['pp_hi']:.1f}) | Rhat {rhat:.3f} ESS {ess:.0f} div {div}")
    for x in slopes[-4:]: print(f"    {x['scope']:9s} OR/decade {x['OR_decade']:.2f} ({x['OR_lo']:.2f}-{x['OR_hi']:.2f}) P(decline) {x['P_decline']:.2f} multi-round {x['multi_round']}")
pd.DataFrame(curves).to_csv(T + "P39_curves.csv", index=False); pd.DataFrame(slopes).to_csv(T + "P39_slopes.csv", index=False)
