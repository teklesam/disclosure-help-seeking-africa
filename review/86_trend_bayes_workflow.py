"""86_trend_bayes_workflow.py - Pre-registration 39, addendum B: Bayesian workflow for the trend meta-regression.
Models for survey-level logits y_cs (design-based SE se_cs, time t in decades from 2015), with survey residual sigma_e:
  T0 no trend          eta = a_c
  T1 common trend      eta = a_c + B t
  T2 country trends    eta = a_c + b_c t,  b_c = B + sigma_b z_c
  T3 primary           eta = a_c + b_c t,  b_c = B_r(c) + sigma_b z_c,  B_r = B + omega_b u_r
  a_c = A_r(c) + sigma_a za_c,  A_r = A + omega_a ua_r  (all models)
Outputs (outputs/tables/P38_review/): P39B_prior_predictive.csv, P39B_model_checks.csv, P39B_prior_sensitivity.csv; figure figS_trend_checks.png
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
import numpy as np, pandas as pd, jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist, arviz as az
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import summary
from scipy.special import expit, logit
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
numpyro.set_host_device_count(4)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9})
T = "outputs/tables/P38_review/"; F = "outputs/figures/disclosure_v3/"; REG = ["West", "Central", "East", "Southern"]
E = pd.read_csv(T + "P39_survey_estimates.csv")
OUT = {"told": "Told anyone", "sought_any": "Sought any help", "formal": "Sought formal help"}

def model(se, t, ci, cr, n_c, n_r, a0, kind="T3", b_sd=1.0, slope_sd=0.5, y=None):
    A = numpyro.sample("A", dist.Normal(a0, 1.5))
    om_a = numpyro.sample("omega_a", dist.HalfNormal(1.0)); sg_a = numpyro.sample("sigma_a", dist.HalfNormal(1.0))
    sg_e = numpyro.sample("sigma_e", dist.HalfNormal(0.5))
    with numpyro.plate("regions", n_r): A_r = A + om_a * numpyro.sample("ua", dist.Normal(0, 1))
    with numpyro.plate("countries", n_c): a_c = A_r[cr] + sg_a * numpyro.sample("za", dist.Normal(0, 1))
    if kind == "T0":
        eta = a_c[ci]
    else:
        B = numpyro.sample("B", dist.Normal(0, b_sd))
        if kind == "T1":
            b_c = B * jnp.ones(n_c)
        else:
            sg_b = numpyro.sample("sigma_b", dist.HalfNormal(slope_sd))
            if kind == "T3":
                om_b = numpyro.sample("omega_b", dist.HalfNormal(slope_sd))
                with numpyro.plate("regions_b", n_r): B_r = B + om_b * numpyro.sample("ub", dist.Normal(0, 1))
                mean_b = B_r[cr]
            else:
                mean_b = B * jnp.ones(n_c)
            with numpyro.plate("countries_b", n_c): b_c = mean_b + sg_b * numpyro.sample("zb", dist.Normal(0, 1))
        eta = a_c[ci] + b_c[ci] * t
    numpyro.deterministic("eta", eta)
    numpyro.sample("y", dist.Normal(eta, jnp.sqrt(se**2 + sg_e**2)), obs=y)

def design(d):
    cs = sorted(d.country.unique()); ci = np.array([cs.index(c) for c in d.country])
    cr = np.array([REG.index(r) for r in d.groupby("country").region.first().reindex(cs)])
    return dict(se=jnp.array(d.se.values), t=jnp.array(d.t.values), ci=jnp.array(ci), cr=jnp.array(cr), n_c=len(cs), n_r=len(REG))

from numpyro.diagnostics import effective_sample_size
def tail_ess(x):   # Vehtari et al. 2021: min ESS of the 5% and 95% quantile indicators; x = (chain, draw[, dims])
    x = x.reshape(x.shape[0], x.shape[1], -1); out = []
    for j in range(x.shape[2]):
        v = x[:, :, j]; lo, hi = np.quantile(v, [0.05, 0.95])
        out += [float(effective_sample_size((v <= lo).astype(float))), float(effective_sample_size((v >= hi).astype(float)))]
    return min(out)

def ols_slope(y, t): tc = t - t.mean(); return (y * tc[None, :]).sum(1) / (tc ** 2).sum()

def fit(d, o, kind="T3", b_sd=1.0, slope_sd=0.5):
    a0 = float(logit(0.03 if o == "formal" else 0.3)); ds = design(d); y = d.logit.values; se = d.se.values; t = d.t.values
    mc = MCMC(NUTS(model, target_accept_prob=0.995, max_tree_depth=12), num_warmup=2000, num_samples=2000, num_chains=4, chain_method="parallel", progress_bar=False)
    mc.run(jax.random.PRNGKey(20261023), **ds, a0=a0, kind=kind, b_sd=b_sd, slope_sd=slope_sd, y=jnp.array(y), extra_fields=("diverging", "energy"))
    sg = mc.get_samples(group_by_chain=True); s = {k: np.asarray(v).reshape(8000, *np.asarray(v).shape[2:]) for k, v in sg.items()}
    keys = [k for k in ["A", "B", "omega_a", "sigma_a", "sigma_e", "sigma_b", "omega_b"] if k in sg]
    sm = summary({k: sg[k] for k in keys}, group_by_chain=True)
    ess_tail = min(tail_ess(np.asarray(sg[k])) for k in keys)
    energy = np.asarray(mc.get_extra_fields(group_by_chain=True)["energy"])
    bfmi = float(np.min([np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2) for e in energy]))
    sd = np.sqrt(se[None, :] ** 2 + s["sigma_e"][:, None] ** 2); eta = s["eta"]
    ll = norm.logpdf(y[None, :], eta, sd)
    Dbar = float(np.mean(-2 * ll.sum(1))); Dhat = float(-2 * norm.logpdf(y, eta.mean(0), np.sqrt(se ** 2 + s["sigma_e"].mean() ** 2)).sum()); pD = Dbar - Dhat
    lppd = float(np.sum(np.log(np.mean(np.exp(ll), 0)))); pw = float(np.sum(np.var(ll, 0)))
    lo = az.loo(az.from_dict({"posterior": {"A": np.asarray(sg["A"])}, "log_likelihood": {"y": ll.reshape(4, 2000, -1)}}), var_name="y", pointwise=True)
    rng = np.random.default_rng(20261023); yrep = eta + rng.normal(size=eta.shape) * sd
    disc = {"chi2": (((yrep - eta) / sd) ** 2).sum(1), "sd": yrep.std(1), "max": yrep.max(1), "min": yrep.min(1), "slope": ols_slope(yrep, t)}
    obs = {"chi2": (((y[None, :] - eta) / sd) ** 2).sum(1), "sd": np.full(8000, y.std()), "max": np.full(8000, y.max()), "min": np.full(8000, y.min()), "slope": np.full(8000, ols_slope(y[None, :], t)[0])}
    r = dict(outcome=o, model=kind, b_sd=b_sd, slope_sd=slope_sd, surveys=len(d),
             rhat_max=max(float(np.max(sm[k]["r_hat"])) for k in keys), ess_bulk_min=min(float(np.min(sm[k]["n_eff"])) for k in keys), ess_tail_min=ess_tail,
             divergences=int(np.asarray(mc.get_extra_fields()["diverging"]).sum()), ebfmi_min=bfmi,
             DIC=Dbar + pD, pD=pD, WAIC=-2 * (lppd - pw), elpd_loo=float(lo.elpd), se_loo=float(lo.se), pareto_k_max=float(np.max(np.asarray(lo.pareto_k))))
    for k in disc: r[f"pB_{k}"] = float(np.mean(disc[k] >= obs[k]))
    if "B" in s:
        q = np.exp(np.quantile(s["B"], [.5, .025, .975])); r.update(OR_decade=q[0], OR_lo=q[1], OR_hi=q[2], P_decline=float((s["B"] < 0).mean()))
    return r, np.asarray(lo.elpd_i), disc, obs

rows, sens, prior_rows, store = [], [], [], {}
for o in OUT:
    d = E[E.outcome == o].reset_index(drop=True); ds = design(d); a0 = float(logit(0.03 if o == "formal" else 0.3))
    # 1. prior predictive
    pp = Predictive(model, num_samples=8000)(jax.random.PRNGKey(7), **ds, a0=a0, kind="T3", y=None)
    ysim = np.asarray(pp["y"]); Bsim = np.asarray(pp["B"])
    prior_rows.append(dict(outcome=o, prev_lo=100 * expit(np.quantile(ysim, .025)), prev_hi=100 * expit(np.quantile(ysim, .975)),
                           share_prev_below_1_or_above_99=float(np.mean((expit(ysim) < 0.01) | (expit(ysim) > 0.99))),
                           OR_decade_lo=float(np.exp(np.quantile(Bsim, .025))), OR_decade_hi=float(np.exp(np.quantile(Bsim, .975))),
                           sd_logit_lo=float(np.quantile(ysim.std(1), .025)), sd_logit_hi=float(np.quantile(ysim.std(1), .975)), sd_logit_obs=float(d.logit.std(ddof=0)),
                           slope_lo=float(np.quantile(ols_slope(ysim, d.t.values), .025)), slope_hi=float(np.quantile(ols_slope(ysim, d.t.values), .975)),
                           slope_obs=float(ols_slope(d.logit.values[None, :], d.t.values)[0])))
    store[(o, "prior_sd")] = ysim.std(1); store[(o, "prior_slope")] = ols_slope(ysim, d.t.values)
    # 2-4. fit, diagnostics, posterior predictive checks, comparison
    res, elpd = {}, {}
    for kind in ["T0", "T1", "T2", "T3"]:
        r, e, disc, obs = fit(d, o, kind); res[kind] = r; elpd[kind] = e
        if kind == "T3": store[(o, "ppc")] = (disc, obs)
    for kind in ["T0", "T1", "T2", "T3"]:
        diff = elpd[kind] - elpd["T3"]; res[kind].update(delta_elpd_vs_T3=float(diff.sum()), se_delta=float(np.sqrt(len(diff) * diff.var())))
        rows.append(res[kind]); r = res[kind]
        print(f"{o:10s} {kind}: DIC {r['DIC']:.1f} WAIC {r['WAIC']:.1f} elpd {r['elpd_loo']:.1f} (d {r['delta_elpd_vs_T3']:.1f}, se {r['se_delta']:.1f}) k {r['pareto_k_max']:.2f} "
              f"pB chi2 {r['pB_chi2']:.2f} sd {r['pB_sd']:.2f} max {r['pB_max']:.2f} min {r['pB_min']:.2f} slope {r['pB_slope']:.2f} rhat {r['rhat_max']:.3f} div {r['divergences']} bfmi {r['ebfmi_min']:.2f}"
              + (f" OR {r['OR_decade']:.2f} ({r['OR_lo']:.2f}-{r['OR_hi']:.2f})" if 'OR_decade' in r else ""), flush=True)
    # 5. prior sensitivity (primary model)
    for b_sd, slope_sd in [(0.5, 0.5), (2.0, 0.5), (1.0, 0.25), (1.0, 1.0)]:
        r, _, _, _ = fit(d, o, "T3", b_sd, slope_sd); sens.append(r)
        print(f"    sens B~N(0,{b_sd}) slopes~HN({slope_sd}): OR {r['OR_decade']:.2f} ({r['OR_lo']:.2f}-{r['OR_hi']:.2f}) Pr(decline) {r['P_decline']:.3f}", flush=True)
pd.DataFrame(prior_rows).to_csv(T + "P39B_prior_predictive.csv", index=False); pd.DataFrame(rows).to_csv(T + "P39B_model_checks.csv", index=False)
pd.DataFrame(sens).to_csv(T + "P39B_prior_sensitivity.csv", index=False)
print(pd.DataFrame(prior_rows).round(2).to_string(index=False))

fig, ax = plt.subplots(3, 3, figsize=(7.2, 6.2))
for j, o in enumerate(OUT):
    pr = prior_rows[j]
    a = ax[0, j]; a.hist(np.clip(store[(o, "prior_sd")], 0, 6), bins=60, color="#B5B5B5", density=True); a.axvline(pr["sd_logit_obs"], color="#9E2A3A", ls="--")
    a.set_title(OUT[o], fontsize=9, fontweight="bold"); a.set_xlabel("SD of survey logits", fontsize=7.5)
    a = ax[1, j]; a.hist(np.clip(store[(o, "prior_slope")], -6, 6), bins=60, color="#B5B5B5", density=True); a.axvline(pr["slope_obs"], color="#9E2A3A", ls="--")
    a.set_xlabel("Slope of survey logits per decade", fontsize=7.5)
    disc, obs = store[(o, "ppc")]; a = ax[2, j]
    a.hist(disc["slope"], bins=60, color="#B5B5B5", density=True); a.axvline(obs["slope"][0], color="#9E2A3A", ls="--")
    a.text(0.02, 0.95, f"p = {np.mean(disc['slope'] >= obs['slope']):.2f}".replace(".", "·"), transform=a.transAxes, va="top", fontsize=8)
    a.set_xlabel("Slope of survey logits per decade", fontsize=7.5)
    for a in ax[:, j]: a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=7); a.set_yticks([])
ax[0, 0].set_ylabel("Prior predictive", fontsize=8.5); ax[1, 0].set_ylabel("Prior predictive", fontsize=8.5); ax[2, 0].set_ylabel("Posterior predictive", fontsize=8.5)
fig.tight_layout(); fig.savefig(F + "figS_trend_checks.png", dpi=300); plt.close(fig)
