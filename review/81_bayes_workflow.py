"""81_bayes_workflow.py - Pre-registration 38, addendum B12: Bayesian workflow for the pooling model.
Models for one factor's country log odds ratios b_c with standard errors s_c:
  M0 common effect          b_c ~ N(mu, s_c^2)
  M1 exchangeable countries b_c ~ N(theta_c, s_c^2), theta_c = mu + tau z_c
  M2 countries in regions   theta_c = mu + omega u_r(c) + tau z_c          (primary, equation 5)
  priors mu ~ N(0, 1); tau, omega ~ HalfNormal(0.5); NUTS 4 x (2000 + 2000), seed 20261023, target acceptance 0.995.
Outputs (outputs/tables/P38_review/): P38_B12_prior_predictive.csv, P38_B12_model_checks.csv; figures figS_prior_predictive.png, figS_ppc.png
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4"); sys.path.insert(0, "analysis")
import numpy as np, pandas as pd, jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist, arviz as az
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import summary
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
numpyro.set_host_device_count(4)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9})
T = "outputs/tables/P38_review/"; F = "outputs/figures/disclosure_v3/"
A2 = pd.read_csv("outputs/tables/P33_youth_v3/A2_country_estimates.csv")
REG = ["West", "Central", "East", "Southern"]

def model(se, reg, n_reg, kind, est=None):
    mu = numpyro.sample("mu", dist.Normal(0, 1))
    if kind == "M0":
        theta = mu * jnp.ones(se.shape[0])
    else:
        tau = numpyro.sample("tau", dist.HalfNormal(0.5))
        if kind == "M2":
            om = numpyro.sample("omega", dist.HalfNormal(0.5))
            with numpyro.plate("r", n_reg): u = numpyro.sample("u", dist.Normal(0, 1))
            m_c = mu + om * u[reg]
        else:
            m_c = mu * jnp.ones(se.shape[0])
        with numpyro.plate("c", se.shape[0]): z = numpyro.sample("z", dist.Normal(0, 1))
        theta = m_c + tau * z
        numpyro.deterministic("new", mu + (om * numpyro.sample("u_new", dist.Normal(0, 1)) if kind == "M2" else 0.0) + tau * numpyro.sample("z_new", dist.Normal(0, 1)))
    numpyro.deterministic("theta", theta)
    with numpyro.plate("obs_plate", se.shape[0]): numpyro.sample("obs", dist.Normal(theta, se), obs=est)

def fit(d, kind):
    regs = [r for r in REG if r in set(d.region)]; ri = jnp.array([regs.index(r) for r in d.region])
    se, est = jnp.array(d.se.values), jnp.array(d.est.values)
    mc = MCMC(NUTS(model, target_accept_prob=0.995, max_tree_depth=12), num_warmup=2000, num_samples=2000, num_chains=4, chain_method="parallel", progress_bar=False)
    mc.run(jax.random.PRNGKey(20261023), se=se, reg=ri, n_reg=len(regs), kind=kind, est=est, extra_fields=("diverging", "energy"))
    s = mc.get_samples(group_by_chain=True)
    keys = [k for k in ["mu", "tau", "omega"] if k in s]
    sm = summary({k: s[k] for k in keys}, group_by_chain=True)
    energy = np.asarray(mc.get_extra_fields(group_by_chain=True)["energy"])
    bfmi = float(np.min([np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2) for e in energy]))
    g = {k: np.asarray(v).reshape(4 * 2000, *np.asarray(v).shape[2:]) for k, v in s.items()}
    b, sv = d.est.values, d.se.values
    # marginal (over the country effect) pointwise log-likelihood, draws x countries
    tau = g.get("tau", np.zeros(len(g["mu"])))
    mean_c = g["mu"][:, None] + (g["omega"][:, None] * g["u"][:, np.asarray(ri)] if kind == "M2" else 0.0)
    sd_c = np.sqrt(sv[None, :] ** 2 + tau[:, None] ** 2)
    ll = norm.logpdf(b[None, :], mean_c, sd_c)
    Dbar = float(np.mean(-2 * ll.sum(1)))
    mean_hat = (g["mu"][:, None] + g["omega"][:, None] * g["u"]).mean(0)[np.asarray(ri)] if kind == "M2" else float(g["mu"].mean())   # posterior mean of each regional mean
    Dhat = float(-2 * norm.logpdf(b, mean_hat, np.sqrt(sv ** 2 + float(tau.mean()) ** 2)).sum())
    pD = Dbar - Dhat
    lppd = float(np.sum(np.log(np.mean(np.exp(ll), 0)))); p_waic = float(np.sum(np.var(ll, 0)))
    lo = az.loo(az.from_dict({"posterior": {"mu": np.asarray(s["mu"])}, "log_likelihood": {"obs": ll.reshape(4, 2000, -1)}}), var_name="obs", pointwise=True)
    # posterior predictive checks (conditional on theta_c)
    rng = np.random.default_rng(20261023); th = g["theta"]; rep = th + rng.normal(size=th.shape) * sv[None, :]
    chi_obs = ((b[None, :] - th) ** 2 / sv[None, :] ** 2).sum(1); chi_rep = ((rep - th) ** 2 / sv[None, :] ** 2).sum(1)
    stats = {"chi2": (chi_rep, chi_obs), "sd": (rep.std(1), np.full(len(rep), b.std())), "max": (rep.max(1), np.full(len(rep), b.max())), "min": (rep.min(1), np.full(len(rep), b.min()))}
    out = dict(model=kind, countries=len(d), regions=len(regs), OR=float(np.exp(np.median(g["mu"]))), OR_lo=float(np.exp(np.quantile(g["mu"], .025))), OR_hi=float(np.exp(np.quantile(g["mu"], .975))),
               rhat_max=max(float(np.max(sm[k]["r_hat"])) for k in keys), ess_bulk_min=min(float(np.min(sm[k]["n_eff"])) for k in keys),
               divergences=int(np.asarray(mc.get_extra_fields()["diverging"]).sum()), ebfmi_min=bfmi,
               DIC=Dbar + pD, pD=pD, WAIC=-2 * (lppd - p_waic), p_waic=p_waic, elpd_loo=float(lo.elpd), se_elpd_loo=float(lo.se), p_loo=float(lo.p),
               pareto_k_max=float(np.max(np.asarray(lo.pareto_k))))
    for k, (r_, o_) in stats.items(): out[f"pB_{k}"] = float(np.mean(r_ >= o_))
    if "new" in g:
        out["PI_lo"], out["PI_hi"] = float(np.exp(np.quantile(g["new"], .025))), float(np.exp(np.quantile(g["new"], .975)))
    return out, np.asarray(lo.elpd_i), stats

TERMS = [("sought_any", "partner", "injury"), ("sought_any", "partner", "severe"), ("sought_any", "partner", "alcohol"), ("sought_any", "main", "vtypeboth"),
         ("sought_any", "main", "maritalformer"), ("sought_any", "main", "working"), ("sought_any", "main", "educsecondary+"), ("sought_any", "main", "wealthq"),
         ("sought_any", "main", "urban"), ("told", "partner", "injury"), ("told", "partner", "severe"), ("told", "main", "vtypeboth"), ("told", "main", "maritalformer")]

# ---- 1. prior predictive (primary model M2, using the injury design: 21 countries, their SEs and regions) ----
d0 = A2[(A2.source == "DHS") & (A2.model == "partner") & (A2.outcome == "sought_any") & (A2.term == "injury")]
regs0 = [r for r in REG if r in set(d0.region)]; ri0 = jnp.array([regs0.index(r) for r in d0.region])
pp = Predictive(model, num_samples=8000)(jax.random.PRNGKey(1), se=jnp.array(d0.se.values), reg=ri0, n_reg=len(regs0), kind="M2", est=None)
new = np.asarray(pp["new"]); bsim = np.asarray(pp["obs"])
PP = pd.DataFrame([dict(quantity="OR for a new country", median=np.exp(np.median(new)), lo=np.exp(np.quantile(new, .025)), hi=np.exp(np.quantile(new, .975)),
                        share_outside_0p1_10=float(np.mean((np.exp(new) < 0.1) | (np.exp(new) > 10)))),
                   dict(quantity="Observed-scale country estimate exp(b_c)", median=np.exp(np.median(bsim)), lo=np.exp(np.quantile(bsim, .025)), hi=np.exp(np.quantile(bsim, .975)),
                        share_outside_0p1_10=float(np.mean((np.exp(bsim) < 0.1) | (np.exp(bsim) > 10)))),
                   dict(quantity="SD of country log ORs (prior predictive)", median=np.median(bsim.std(1)), lo=np.quantile(bsim.std(1), .025), hi=np.quantile(bsim.std(1), .975),
                        share_outside_0p1_10=np.nan)])
PP["observed_injury_SD_of_log_ORs"] = d0.est.std(); PP["observed_injury_OR_range"] = f"{np.exp(d0.est.min()):.2f}-{np.exp(d0.est.max()):.2f}"
PP.to_csv(T + "P38_B12_prior_predictive.csv", index=False); print(PP.round(3).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
bins = np.linspace(-4, 4, 81)
ax[0].hist(new, bins=bins, color="#B5B5B5", density=True); ax[0].set_xticks(np.log([0.05, 0.25, 1, 4, 20])); ax[0].set_xticklabels(["0·05", "0·25", "1", "4", "20"])
for x in d0.est: ax[0].axvline(x, ymax=0.08, color="black", lw=0.8)
ax[0].set_xlabel("Odds ratio for a new country (log scale)\nprior predictive; ticks = observed injury estimates"); ax[0].set_ylabel("Density"); ax[0].text(-0.12, 1.02, "A", transform=ax[0].transAxes, fontweight="bold", fontsize=11)
ax[1].hist(bsim.std(1), bins=60, range=(0, 2.5), color="#B5B5B5", density=True); ax[1].axvline(d0.est.std(), color="#9E2A3A", ls="--")
ax[1].set_xlabel("SD of the 21 country log odds ratios\nprior predictive; dashed line = observed (injury)"); ax[1].text(-0.12, 1.02, "B", transform=ax[1].transAxes, fontweight="bold", fontsize=11)
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(F + "figS_prior_predictive.png", dpi=300); plt.close(fig)

# ---- 2-5. fit M0, M1, M2; diagnostics, posterior predictive checks, DIC / WAIC / LOO ----
rows, ppc_store = [], {}
for o, m, t in TERMS:
    d = A2[(A2.source == "DHS") & (A2.model == m) & (A2.outcome == o) & (A2.term == t)].reset_index(drop=True)
    res, elpd_i = {}, {}
    for kind in ["M0", "M1", "M2"]:
        r, e, st = fit(d, kind); res[kind] = r; elpd_i[kind] = e
        if kind == "M2" and o == "sought_any" and t in ("injury", "severe", "vtypeboth", "maritalformer"): ppc_store[t] = st
    for kind in ["M0", "M1", "M2"]:
        diff = elpd_i[kind] - elpd_i["M2"]
        r = res[kind]; r.update(outcome=o, term=t, elpd_diff_vs_M2=float(diff.sum()), se_diff_vs_M2=float(np.sqrt(len(diff) * diff.var())))
        rows.append(r)
        print(f"{o:10s} {t:14s} {kind}: OR {r['OR']:.2f} ({r['OR_lo']:.2f}-{r['OR_hi']:.2f}) DIC {r['DIC']:.1f} WAIC {r['WAIC']:.1f} elpd {r['elpd_loo']:.1f} "
              f"(diff {r['elpd_diff_vs_M2']:.1f}, se {r['se_diff_vs_M2']:.1f}) k {r['pareto_k_max']:.2f} pB chi2 {r['pB_chi2']:.2f} sd {r['pB_sd']:.2f} max {r['pB_max']:.2f} min {r['pB_min']:.2f} "
              f"rhat {r['rhat_max']:.3f} div {r['divergences']} bfmi {r['ebfmi_min']:.2f}", flush=True)
R = pd.DataFrame(rows); R.to_csv(T + "P38_B12_model_checks.csv", index=False)

fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.8))
lab = {"injury": "Injury", "severe": "Severe violence", "vtypeboth": "Both types", "maritalformer": "Formerly in union"}
for j, t in enumerate(["injury", "severe", "vtypeboth", "maritalformer"]):
    st = ppc_store[t]
    for i, k in enumerate(["chi2", "sd"]):
        r_, o_ = st[k]; a = ax[i, j]
        if k == "chi2": a.scatter(o_, r_, s=1, alpha=0.15, color="grey"); lim = [0, max(o_.max(), r_.max())]; a.plot(lim, lim, color="black", lw=0.7)
        else: a.hist(r_, bins=50, color="#B5B5B5"); a.axvline(o_[0], color="#9E2A3A", ls="--")
        a.set_title(f"{lab[t]}: p = {np.mean(r_ >= o_):.2f}".replace(".", "·"), fontsize=8)
        a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=7)
ax[0, 0].set_ylabel("Chi-square, replicated", fontsize=8); ax[1, 0].set_ylabel("Count", fontsize=8)
for j in range(4): ax[0, j].set_xlabel("Chi-square, observed", fontsize=7); ax[1, j].set_xlabel("SD of country log ORs", fontsize=7)
fig.tight_layout(); fig.savefig(F + "figS_ppc.png", dpi=300); plt.close(fig)
