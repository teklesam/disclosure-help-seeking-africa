"""74_p37_pool.py - Pre-registration 37: pool DHS and VACS country estimates with a programme term.
    b_j ~ N(theta_j, s_j);  theta_j = mu + delta * x_j + omega * u_r(j) + tau * z_j;  x_j = -1/2 DHS, +1/2 VACS
    mu, delta ~ N(0, 1); tau, omega ~ HalfNormal(0.5); NUTS 4 x (2000 + 2000), seed 20261023, target acceptance 0.995 (as pool_numpyro.py).
Sensitivity 1: delta fixed at 0 (pool_numpyro.pool). Sensitivity 2: countries with both a DHS and a VACS estimate only.
Output: outputs/tables/P37_harmonised/P37_pooled.csv
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
sys.path.insert(0, "analysis")
import numpy as np, pandas as pd, jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary
import pool_numpyro as pn
numpyro.set_host_device_count(4)
T = "outputs/tables/P37_harmonised/"; REG = ["West", "Central", "East", "Southern"]

def model(est, se, x, reg, n_reg):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0)); delta = numpyro.sample("delta", dist.Normal(0.0, 1.0))
    omega = numpyro.sample("omega", dist.HalfNormal(0.5)); tau = numpyro.sample("tau", dist.HalfNormal(0.5))
    with numpyro.plate("regions", n_reg):
        u = numpyro.sample("u", dist.Normal(0.0, 1.0))
    mu_r = numpyro.deterministic("mu_r", mu + omega * u)
    with numpyro.plate("units", est.shape[0]):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
        numpyro.sample("obs", dist.Normal(mu_r[reg] + delta * x + tau * z, se), obs=est)

def pool2(d, seed=20261023):
    regs = [r for r in REG if r in set(d.region)]; ri = np.array([regs.index(r) for r in d.region]); x = np.where(d.source == "VACS", 0.5, -0.5)
    m = MCMC(NUTS(model, target_accept_prob=0.995, max_tree_depth=12), num_warmup=2000, num_samples=2000, num_chains=4, chain_method="parallel", progress_bar=False)
    m.run(jax.random.PRNGKey(seed), est=jnp.array(d.est.values), se=jnp.array(d.se.values), x=jnp.array(x), reg=jnp.array(ri), n_reg=len(regs))
    s = m.get_samples(); sg = m.get_samples(group_by_chain=True)
    dg = summary({k: sg[k] for k in ["mu", "delta", "tau", "omega", "mu_r"]}, group_by_chain=True)
    mu, de, mur = np.asarray(s["mu"]), np.asarray(s["delta"]), np.asarray(s["mu_r"])
    q = lambda v: (float(np.exp(np.median(v))), float(np.exp(np.quantile(v, .025))), float(np.exp(np.quantile(v, .975))))
    bd, bv = mu - de / 2, mu + de / 2
    out = dict(units=len(d), dhs_countries=int((d.source == "DHS").sum()), vacs_countries=int((d.source == "VACS").sum()))
    for name, v in [("pooled", mu), ("DHS", bd), ("VACS", bv), ("ratio_VACS_DHS", de)]:
        out[f"OR_{name}"], out[f"OR_{name}_lo"], out[f"OR_{name}_hi"] = q(v)
    out["P_same_sign"] = float((np.sign(bd) == np.sign(bv)).mean())
    out["tau"], out["omega"] = float(np.median(s["tau"])), float(np.median(s["omega"]))
    if "Southern" in regs:   # Southern regional mean, programme-balanced
        out["OR_Southern"], out["OR_Southern_lo"], out["OR_Southern_hi"] = q(mur[:, regs.index("Southern")])
    out["rhat_max"] = max(float(np.max(dg[k]["r_hat"])) for k in dg); out["ess_min"] = min(float(np.min(dg[k]["n_eff"])) for k in dg)
    out["divergences"] = int(m.get_extra_fields()["diverging"].sum())
    return out

CE = pd.read_csv(T + "P37_country_estimates.csv")
both_c = set(CE[CE.source == "DHS"].country) & set(CE[CE.source == "VACS"].country)
rows = []
for (o, t), d in CE.groupby(["outcome", "term"], sort=False):
    d = d.reset_index(drop=True)
    r = dict(outcome=o, term=t, **pool2(d))
    r["agreement"] = "differ" if (r["OR_ratio_VACS_DHS_lo"] > 1 or r["OR_ratio_VACS_DHS_hi"] < 1) else ("agree" if r["P_same_sign"] > 0.9 else "uncertain")
    s1 = pn.pool(d.est.values, d.se.values, list(d.region.values))                      # sensitivity 1: delta = 0
    r.update(S1_OR=s1["OR"], S1_OR_lo=s1["OR_lo"], S1_OR_hi=s1["OR_hi"])
    d2 = d[d.country.isin(both_c)].reset_index(drop=True)                                # sensitivity 2: overlap countries
    if d2.source.nunique() == 2 and len(d2) >= 6:
        s2 = pool2(d2); r.update(S2_units=s2["units"], S2_OR=s2["OR_pooled"], S2_OR_lo=s2["OR_pooled_lo"], S2_OR_hi=s2["OR_pooled_hi"],
                                 S2_ratio=s2["OR_ratio_VACS_DHS"], S2_ratio_lo=s2["OR_ratio_VACS_DHS_lo"], S2_ratio_hi=s2["OR_ratio_VACS_DHS_hi"])
    rows.append(r)
    print(f"{o:10s} {t:9s} pooled {r['OR_pooled']:.2f} ({r['OR_pooled_lo']:.2f}-{r['OR_pooled_hi']:.2f}) | DHS {r['OR_DHS']:.2f} ({r['OR_DHS_lo']:.2f}-{r['OR_DHS_hi']:.2f})"
          f" | VACS {r['OR_VACS']:.2f} ({r['OR_VACS_lo']:.2f}-{r['OR_VACS_hi']:.2f}) | ratio {r['OR_ratio_VACS_DHS']:.2f} ({r['OR_ratio_VACS_DHS_lo']:.2f}-{r['OR_ratio_VACS_DHS_hi']:.2f})"
          f" | {r['agreement']} | Rhat {r['rhat_max']:.3f} ESS {r['ess_min']:.0f} div {r['divergences']}", flush=True)
P = pd.DataFrame(rows); P.to_csv(T + "P37_pooled.csv", index=False)
print("overlap countries:", sorted(both_c))
