"""79_review_pool.py - Pre-registration 38, Part B pooling.
B1: for every Table 2 term (sought any help), the hierarchical model with omega ~ HalfNormal(0.25 / 0.5 / 1.0) and the 95% prediction
    interval for a new country, exp(mu + omega*u_new + tau*z_new).
B2-B5, B11: the country estimates from 78_review_country_models.R pooled with the primary model (pool_numpyro.pool_table).
Outputs: outputs/tables/P38_review/P38_B1_prior_sensitivity.csv, P38_pooled.csv
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4"); sys.path.insert(0, "analysis")
import numpy as np, pandas as pd, jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import pool_numpyro as pn
numpyro.set_host_device_count(4)
T = "outputs/tables/P38_review/"
A2 = pd.read_csv("outputs/tables/P33_youth_v3/A2_country_estimates.csv")

def model(est, se, reg, n_reg, s_om):
    mu = numpyro.sample("mu", dist.Normal(0, 1)); om = numpyro.sample("omega", dist.HalfNormal(s_om)); tau = numpyro.sample("tau", dist.HalfNormal(0.5))
    with numpyro.plate("r", n_reg): u = numpyro.sample("u", dist.Normal(0, 1))
    with numpyro.plate("c", est.shape[0]):
        z = numpyro.sample("z", dist.Normal(0, 1)); numpyro.sample("obs", dist.Normal(mu + om * u[reg] + tau * z, se), obs=est)
    numpyro.deterministic("new", mu + om * numpyro.sample("u_new", dist.Normal(0, 1)) + tau * numpyro.sample("z_new", dist.Normal(0, 1)))

TERMS = [("partner", "injury"), ("partner", "severe"), ("partner", "alcohol"), ("main", "vtypeboth"), ("main", "maritalformer"),
         ("main", "working"), ("main", "educsecondary+"), ("main", "wealthq"), ("main", "urban")]
rows = []
for m, t in TERMS:
    d = A2[(A2.source == "DHS") & (A2.model == m) & (A2.outcome == "sought_any") & (A2.term == t)]
    regs = [r for r in pn.REG_ORDER if r in set(d.region)]; ri = jnp.array([regs.index(r) for r in d.region])
    for s_om in [0.25, 0.5, 1.0]:
        mc = MCMC(NUTS(model, target_accept_prob=0.995, max_tree_depth=12), num_warmup=2000, num_samples=2000, num_chains=4, chain_method="parallel", progress_bar=False)
        mc.run(jax.random.PRNGKey(20261023), est=jnp.array(d.est.values), se=jnp.array(d.se.values), reg=ri, n_reg=len(regs), s_om=s_om)
        s = mc.get_samples(); q = lambda v: np.exp(np.quantile(np.asarray(v), [.5, .025, .975]))
        p, pi = q(s["mu"]), q(s["new"])
        rows.append(dict(model=m, term=t, omega_prior_sd=s_om, countries=len(d), OR=p[0], OR_lo=p[1], OR_hi=p[2], PI_lo=pi[1], PI_hi=pi[2],
                         omega=float(np.median(s["omega"])), tau=float(np.median(s["tau"])), divergences=int(mc.get_extra_fields()["diverging"].sum())))
        print(f"{t:15s} HN({s_om}): OR {p[0]:.2f} ({p[1]:.2f}-{p[2]:.2f}) | PI {pi[1]:.2f}-{pi[2]:.2f}", flush=True)
pd.DataFrame(rows).to_csv(T + "P38_B1_prior_sensitivity.csv", index=False)

E = pd.read_csv(T + "P38_country_estimates.csv")
P = pn.pool_table(E, ["analysis", "model", "outcome", "term"], holm_by=("analysis", "model", "outcome"))
P.to_csv(T + "P38_pooled.csv", index=False)
