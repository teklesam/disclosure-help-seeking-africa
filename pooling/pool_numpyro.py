"""pool_numpyro.py - Pre-reg 33 amendment A8.4: Bayesian hierarchical pooling of country estimates in NumPyro (NUTS).

Model (same as registered; non-centred):
    estimate_c ~ Normal(theta_c, se_c)          observed country log-odds ratio and its standard error
    theta_c    = mu_r[c] + tau * z_c            true country effect, z_c ~ Normal(0, 1)
    mu_r       = mu + omega * u_r               regional mean,        u_r ~ Normal(0, 1)
    mu ~ Normal(0, 1);  tau, omega ~ HalfNormal(0.5)
4 chains x (2,000 warm-up + 2,000 draws), seed 20261023, target acceptance 0.995 (small steps to avoid divergences in the omega funnel). Diagnostics: R-hat, bulk ESS, divergences.

Usage: python3 pool_numpyro.py <country_estimates.csv> <output.csv> [group columns...]
  input needs columns: est, se, region, plus the grouping columns (default: source model outcome term).
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary
numpyro.set_host_device_count(4)
REG_ORDER = ["West", "Central", "East", "Southern"]

def model(est, se, reg, n_reg):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    omega = numpyro.sample("omega", dist.HalfNormal(0.5))
    tau = numpyro.sample("tau", dist.HalfNormal(0.5))
    with numpyro.plate("regions", n_reg):
        u = numpyro.sample("u", dist.Normal(0.0, 1.0))
    mu_r = numpyro.deterministic("mu_r", mu + omega * u)
    with numpyro.plate("countries", est.shape[0]):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
        theta = mu_r[reg] + tau * z
        numpyro.sample("obs", dist.Normal(theta, se), obs=est)

def pool(est, se, region, seed=20261023, warmup=2000, draws=2000, chains=4):
    """Pool one predictor: returns a dict of posterior summaries (odds-ratio scale) and diagnostics."""
    regs = [r for r in REG_ORDER if r in set(region)]
    ri = np.array([regs.index(r) for r in region])
    mcmc = MCMC(NUTS(model, target_accept_prob=0.995, max_tree_depth=12), num_warmup=warmup, num_samples=draws, num_chains=chains, chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), est=jnp.array(est), se=jnp.array(se), reg=jnp.array(ri), n_reg=len(regs))
    s = mcmc.get_samples(); sg = mcmc.get_samples(group_by_chain=True)
    diag = summary({k: sg[k] for k in ["mu", "tau", "omega", "mu_r"]}, group_by_chain=True)
    rhat = max(float(np.max(diag[k]["r_hat"])) for k in diag); ess = min(float(np.min(diag[k]["n_eff"])) for k in diag)
    div = int(mcmc.get_extra_fields()["diverging"].sum())
    mu = np.asarray(s["mu"]); mur = np.asarray(s["mu_r"]); sg_ = np.sign(np.median(mu))
    out = dict(countries=len(est), regions=len(regs), OR=np.exp(np.median(mu)), OR_lo=np.exp(np.quantile(mu, .025)), OR_hi=np.exp(np.quantile(mu, .975)),
               P_pos=float((mu > 0).mean()), P_sign_agree_regions=float((np.sign(mur) == sg_).all(axis=1).mean()),
               tau=float(np.median(s["tau"])), omega=float(np.median(s["omega"])), rhat_max=rhat, ess_min=ess, divergences=div)
    for j, r in enumerate(regs):
        out[f"OR_{r}"] = float(np.exp(np.median(mur[:, j]))); out[f"OR_{r}_lo"] = float(np.exp(np.quantile(mur[:, j], .025))); out[f"OR_{r}_hi"] = float(np.exp(np.quantile(mur[:, j], .975)))
    return out

def dl(b, s):
    w = 1 / s**2; bf = (w * b).sum() / w.sum(); Q = (w * (b - bf)**2).sum(); k = len(b)
    t2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum())) if k > 1 else 0.0
    wr = 1 / (s**2 + t2); return (wr * b).sum() / wr.sum(), 1 / wr.sum()

def qbetween(b, s, rg):
    """Frequentist test that regional means differ (DerSimonian-Laird within region, then Q between regions)."""
    from scipy.stats import chi2
    R = pd.unique(rg); E = np.array([dl(b[rg == r], s[rg == r]) for r in R])
    if len(E) < 2: return np.nan
    w = 1 / E[:, 1]; m = (w * E[:, 0]).sum() / w.sum(); Q = (w * (E[:, 0] - m)**2).sum()
    return float(chi2.sf(Q, len(E) - 1))

def holm(p):
    p = np.asarray(p, float); o = np.argsort(np.where(np.isnan(p), np.inf, p)); m = np.isfinite(p).sum(); adj = np.full_like(p, np.nan); run = 0.0
    for rank, i in enumerate(o):
        if not np.isfinite(p[i]): continue
        run = max(run, min(1.0, (m - rank) * p[i])); adj[i] = run
    return adj

def pool_table(df, groups, holm_by=("source", "model", "outcome"), min_countries=4):
    rows = []
    for key, x in df.groupby(list(groups), sort=False):
        x = x[np.isfinite(x.est) & np.isfinite(x.se)]
        if len(x) < min_countries: continue
        r = pool(x.est.values, x.se.values, list(x.region.values))
        r.update(dict(zip(groups, key if isinstance(key, tuple) else (key,))))
        r["p_Qbetween"] = qbetween(x.est.values, x.se.values, x.region.values); rows.append(r)
        print(f"  {key}: OR {r['OR']:.2f} ({r['OR_lo']:.2f}-{r['OR_hi']:.2f}) | Rhat {r['rhat_max']:.3f} ESS {r['ess_min']:.0f} div {r['divergences']}", flush=True)
    P = pd.DataFrame(rows)
    hb = [g for g in holm_by if g in P.columns]
    P["p_holm_Q"] = P.groupby(hb)["p_Qbetween"].transform(lambda p: pd.Series(holm(p.values), index=p.index)) if hb else holm(P.p_Qbetween.values)
    P["class"] = np.where(P.p_holm_Q < 0.05, "region-specific", np.where((P.P_sign_agree_regions > 0.9) & ~(P.p_holm_Q < 0.05), "universal", "uncertain"))
    lead = list(groups) + ["countries", "regions", "OR", "OR_lo", "OR_hi", "P_pos", "P_sign_agree_regions", "tau", "omega", "p_Qbetween", "p_holm_Q", "class", "rhat_max", "ess_min", "divergences"]
    return P[lead + [c for c in P.columns if c not in lead]]

if __name__ == "__main__":
    inp, outp = sys.argv[1], sys.argv[2]; groups = sys.argv[3:] or ["source", "model", "outcome", "term"]
    df = pd.read_csv(inp)
    P = pool_table(df, groups); P.to_csv(outp, index=False)
    print(f"wrote {outp}: {len(P)} pooled estimates | max R-hat {P.rhat_max.max():.3f} | min ESS {P.ess_min.min():.0f} | divergences {int(P.divergences.sum())}")
