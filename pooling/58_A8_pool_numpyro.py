"""58_A8_pool_numpyro.py - Pre-reg 33 amendment A8.4: pool all youth (v3, continuous age) country estimates with the NumPyro hierarchical model.
Outputs in outputs/tables/P33_youth_v3/: A3_pooled_numpyro.csv, A3_replication_numpyro.csv, A3_sens50_numpyro.csv, A5_trends_numpyro.csv, A8_2_trends_numpyro.csv
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "analysis")
import numpy as np, pandas as pd, pool_numpyro as pn
T = "outputs/tables/P33_youth_v3/"
G = ["source", "model", "outcome", "term"]
CE = pd.read_csv(T + "A2_country_estimates.csv")
print("== A3 all countries"); PA = pn.pool_table(CE, G); PA.to_csv(T + "A3_pooled_numpyro.csv", index=False)
print("== A3 sensitivity >= 50 events"); pn.pool_table(CE[CE.events >= 50], G).to_csv(T + "A3_sens50_numpyro.csv", index=False)
print("== replication"); PD = pn.pool_table(CE[CE.set == "discovery"], G); PH = pn.pool_table(CE[CE.set == "held-out"], G)
k = ["OR", "OR_lo", "OR_hi", "P_pos"]
REP = PD[G + k].merge(PH[G + k], on=G, suffixes=("_d", "_h"))
REP["same_sign"] = np.sign(np.log(REP.OR_d)) == np.sign(np.log(REP.OR_h))
cred = lambda lo, hi: (lo > 1) | (hi < 1)
REP["both_credible"] = cred(REP.OR_lo_d, REP.OR_hi_d) & cred(REP.OR_lo_h, REP.OR_hi_h) & REP.same_sign
REP.to_csv(T + "A3_replication_numpyro.csv", index=False)
print("== A5 trends (primary, all rounds)")
TR = pd.read_csv(T + "A5_country_trends.csv"); TR = TR[TR.model != "adjusted_LPM_pp"]
pn.pool_table(TR, ["model", "outcome", "term"], holm_by=("model", "outcome")).to_csv(T + "A5_trends_numpyro.csv", index=False)
LP = pd.read_csv(T + "A5_country_trends.csv"); LP = LP[LP.model == "adjusted_LPM_pp"]   # linear probability: proportion scale; reported x100 = percentage points per decade
L = pn.pool_table(LP, ["model", "outcome", "term"], holm_by=("model", "outcome"))
for c in [c for c in L.columns if c.startswith("OR")]: L[c.replace("OR", "pp")] = 100 * np.log(L[c])
L.to_csv(T + "A5_trends_pp_numpyro.csv", index=False)
print("== A8.2 victimisation and severity-adjusted trends")
pn.pool_table(pd.read_csv(T + "A8_2_country_trends.csv"), G, holm_by=("model",)).to_csv(T + "A8_2_trends_numpyro.csv", index=False)
print("done")
