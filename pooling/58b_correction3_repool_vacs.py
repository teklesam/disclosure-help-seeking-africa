"""58b_correction3_repool_vacs.py - Pre-reg 33 CORRECTION 3: re-pool only the VACS rows after Kenya was added to A2 (50b).
Same model, seed and settings as 58_A8_pool_numpyro.py; DHS rows of every output are copied unchanged from the backup."""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "analysis")
import numpy as np, pandas as pd, pool_numpyro as pn
T = "outputs/tables/P33_youth_v3/"; B = T + "_backup_pre_correction3/"
G = ["source", "model", "outcome", "term"]
CE = pd.read_csv(T + "A2_country_estimates.csv"); V = CE[CE.source == "VACS"]
print("VACS countries per model/outcome:", V.groupby(["model", "outcome"]).country.nunique().to_dict())
def swap(name, new):
    old = pd.read_csv(B + name); out = pd.concat([old[old.source != "VACS"], new.reindex(columns=old.columns)], ignore_index=True); out.to_csv(T + name, index=False); return out
swap("A3_pooled_numpyro.csv", pn.pool_table(V, G))
swap("A3_sens50_numpyro.csv", pn.pool_table(V[V.events >= 50], G))
# Replication (A3_replication_numpyro.csv) is DHS only: VACS has too few countries per discovery/held-out set to pool, so it is unchanged.
o = pd.read_csv(B + "A3_pooled_numpyro.csv"); n = pd.read_csv(T + "A3_pooled_numpyro.csv")
assert o[o.source == "DHS"].reset_index(drop=True).equals(n[n.source == "DHS"].reset_index(drop=True)), "DHS rows changed"
m = o[o.source == "VACS"].merge(n[n.source == "VACS"], on=G, suffixes=("_old", "_new"))
print(m[G[1:] + ["countries_old", "countries_new", "OR_old", "OR_lo_old", "OR_hi_old", "OR_new", "OR_lo_new", "OR_hi_new", "class_old", "class_new", "divergences_new", "rhat_max_new"]].round(2).to_string())
