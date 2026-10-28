# 59_A8_cascade_ci.R - Pre-reg 33 amendment A8.5: survey-design 95% CIs for every cascade percentage (youth: DHS women 15-24; VACS 13-24).
# Weighted, PSU-within-survey clustering, logit-scale intervals (svyciprop, method = "logit").
suppressMessages({library(data.table); library(survey)})
options(survey.lonely.psu = "adjust")
ROOT <- path.expand("~/Documents/OMS DHS Births Climate Policy"); OUT <- file.path(ROOT, "outputs/tables/P33_youth_v3")
REG <- c(BF = "West", BJ = "West", CI = "West", GH = "West", GM = "West", GN = "West", LB = "West", ML = "West", MR = "West", NG = "West", SL = "West", SN = "West", TG = "West",
         AO = "Central", CD = "Central", CM = "Central", GA = "Central", TD = "Central",
         BU = "East", ET = "East", KE = "East", KM = "East", MD = "East", MW = "East", MZ = "East", RW = "East", TZ = "East", UG = "East", ZM = "East", ZW = "East",
         LS = "Southern", NM = "Southern", SZ = "Southern", ZA = "Southern")
W <- rbind(fread(file.path(ROOT, "data/derived/women_disclosure.csv.gz")), fread(file.path(ROOT, "data/derived/women_disclosure_heldout.csv.gz")), fill = TRUE)
D <- W[!is.na(sought_any) & vtype != "none" & survey != "SN_2019_8B"]
D[, told_inf := mean(told != sought_any, na.rm = TRUE) >= 0.01, by = survey]; D[told_inf == FALSE, told := NA]
D <- D[v012 <= 24]; D[, `:=`(w = d005 / 1e6, clus = paste(survey, v001), region = REG[country])]
ci_by <- function(dat, y, by, ids, wt) {
  d <- dat[!is.na(get(y))]; des <- svydesign(ids = as.formula(paste0("~", ids)), weights = as.formula(paste0("~", wt)), data = d, nest = TRUE)
  r <- svyby(as.formula(paste0("~", y)), as.formula(paste0("~", paste(by, collapse = "+"))), des, svyciprop, method = "logit", vartype = "ci", na.rm = TRUE)
  r <- as.data.table(r); setnames(r, c(by, "pct", "lo", "hi")); n <- d[, .(n = .N), by = by]
  merge(r, n, by = by)[, `:=`(outcome = y, pct = 100 * pct, lo = 100 * lo, hi = 100 * hi)][] }
out <- list()
for (y in c("told", "sought_any", "formal")) {
  out[[length(out) + 1]] <- cbind(level = "region", ci_by(D, y, "region", "clus", "w"))
  out[[length(out) + 1]] <- cbind(level = "country", ci_by(D, y, c("region", "country"), "clus", "w"))
  D[, all := "All countries"]; out[[length(out) + 1]] <- cbind(level = "all", ci_by(D, y, "all", "clus", "w")) }
DH <- rbindlist(out, fill = TRUE); fwrite(DH, file.path(OUT, "A8_5_dhs_cascade_ci.csv"))
V <- fread(path.expand("~/Documents/OMS VACS Climate Violence/data/processed/vacs_disclosure_harmonised.csv"))
V[, `:=`(psu = paste(file, psu), sex = fifelse(female == 1, "girls", "boys"))]
V[, w := w / mean(w, na.rm = TRUE), by = file]   # weights normalised within country (sample-size weighting across countries, as for DHS d005/1e6 and the A1 VACS cascade)
vo <- list()
for (t in c("sv", "pv")) for (o in c("tell", "knewplace", "sought", "recvdserv")) {
  d <- V[get(paste0(t, "_victim")) == 1]; y <- paste0(t, "_", o)
  vo[[length(vo) + 1]] <- cbind(type = toupper(t), step = o, level = "all VACS countries", ci_by(d, y, "sex", "psu", "w"))
  vo[[length(vo) + 1]] <- cbind(type = toupper(t), step = o, level = "country", ci_by(d, y, c("country", "sex"), "psu", "w")) }
VC <- rbindlist(vo, fill = TRUE); fwrite(VC, file.path(OUT, "A8_5_vacs_cascade_ci.csv"))
f <- function(x) sprintf("%.1f%% (%.1f-%.1f)", x$pct, x$lo, x$hi)
cat("\nDHS women 15-24 by region:\n"); print(DH[level %in% c("region", "all"), .(level, region = fcoalesce(region, all), outcome, est = sprintf("%.1f (%.1f-%.1f)", pct, lo, hi), n)])
cat("\nCountry ranges (min - max):\n"); print(DH[level == "country", .(min = sprintf("%.1f%% %s", min(pct), country[which.min(pct)]), max = sprintf("%.1f%% %s", max(pct), country[which.max(pct)]), countries = .N), by = outcome])
cat("\nVACS (all countries):\n"); print(VC[level == "all VACS countries", .(type, step, sex, est = sprintf("%.1f (%.1f-%.1f)", pct, lo, hi), n)])
