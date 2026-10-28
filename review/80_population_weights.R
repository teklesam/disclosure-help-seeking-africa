# 80_population_weights.R - Pre-registration 38, B9 (and B11 cascade): population-weighted headline cascade.
# DHS weights sum to each survey's sample size, so pooled percentages weight surveys by sample size. Here each country counts in
# proportion to its female population aged 15-24 (WorldPop R2025A, 1 km, survey year; years before 2015 use 2015, the first available),
# and a country's rounds share its weight equally:  w* = d005/1e6 x (N_cs / R_c) / sum over the survey of d005/1e6.
suppressMessages({library(data.table); library(terra); library(survey)})
ROOT <- path.expand("~/Documents/OMS DHS Births Climate Policy"); OUT2 <- file.path(ROOT, "outputs/tables/P38_review")
WP <- path.expand("~/Documents/OMS Climate Exposure Pipeline/data/raw/worldpop")
s50 <- readLines(file.path(ROOT, "analysis/50_disclosure_predictors.R")); Sys.setenv(P33_SCOPE = "youth", AGE_CONT = "1")
eval(parse(text = s50[2:(grep("^V <- fread", s50)[1] - 1)]))
YR <- fread(file.path(ROOT, "data/derived/dhs_survey_worldpop_years.csv"))[survey %in% unique(D$survey)]
YR[, pop_f1524 := mapply(function(iso, yr) {
  f <- file.path(WP, iso, yr, sprintf("%s_f_%s_%d_CN_1km_R2025A_UA_v1.tif", tolower(iso), c("15", "20"), yr))
  sum(global(rast(f[1]), "sum", na.rm = TRUE)$sum, global(rast(f[2]), "sum", na.rm = TRUE)$sum) }, iso3, wpyear)]
YR[, rounds := .N, by = country]
fwrite(YR, file.path(OUT2, "P38_B9_population_by_survey.csv"))
D <- merge(D, YR[, .(survey, pop_f1524, rounds)], by = "survey")
D[, w_pop := w * (pop_f1524 / rounds) / sum(w), by = survey]
D[, w_pop := w_pop / mean(w_pop)]                                                  # rescale to mean 1 (does not change percentages)
est <- function(d, y, wt, lab) { d <- d[!is.na(get(y))]; des <- svydesign(ids = ~clus, weights = as.formula(paste0("~", wt)), data = d, nest = TRUE)
  p <- svyciprop(as.formula(paste0("~", y)), des, method = "logit"); data.table(weighting = lab, outcome = y, pct = 100 * as.numeric(p), lo = 100 * confint(p)[1], hi = 100 * confint(p)[2], n = nrow(d)) }
R <- rbindlist(lapply(c("told", "sought_any", "formal", "formal_wide"), function(y) rbind(
  cbind(region = "All", rbind(est(D, y, "w", "sample-size (primary)"), est(D, y, "w_pop", "population"))),
  rbindlist(lapply(c("West", "Central", "East", "Southern"), function(r) cbind(region = r, rbind(est(D[region == r], y, "w", "sample-size (primary)"), est(D[region == r], y, "w_pop", "population"))))))))
fwrite(R, file.path(OUT2, "P38_B9_population_weighted_cascade.csv"))
print(R[, .(region, weighting, outcome, est = sprintf("%.1f (%.1f-%.1f)", pct, lo, hi))])
