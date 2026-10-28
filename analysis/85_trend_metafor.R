# 85_trend_metafor.R - Pre-registration 39, formal frequentist tests: random-effects meta-regression of survey-level logit estimates on
# time (decades from 2015), with random intercepts and slopes by country (REML, metafor::rma.mv); Wald tests of the slope and of the
# region x time interaction (West, Central, East; Southern has no repeated surveys).
suppressMessages({library(data.table); library(metafor)})
ROOT <- path.expand("~/Documents/OMS DHS Births Climate Policy")
E <- fread(file.path(ROOT, "outputs/tables/P38_review/P39_survey_estimates.csv"))
R <- rbindlist(lapply(c("told", "sought_any", "formal"), function(o) {
  d <- E[outcome == o]; d[, vi := se^2]
  m1 <- rma.mv(logit, vi, mods = ~ t, random = list(~ t | country, ~ 1 | survey), struct = "GEN", data = d, method = "REML")   # Deviation 1
  d3 <- d[region != "Southern"]; d3[, region := factor(region, levels = c("West", "Central", "East"))]
  m2 <- rma.mv(logit, vi, mods = ~ region * t, random = list(~ t | country, ~ 1 | survey), struct = "GEN", data = d3, method = "REML")
  int <- anova(m2, btt = grep(":t$", rownames(coef(summary(m2)))))
  b <- coef(summary(m1))["t", ]
  data.table(outcome = o, surveys = nrow(d), OR_decade = exp(b$estimate), lo = exp(b$ci.lb), hi = exp(b$ci.ub), p_slope = b$pval,
             p_region_x_time = int$QMp, tau_slope = sqrt(m1$tau2[2]), sd_survey = sqrt(m1$sigma2[1])) }))
fwrite(R, file.path(ROOT, "outputs/tables/P38_review/P39_metafor_tests.csv"))
print(R[, lapply(.SD, function(x) if (is.numeric(x)) signif(x, 3) else x)])
