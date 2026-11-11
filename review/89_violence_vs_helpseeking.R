# 89_violence_vs_helpseeking.R - Correction 5, post-hoc addition 6: does falling violence explain the fall in help-seeking?
# (1) Survey-weighted prevalence of violence (ever; past 12 months) among ALL adolescent girls and young women aged 15-24 in the DV module.
# (2) Trend in prevalence: random-effects meta-regression (metafor, REML; random intercepts and slopes by country).
# (3) Help-seeking meta-regression (P39 survey estimates) with time only vs time + prevalence (country mean + within-country deviation).
# (4) Countries with >= 2 surveys: change in prevalence vs change in help-seeking, first to last survey.
# Outputs (outputs/tables/P38_review/): P89_violence_survey.csv, P89_results.csv, P89_country_changes.csv; figure outputs/figures/disclosure_v3/figS_violence_vs_help.png
suppressMessages({library(data.table); library(survey); library(metafor); library(ggplot2)})
ROOT <- Sys.getenv("RR"); T <- file.path(ROOT, "outputs/tables/P38_review"); FG <- file.path(ROOT, "outputs/figures/disclosure_v3")
W <- rbind(fread(file.path(ROOT, "data/derived/women_disclosure.csv.gz")), fread(file.path(ROOT, "data/derived/women_disclosure_heldout.csv.gz")), fill = TRUE)
W <- W[survey != "SN_2019_8B" & v012 <= 24]
W[, `:=`(w = d005 / 1e6, clus = paste(survey, v001))]; W[, year := 1900 + mean(v008, na.rm = TRUE) / 12, by = survey]
H <- fread(file.path(T, "P39_survey_estimates.csv"))[outcome == "sought_any"]      # help-seeking among survivors, by survey (script 83)
W <- W[survey %in% H$survey]
options(survey.lonely.psu = "adjust"); qs <- function(p) qlogis(pmin(pmax(p, 1e-4), 1 - 1e-4))
prev <- function(y) {
  d <- W[!is.na(get(y))]; des <- svydesign(ids = ~clus, weights = ~w, data = d, nest = TRUE)
  r <- as.data.table(svyby(as.formula(paste0("~", y)), ~survey, des, svyciprop, method = "logit", vartype = "ci", na.rm = TRUE)); setnames(r, c("survey", "p", "lo", "hi"))
  r <- merge(r, d[, .(country = country[1], year = year[1], n = .N), by = survey], by = "survey")
  r[, `:=`(measure = y, logit = qs(p), se = (qs(hi) - qs(lo)) / (2 * 1.959964))] }
P <- rbind(prev("victim_ever"), prev("victim_12m")); P[, t := (year - 2015) / 10]
fwrite(P, file.path(T, "P89_violence_survey.csv"))
cat("surveys with prevalence:", P[, .N, by = measure][, paste(measure, N)], "\n")

res <- list()
# (2) trend in prevalence
for (m in c("victim_ever", "victim_12m")) {
  d <- P[measure == m]; f <- rma.mv(logit, se^2, mods = ~ t, random = list(~ t | country, ~ 1 | survey), struct = "GEN", data = d, method = "REML")
  res[[length(res) + 1]] <- data.table(analysis = "prevalence trend", measure = m, term = "OR per decade", est = exp(coef(f)[["t"]]),
                                       lo = exp(f$ci.lb[2]), hi = exp(f$ci.ub[2]), p = f$pval[2], surveys = nrow(d), countries = uniqueN(d$country))
}
# (3) help-seeking trend with and without prevalence
for (m in c("victim_ever", "victim_12m")) {
  d <- merge(H[, .(survey, country, region, hlogit = logit, hse = se, t)], P[measure == m, .(survey, vlogit = logit)], by = "survey")
  d[, `:=`(v_mean = mean(vlogit), v_within = vlogit - mean(vlogit)), by = country]
  f0 <- rma.mv(hlogit, hse^2, mods = ~ t, random = list(~ t | country, ~ 1 | survey), struct = "GEN", data = d, method = "REML")
  f1 <- rma.mv(hlogit, hse^2, mods = ~ t + v_within + v_mean, random = list(~ t | country, ~ 1 | survey), struct = "GEN", data = d, method = "REML")
  g <- function(f, k) { i <- which(names(coef(f)) == k); c(exp(coef(f)[i]), exp(f$ci.lb[i]), exp(f$ci.ub[i]), f$pval[i]) }
  for (row in list(list("time only", "t", f0), list("time + prevalence", "t", f1), list("time + prevalence", "v_within", f1), list("time + prevalence", "v_mean", f1))) {
    v <- g(row[[3]], row[[2]])
    res[[length(res) + 1]] <- data.table(analysis = paste("help-seeking:", row[[1]]), measure = m,
      term = c(t = "OR per decade (time)", v_within = "OR per 1 logit higher prevalence, within country", v_mean = "OR per 1 logit higher prevalence, between countries")[[row[[2]]]],
      est = v[1], lo = v[2], hi = v[3], p = v[4], surveys = nrow(d), countries = uniqueN(d$country))
  }
}
R <- rbindlist(res); fwrite(R, file.path(T, "P89_results.csv")); print(R[, .(analysis, measure, term, est = sprintf("%.2f (%.2f-%.2f)", est, lo, hi), p = signif(p, 2), surveys)])

# (4) country changes, first to last survey
CH <- merge(H[, .(survey, country, region, year, help = 100 * p)], P[measure == "victim_ever", .(survey, viol = 100 * p, n)], by = "survey")
CH <- CH[, if (.N >= 2) { a <- .SD[which.min(year)]; b <- .SD[which.max(year)]
  .(region = region[1], first = floor(a$year), last = floor(b$year), d_viol = b$viol - a$viol, d_help = b$help - a$help, n = sum(n)) }, by = country]
fwrite(CH, file.path(T, "P89_country_changes.csv"))
ct <- cov.wt(CH[, .(d_viol, d_help)], wt = CH$n / sum(CH$n), cor = TRUE)$cor[1, 2]
cat(sprintf("\ncountries with >=2 surveys: %d | violence fell in %d, help-seeking fell in %d | weighted correlation of changes %.2f\n",
            nrow(CH), sum(CH$d_viol < 0), sum(CH$d_help < 0), ct))
md <- function(x, d = 1) sub("^-", "−", gsub(".", "·", formatC(x, format = "f", digits = d), fixed = TRUE))
g <- ggplot(CH, aes(d_viol, d_help)) + geom_hline(yintercept = 0, colour = "grey60", linetype = "22") + geom_vline(xintercept = 0, colour = "grey60", linetype = "22") +
  geom_point(aes(size = n, colour = region), alpha = 0.8) + geom_text(aes(label = country), size = 2.6, vjust = -1.1, family = "Helvetica") +
  scale_size_area(max_size = 6, guide = "none") + scale_colour_manual(values = c(West = "#C0392B", Central = "#2E8B3E", East = "#1A93A6"), name = NULL) +
  labs(x = "Change in % of adolescent girls and young women who ever experienced violence\n(first to last survey, percentage points)",
       y = "Change in % of survivors who sought any help\n(percentage points)",
       subtitle = sprintf("%d countries with repeated DHS surveys; point size = women analysed; weighted correlation %s", nrow(CH), md(ct, 2))) +
  theme_minimal(base_family = "Helvetica", base_size = 10) + theme(panel.grid.minor = element_blank(), legend.position = "top")
ragg::agg_png(file.path(FG, "figS_violence_vs_help.png"), width = 7.2, height = 5.4, units = "in", res = 300, background = "white"); print(g); invisible(dev.off())
cat("done\n")
