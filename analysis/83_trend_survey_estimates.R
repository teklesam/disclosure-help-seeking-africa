# 83_trend_survey_estimates.R - Pre-registration 39: survey-level cascade estimates for the trend curves.
# For each DHS survey: survey-weighted % of young women (15-24) with violence who told anyone, sought any help, sought formal help,
# with logit-method 95% CIs (PSU clustering) and the logit-scale standard error; survey time = mean interview date.
suppressMessages({library(data.table); library(survey)})
ROOT <- path.expand("~/Documents/OMS DHS Births Climate Policy")
s50 <- readLines(file.path(ROOT, "analysis/50_disclosure_predictors.R")); Sys.setenv(P33_SCOPE = "youth", AGE_CONT = "1")
eval(parse(text = s50[2:(grep("^V <- fread", s50)[1] - 1)]))                    # D exactly as in the primary analysis
D[, year := 1900 + mean(v008, na.rm = TRUE) / 12, by = survey]
options(survey.lonely.psu = "adjust")
qlogis_safe <- function(p) qlogis(pmin(pmax(p, 1e-4), 1 - 1e-4))
E <- rbindlist(lapply(c("told", "sought_any", "formal"), function(y) {
  d <- D[!is.na(get(y))]; des <- svydesign(ids = ~clus, weights = ~w, data = d, nest = TRUE)
  r <- as.data.table(svyby(as.formula(paste0("~", y)), ~survey, des, svyciprop, method = "logit", vartype = "ci", na.rm = TRUE))
  setnames(r, c("survey", "p", "lo", "hi"))
  r <- merge(r, d[, .(country = country[1], region = region[1], year = year[1], n = .N, events = sum(get(y))), by = survey], by = "survey")
  r[, `:=`(outcome = y, logit = qlogis_safe(p), se = (qlogis_safe(hi) - qlogis_safe(lo)) / (2 * 1.959964))] }))
E[, t := (year - 2015) / 10]
fwrite(E, file.path(ROOT, "outputs/tables/P38_review/P39_survey_estimates.csv"))
print(E[, .(surveys = .N, countries = uniqueN(country), min_pct = round(100 * min(p), 1), max_pct = round(100 * max(p), 1), zero_events = sum(events == 0)), by = outcome])
