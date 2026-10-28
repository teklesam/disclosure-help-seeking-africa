# 73_p37_country_models.R - Pre-registration 37: country x programme models for the harmonised DHS-VACS pool.
# Young women 15-24 who experienced violence; outcomes told anyone and sought any help; common predictors age (centred 20), rural residence,
# female household head, both physical and sexual violence. Survey-weighted logit, PSU-clustered SEs; DHS with survey-round fixed effects.
suppressMessages({library(data.table); library(fixest)})
ROOT <- path.expand("~/Documents/OMS DHS Births Climate Policy"); OUT <- file.path(ROOT, "outputs/tables/P37_harmonised"); dir.create(OUT, showWarnings = FALSE)
REG <- c(BF = "West", BJ = "West", CI = "West", GH = "West", GM = "West", GN = "West", LB = "West", ML = "West", MR = "West", NG = "West", SL = "West", SN = "West", TG = "West", NI = "West",
         AO = "Central", CD = "Central", CM = "Central", GA = "Central", TD = "Central",
         BU = "East", ET = "East", KE = "East", KM = "East", MD = "East", MW = "East", MZ = "East", RW = "East", TZ = "East", UG = "East", ZM = "East", ZW = "East",
         LS = "Southern", NM = "Southern", SZ = "Southern", ZA = "Southern")
TERMS <- c("age", "rural", "hh_female", "both")
# ---- DHS (same sample as the primary youth analysis) ----
W <- rbind(fread(file.path(ROOT, "data/derived/women_disclosure.csv.gz")), fread(file.path(ROOT, "data/derived/women_disclosure_heldout.csv.gz")), fill = TRUE)
D <- W[!is.na(sought_any) & vtype != "none" & survey != "SN_2019_8B"]
D[, told_inf := mean(told != sought_any, na.rm = TRUE) >= 0.01, by = survey]; D[told_inf == FALSE, told := NA]   # Correction 2 rule
D <- D[v012 <= 24]
H <- fread(file.path(ROOT, "data/derived/women_hhhead.csv.gz"), colClasses = c(caseid = "character"))
D[, caseid := as.character(caseid)]; D <- merge(D, H, by = c("survey", "caseid"), all.x = TRUE); stopifnot(!anyNA(D$hh_female))
D[, `:=`(w = d005 / 1e6, clus = paste(survey, v001), age = v012 - 20, rural = as.integer(v025 == 2), both = as.integer(vtype == "both"))]
# ---- VACS girls 15-24 ----
V <- fread(path.expand("~/Documents/OMS VACS Climate Violence/data/processed/vacs_disclosure_harmonised.csv"))
V <- V[female == 1 & age >= 15 & age <= 24 & (sv_victim == 1 | pv_victim == 1)]
anyof <- function(a, b) fifelse(is.na(a) & is.na(b), NA_real_, as.numeric(pmax(a, b, na.rm = TRUE) > 0))
V[, `:=`(told = anyof(fifelse(sv_victim == 1, sv_tell, NA_real_), fifelse(pv_victim == 1, pv_tell, NA_real_)),
         sought_any = anyof(fifelse(sv_victim == 1, sv_sought, NA_real_), fifelse(pv_victim == 1, pv_sought, NA_real_)),
         age = age - 20, both = as.integer(poly == 1), clus = psu)]
cat(sprintf("DHS young women: %s (%d countries) | VACS girls 15-24 with violence: %s (%d countries)\n", format(nrow(D), big.mark = ","), uniqueN(D$country),
            format(nrow(V), big.mark = ","), uniqueN(V$country)))
fit <- function(d, y, src, k) {
  d <- d[!is.na(get(y)) & !is.na(age) & !is.na(hh_female) & !is.na(both)]
  terms <- TERMS[sapply(TERMS, function(t) { x <- d[[t]]; !all(is.na(x)) && uniqueN(x[!is.na(x)]) > 1 })]   # e.g. rural is missing in Kenya VACS
  d <- d[complete.cases(d[, ..terms])]
  nmin <- if (src == "DHS") c(300, 30) else c(150, 15); if (nrow(d) < nmin[1] || sum(d[[y]]) < nmin[2]) return(NULL)
  f <- as.formula(paste(y, "~", paste(terms, collapse = " + "), if (src == "DHS" && uniqueN(d$survey) > 1) "| survey" else ""))
  m <- tryCatch(feglm(f, d, family = binomial("logit"), weights = ~w, cluster = ~clus, notes = FALSE, warn = FALSE), error = function(e) { message(src, k, y, ": ", conditionMessage(e)); NULL })
  if (is.null(m)) return(NULL); b <- coef(m); s <- se(m); keep <- names(b) %in% TERMS & is.finite(s) & s < 5
  data.table(source = src, country = k, region = REG[[k]], outcome = y, term = names(b)[keep], est = b[keep], se = s[keep], n = nobs(m), events = sum(d[[y]])) }
CE <- rbindlist(c(lapply(sort(unique(D$country)), function(k) rbindlist(lapply(c("told", "sought_any"), function(y) fit(D[country == k], y, "DHS", k)))),
                  lapply(sort(unique(V$country)), function(k) rbindlist(lapply(c("told", "sought_any"), function(y) fit(V[country == k], y, "VACS", k))))))
fwrite(CE, file.path(OUT, "P37_country_estimates.csv"))
print(CE[, .(countries = uniqueN(country), regions = paste(sort(unique(region)), collapse = ",")), by = .(source, outcome, term)])
