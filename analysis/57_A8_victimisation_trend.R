# 57_A8_victimisation_trend.R - Pre-reg 33 amendment A8.2: is the decline in help-seeking among young women explained by changes in victimisation?
#   (a) trend in past-12-month victimisation among ALL DV-module women aged 15-24 (countries with >= 2 rounds)
#   (b) trend in sought any help among partner-violence victims, additionally adjusted for severe violence and injury
# Country estimates are written here; pooling is done in NumPyro (pool_numpyro.py).
suppressMessages({library(data.table); library(fixest)})
ROOT <- path.expand("~/Documents/OMS DHS Births Climate Policy"); OUT <- file.path(ROOT, "outputs/tables/P33_youth_v3"); dir.create(OUT, showWarnings = FALSE)
REG <- c(BF = "West", BJ = "West", CI = "West", GH = "West", GM = "West", GN = "West", LB = "West", ML = "West", MR = "West", NG = "West", SL = "West", SN = "West", TG = "West",
         AO = "Central", CD = "Central", CM = "Central", GA = "Central", TD = "Central",
         BU = "East", ET = "East", KE = "East", KM = "East", MD = "East", MW = "East", MZ = "East", RW = "East", TZ = "East", UG = "East", ZM = "East", ZW = "East",
         LS = "Southern", NM = "Southern", SZ = "Southern", ZA = "Southern")
W <- rbind(fread(file.path(ROOT, "data/derived/women_disclosure.csv.gz")), fread(file.path(ROOT, "data/derived/women_disclosure_heldout.csv.gz")), fill = TRUE)
W <- W[survey != "SN_2019_8B" & v012 <= 24]
W[, `:=`(w = d005 / 1e6, clus = paste(survey, v001), region = REG[country], age = v012 - 20,
         educ = factor(fcase(v106 == 0, "none", v106 == 1, "primary", default = "secondary+"), levels = c("none", "primary", "secondary+")),
         wealthq = v190, urban = as.integer(v025 == 1),
         marital = factor(fcase(v501 %in% 1:2, "current", v501 == 0, "never", default = "former"), levels = c("current", "former", "never")))]
W[, syear := 1900 + mean(v008, na.rm = TRUE) / 12, by = survey]; W[, yr10 := (syear - 2015) / 10]
MULTI <- W[!is.na(victim_12m), uniqueN(survey), by = country][V1 >= 2, country]
fit <- function(d, y, rhs) { d <- d[!is.na(get(y))]; if (uniqueN(d$survey) < 2 || sum(d[[y]]) < 20) return(NULL)
  m <- tryCatch(feglm(as.formula(paste(y, "~ yr10 +", rhs)), d, family = binomial("logit"), weights = ~w, cluster = ~clus, notes = FALSE, warn = FALSE), error = function(e) NULL)
  if (is.null(m) || !"yr10" %in% names(coef(m))) return(NULL); data.table(est = coef(m)["yr10"], se = se(m)["yr10"], n = nobs(m), rounds = uniqueN(d$survey)) }
BASE <- "age + educ + wealthq + urban + marital"
E <- list()
for (k in MULTI) { dk <- W[country == k]
  r <- fit(dk[!is.na(victim_12m)], "victim_12m", BASE); if (!is.null(r)) E[[length(E) + 1]] <- cbind(source = "DHS", model = "A8.2a victimisation (all women 15-24)", outcome = "victim_12m", term = "yr10", country = k, r)
  pv <- dk[!is.na(sought_any) & perp %in% c("partner", "both") & !is.na(severe) & !is.na(injury)]
  r <- fit(pv, "sought_any", BASE); if (!is.null(r)) E[[length(E) + 1]] <- cbind(source = "DHS", model = "A8.2b sought, partner victims, base covariates", outcome = "sought_any", term = "yr10", country = k, r)
  r <- fit(pv, "sought_any", paste(BASE, "+ severe + injury")); if (!is.null(r)) E[[length(E) + 1]] <- cbind(source = "DHS", model = "A8.2b sought, partner victims, + severity & injury", outcome = "sought_any", term = "yr10", country = k, r) }
E <- rbindlist(E); E[, region := REG[country]]
fwrite(E, file.path(OUT, "A8_2_country_trends.csv"))
desc <- W[country %in% MULTI & !is.na(victim_12m), .(syear = round(syear[1], 1), women = .N, victim_12m = weighted.mean(victim_12m, w),
                                                    sev_among_pv = W[survey == .BY$survey & perp %in% c("partner", "both") & !is.na(severe), weighted.mean(severe, w)]), by = .(region, country, survey)]
fwrite(desc[order(country, syear)], file.path(OUT, "A8_2_victimisation_by_round.csv"))
print(E[, .N, by = model])
