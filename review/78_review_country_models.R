# 78_review_country_models.R - Pre-registration 38, Part B (country level and one-stage parts), on the Correction 4 data.
#   B2 crude ORs; B3 reduced socio-demographic model; B4 timing-aligned sample; B5 prevalence ratios and risk differences;
#   B6 one-stage checks; B7 generalised VIF; B8 help sources by region; B11 formal_wide.
# Data preparation is taken verbatim from 50_disclosure_predictors.R (youth, continuous age), as in script 69.
suppressMessages({library(data.table); library(fixest); library(lme4); library(car)})
ROOT <- path.expand("~/Documents/OMS DHS Births Climate Policy"); OUT2 <- file.path(ROOT, "outputs/tables/P38_review"); dir.create(OUT2, showWarnings = FALSE)
s50 <- readLines(file.path(ROOT, "analysis/50_disclosure_predictors.R"))
Sys.setenv(P33_SCOPE = "youth", AGE_CONT = "1")
eval(parse(text = s50[2:(grep("^V <- fread", s50)[1] - 1)]))                     # builds D exactly as the primary analysis
RHS_M <- "agegrp + educ + wealthq + urban + marital + working + kids3 + media + decisions2 + justify + vtype + perp + recent + private"
RHS_P <- paste(RHS_M, "+ severe + injury + inj_na + alcohol + controlling3")
D <- D[region != "Southern"]                                                       # pooled analyses cover West, Central, East (as Table 2)
cat(sprintf("young women (West, Central, East): %s\n", format(nrow(D), big.mark = ",")))

fit <- function(d, y, rhs, keep, type = "logit", nmin = 300, emin = 30) {
  d <- d[!is.na(get(y))]; if (nrow(d) < nmin || sum(d[[y]]) < emin) return(NULL)
  f <- as.formula(paste(y, "~", rhs, "| survey"))
  m <- tryCatch(switch(type,
                       logit = feglm(f, d, family = binomial("logit"), weights = ~w, cluster = ~clus, notes = FALSE, warn = FALSE),
                       pr    = fepois(f, d, weights = ~w, cluster = ~clus, notes = FALSE, warn = FALSE),              # modified Poisson: prevalence ratio
                       rd    = feols(f, d, weights = ~w, cluster = ~clus, notes = FALSE, warn = FALSE)),              # linear probability: risk difference
                error = function(e) NULL)
  if (is.null(m)) return(NULL); b <- coef(m); s <- se(m); k <- intersect(keep, names(b)); k <- k[is.finite(s[k]) & s[k] < 5]
  if (!length(k)) return(NULL); data.table(term = k, est = b[k], se = s[k], n = nobs(m), events = sum(d[[y]])) }

T2 <- c("injury", "severe", "alcohol", "vtypeboth", "maritalformer", "working", "educsecondary+", "wealthq", "urban")
PART <- c("injury", "severe", "alcohol")
MAIN4 <- list(c("partner", "injury"), c("partner", "severe"), c("main", "vtypeboth"), c("main", "maritalformer"))
crude_rhs <- c(injury = "injury", severe = "severe", alcohol = "alcohol", vtypeboth = "vtype", maritalformer = "marital", working = "working",
               `educsecondary+` = "educ", wealthq = "wealthq", urban = "urban")
if (file.exists(file.path(OUT2, "P38_country_estimates.csv")) && nzchar(Sys.getenv("P38_REUSE"))) {
  E <- fread(file.path(OUT2, "P38_country_estimates.csv")) } else {
E <- list(); add <- function(r, analysis, model, y, k) if (!is.null(r)) E[[length(E) + 1]] <<- cbind(analysis = analysis, model = model, outcome = y, country = k, region = REG[[k]], r)
for (k in sort(unique(D$country))) {
  dk <- D[country == k]; dp <- dk[perp %in% c("partner", "both")]
  for (t in T2) {                                                                  # B2 crude
    dd <- if (t %in% PART) dp else dk; if (t == "injury") dd <- dd[inj_na == 0]
    add(fit(dd, "sought_any", crude_rhs[[t]], t), "B2_crude", if (t %in% PART) "partner" else "main", "sought_any", k) }
  add(fit(dk, "sought_any", "agegrp + educ + wealthq + urban", c("agegrp", "educprimary", "educsecondary+", "wealthq", "urban")), "B3_reduced", "main", "sought_any", k)
  dr <- dk[recent_only == 1]; drp <- dr[perp %in% c("partner", "both")]            # B4 timing-aligned
  add(fit(dr, "sought_any", RHS_M, c("vtypeboth", "maritalformer")), "B4_timing", "main", "sought_any", k)
  add(fit(drp, "sought_any", RHS_P, c("injury", "severe")), "B4_timing", "partner", "sought_any", k)
  for (ty in c("pr", "rd")) {                                                       # B5
    add(fit(dk, "sought_any", RHS_M, c("vtypeboth", "maritalformer"), ty), paste0("B5_", ty), "main", "sought_any", k)
    add(fit(dp, "sought_any", RHS_P, c("injury", "severe"), ty), paste0("B5_", ty), "partner", "sought_any", k) }
  add(fit(dk, "formal_wide", RHS_M, c("vtypeboth", "maritalformer")), "B11_formal_wide", "main", "formal_wide", k)   # B11
  add(fit(dp, "formal_wide", RHS_P, c("injury", "severe")), "B11_formal_wide", "partner", "formal_wide", k)
  add(fit(dk, "formal", RHS_M, c("vtypeboth", "maritalformer")), "B11_formal_primary", "main", "formal", k)
  add(fit(dp, "formal", RHS_P, c("injury", "severe")), "B11_formal_primary", "partner", "formal", k)
}
E <- rbindlist(E); E[, set := SET[country]]
fwrite(E, file.path(OUT2, "P38_country_estimates.csv")) }
print(E[, .(countries = uniqueN(country)), by = .(analysis, model, term)][order(analysis, model, term)])

# ---- B6 one-stage checks (countries that entered the two-stage models) ----
A2 <- fread(file.path(ROOT, "outputs/tables/P33_youth_v3/A2_country_estimates.csv"))[source == "DHS" & outcome == "sought_any"]
OS <- rbindlist(lapply(MAIN4, function(mt) {
  m_name <- mt[1]; t <- mt[2]; cs <- A2[model == m_name & term == t, unique(country)]
  d <- D[country %in% cs & !is.na(sought_any)]; if (m_name == "partner") d <- d[perp %in% c("partner", "both")]
  rhs <- if (m_name == "partner") RHS_P else RHS_M
  m1 <- feglm(as.formula(paste("sought_any ~", rhs, "| survey")), d, family = binomial("logit"), weights = ~w, cluster = ~clus, notes = FALSE)
  d2 <- copy(d); d2[, x := if (t == "vtypeboth") as.numeric(vtype == "both") else if (t == "maritalformer") as.numeric(marital == "former") else as.numeric(get(t))]
  d2 <- d2[complete.cases(d2[, all.vars(as.formula(paste("~", rhs))), with = FALSE])]
  m2 <- glmer(as.formula(paste("sought_any ~", rhs, "+ (1 | survey) + (0 + x | country)")), d2, family = binomial("logit"),
              control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)), nAGQ = 0)
  b2 <- fixef(m2)[t]; s2 <- sqrt(diag(vcov(m2)))[t]
  data.table(model = m_name, term = t, countries = length(cs), women = nobs(m1),
             onestage_weighted_OR = exp(coef(m1)[t]), lo1 = exp(coef(m1)[t] - 1.96 * se(m1)[t]), hi1 = exp(coef(m1)[t] + 1.96 * se(m1)[t]),
             glmm_OR = exp(b2), lo2 = exp(b2 - 1.96 * s2), hi2 = exp(b2 + 1.96 * s2), glmm_slope_sd = sqrt(VarCorr(m2)$country[1])) }))
fwrite(OS, file.path(OUT2, "P38_B6_onestage.csv")); print(OS)

# ---- B7 generalised VIF on the pooled data (linear proxy; collinearity is a property of the predictors) ----
vif_tab <- function(d, rhs, lab) { v <- car::vif(lm(as.formula(paste("sought_any ~", rhs)), data = d))
  v <- if (is.matrix(v)) data.table(term = rownames(v), adj = v[, 3]) else data.table(term = names(v), adj = sqrt(v)); v[, model := lab][] }
VIF <- rbind(vif_tab(D, RHS_M, "main"), vif_tab(D[perp %in% c("partner", "both")], RHS_P, "partner"))
setnames(VIF, "adj", "GVIF_1_2df"); fwrite(VIF, file.path(OUT2, "P38_B7_vif.csv")); cat("max GVIF^(1/(2df)):", round(max(VIF$GVIF_1_2df), 2), "\n")

# ---- B8 help sources among young women who sought help (all four regions) ----
eval(parse(text = s50[2:(grep("^V <- fread", s50)[1] - 1)]))                        # rebuild D including Southern Africa
S <- D[sought_any == 1]
src <- grep("^src_", names(S), value = TRUE)
HS <- rbindlist(lapply(src, function(v) rbind(
  S[!is.na(get(v)), .(source = v, region = "All", pct = 100 * weighted.mean(get(v), w), women = .N, surveys = uniqueN(survey))],
  S[!is.na(get(v)), .(source = v, pct = 100 * weighted.mean(get(v), w), women = .N, surveys = uniqueN(survey)), by = region])))
fwrite(HS, file.path(OUT2, "P38_B8_help_sources.csv")); print(dcast(HS, source ~ region, value.var = "pct")[order(-All)])

# ---- Sample characteristics (descriptive; STROBE item 14), survey-weighted, by region ----
pc <- function(x, w) 100 * weighted.mean(x, w, na.rm = TRUE)
ch <- function(d) d[, .(women = .N, surveys = uniqueN(survey), countries = uniqueN(country), age_mean = weighted.mean(v012, w),
                        secondary = pc(educ == "secondary+", w), urban = pc(urban, w), in_union = pc(marital == "current", w),
                        formerly_in_union = pc(marital == "former", w), never_in_union = pc(marital == "never", w), working = pc(working, w),
                        physical_only = pc(vtype == "physical", w), sexual_only = pc(vtype == "sexual", w), both_types = pc(vtype == "both", w),
                        partner_violence = pc(perp %in% c("partner", "both"), w), past_12_months = pc(recent, w),
                        injury_among_partner = pc(fifelse(inj_na == 1, NA_real_, injury)[perp %in% c("partner", "both")], w[perp %in% c("partner", "both")]),
                        severe_among_partner = pc(severe[perp %in% c("partner", "both")], w[perp %in% c("partner", "both")]),
                        missing_working = sum(is.na(working)), missing_recent = sum(is.na(recent)), missing_severe = sum(is.na(severe) & perp %in% c("partner", "both")))]
CH <- rbind(cbind(region = "All", ch(D)), rbindlist(lapply(c("West", "Central", "East", "Southern"), function(r) cbind(region = r, ch(D[region == r])))))
fwrite(CH, file.path(OUT2, "P38_sample_characteristics.csv")); print(t(CH))
