# 50_disclosure_predictors.R - Pre-registration 33 (run once): disclosure cascade, predictors, regional agreement, replication, climate moderation.
suppressMessages({library(data.table); library(fixest)})
ROOT <- path.expand("~/Documents/OMS DHS Births Climate Policy"); PIPE <- path.expand("~/Documents/OMS Climate Exposure Pipeline/outputs")
SCOPE <- Sys.getenv("P33_SCOPE", "all")   # AMENDMENT A6: "youth" = DHS women 15-24 (aligned with VACS 13-24); "all" = registered 15-49
AGECONT <- Sys.getenv("AGE_CONT", "0") == "1"   # AMENDMENT A8.3: age in years (youth: centred 20; VACS: centred 18)
OUT <- file.path(ROOT, "outputs/tables", if (SCOPE == "youth") (if (AGECONT) "P33_youth_v3" else "P33_youth") else "P33"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
REG <- c(BF = "West", BJ = "West", CI = "West", GH = "West", GM = "West", GN = "West", LB = "West", ML = "West", MR = "West", NG = "West", SL = "West", SN = "West", TG = "West", NI = "West",
         AO = "Central", CD = "Central", CM = "Central", GA = "Central", TD = "Central",
         BU = "East", ET = "East", KE = "East", KM = "East", MD = "East", MW = "East", MZ = "East", RW = "East", TZ = "East", UG = "East", ZM = "East", ZW = "East",
         LS = "Southern", NM = "Southern", SZ = "Southern", ZA = "Southern")
SPLIT <- fread(file.path(ROOT, "data/derived/split_countries.csv")); SET <- setNames(SPLIT$set, SPLIT$country)
# ================= DATA =================
W <- rbind(fread(file.path(ROOT, "data/derived/women_disclosure.csv.gz")), fread(file.path(ROOT, "data/derived/women_disclosure_heldout.csv.gz")), fill = TRUE)
D <- W[!is.na(sought_any) & vtype != "none"]
D <- D[survey != "SN_2019_8B"]; D[, told_inf := mean(told != sought_any, na.rm = TRUE) >= 0.01, by = survey]; D[told_inf == FALSE, told := NA]   # CORRECTION 2 (pre-reg 33): told only where d128 asked; drop 1-record SN 2019 file
if (SCOPE == "youth") D <- D[v012 <= 24]   # AMENDMENT A6
D[, `:=`(w = d005 / 1e6, clus = paste(survey, v001), region = REG[country],
         agegrp = if (AGECONT) v012 - 20 else if (SCOPE == "youth") factor(fifelse(v012 < 18, "15-17", "18-24"), levels = c("15-17", "18-24")) else factor(fcase(v012 < 25, "15-24", v012 < 35, "25-34", default = "35-49"), levels = c("25-34", "15-24", "35-49")),
         educ = factor(fcase(v106 == 0, "none", v106 == 1, "primary", default = "secondary+"), levels = c("none", "primary", "secondary+")),
         wealthq = v190, urban = as.integer(v025 == 1),
         marital = factor(fcase(v501 %in% 1:2, "current", v501 == 0, "never", default = "former"), levels = c("current", "former", "never")),
         kids3 = as.integer(children >= 3), decisions2 = fifelse(v501 %in% 1:2 & !is.na(decisions2), decisions2, 0),   # CORRECTION: effect among women in union; no dm_na
         vtype = factor(vtype, levels = c("physical", "sexual", "both")), perp = factor(perp, levels = c("partner", "nonpartner", "both")),
         recent = fifelse(is.na(victim_12m), NA_integer_, as.integer(victim_12m)), inj_na = as.integer(is.na(injury)), injury = fifelse(is.na(injury), 0, injury))]
for (y in c("formal", "sought_any", "told")) D[, (paste0(y, "_k")) := get(y)]
V <- fread(path.expand("~/Documents/OMS VACS Climate Violence/data/processed/vacs_disclosure_harmonised.csv"))
V[, `:=`(region = REG[country], w = w / mean(w, na.rm = TRUE))]
if (AGECONT) V[, age1824 := age - 18]   # A8.3: VACS age in years centred at 18 (term keeps the name age1824 = "per year of age")
cat(sprintf("DHS: %s ever-victims, %d surveys, %d countries | VACS: %s youth, %d countries\n", format(nrow(D), big.mark = ","), uniqueN(D$survey), uniqueN(D$country),
            format(nrow(V), big.mark = ","), uniqueN(V$country)))
# ================= A1 PREVALENCE =================
wm <- function(x, w) { ok <- !is.na(x) & !is.na(w); if (!any(ok)) NA_real_ else sum(x[ok] * w[ok]) / sum(w[ok]) }
P1 <- D[, .(n = .N, told = wm(told, w), sought_any = wm(sought_any, w), formal = wm(formal, w), recent_victims = sum(recent == 1, na.rm = TRUE)), by = .(region, country)][order(region, country)]
P1r <- D[, .(n = .N, told = wm(told, w), sought_any = wm(sought_any, w), formal = wm(formal, w)), by = region]
VP <- rbindlist(lapply(c("sv", "pv"), function(t) V[get(paste0(t, "_victim")) == 1, .(type = toupper(t), n = .N, tell = wm(get(paste0(t, "_tell")), w), knewplace = wm(get(paste0(t, "_knewplace")), w),
                       sought = wm(get(paste0(t, "_sought")), w), recvdserv = wm(get(paste0(t, "_recvdserv")), w)), by = .(region, country, female)]))
fwrite(P1, file.path(OUT, "A1_dhs_cascade_country.csv")); fwrite(P1r, file.path(OUT, "A1_dhs_cascade_region.csv")); fwrite(VP, file.path(OUT, "A1_vacs_cascade.csv"))
# ================= A2 COUNTRY MODELS =================
RHS_M <- "agegrp + educ + wealthq + urban + marital + working + kids3 + media + decisions2 + justify + vtype + perp + recent + private"
RHS_P <- paste(RHS_M, "+ severe + injury + inj_na + alcohol + controlling3")
cm <- function(d, y, rhs, fe = "survey", cl = ~clus) {
  d <- d[!is.na(get(y))]; if (nrow(d) < 300 || sum(d[[y]]) < 30) return(NULL)
  m <- tryCatch(feglm(as.formula(paste(y, "~", rhs, "|", fe)), d, family = binomial("logit"), weights = ~w, cluster = cl, notes = FALSE, warn = FALSE), error = function(e) NULL)
  if (is.null(m)) return(NULL); b <- coef(m); s <- se(m); ok <- is.finite(s) & s < 5
  data.table(term = names(b)[ok], est = b[ok], se = s[ok], n = nobs(m)) }
CE <- list()
for (k in sort(unique(D$country))) { dk <- D[country == k]
  for (y in c("formal", "sought_any", "told")) {
    r <- cm(dk, y, RHS_M); if (!is.null(r)) CE[[length(CE) + 1]] <- cbind(source = "DHS", model = "main", outcome = y, country = k, r)
    r <- cm(dk[perp %in% c("partner", "both")], y, RHS_P); if (!is.null(r)) CE[[length(CE) + 1]] <- cbind(source = "DHS", model = "partner", outcome = y, country = k, r) } }
for (k in sort(unique(V$country))) { vk <- V[country == k]
  for (t in c("sv", "pv")) for (o in c("tell", "sought", "knewplace", "recvdserv")) { y <- paste0(t, "_", o); dd <- vk[get(paste0(t, "_victim")) == 1]
    rhs <- paste(c(if (dd[, uniqueN(female)] >= 2) "female", "age1824", if (dd[, any(!is.na(rural))]) "rural", "hh_female", "poly"), collapse = " + ")   # CORRECTION 3 (2026-10-27): no positional index (it dropped Kenya)
    d2 <- dd[!is.na(get(y))]; if (nrow(d2) < 150 || sum(d2[[y]]) < 15) next
    m <- tryCatch(feglm(as.formula(paste(y, "~", rhs)), d2, family = binomial("logit"), weights = ~w, cluster = ~psu, notes = FALSE, warn = FALSE), error = function(e) NULL)
    if (is.null(m)) next; b <- coef(m)[-1]; s <- se(m)[-1]; ok <- is.finite(s) & s < 5
    CE[[length(CE) + 1]] <- data.table(source = "VACS", model = toupper(t), outcome = o, country = k, term = names(b)[ok], est = b[ok], se = s[ok], n = nobs(m)) } }
CE <- rbindlist(CE); CE[, `:=`(region = REG[country], set = SET[country])]
EV <- rbind(D[, .(source = "DHS", formal = sum(formal == 1, na.rm = TRUE), sought_any = sum(sought_any == 1, na.rm = TRUE), told = sum(told == 1, na.rm = TRUE)), by = country] |> melt(id.vars = c("source", "country"), variable.name = "outcome", value.name = "events"),
            rbindlist(lapply(c("tell", "sought", "knewplace", "recvdserv"), function(o) V[, .(source = "VACS", outcome = o, events = sum(sv_victim == 1 & get(paste0("sv_", o)) == 1, na.rm = TRUE) + sum(pv_victim == 1 & get(paste0("pv_", o)) == 1, na.rm = TRUE)), by = country])), fill = TRUE)
CE <- merge(CE, EV, by = c("source", "country", "outcome"), all.x = TRUE)
fwrite(CE, file.path(OUT, "A2_country_estimates.csv"))
cat(sprintf("country estimates: %d rows; DHS countries with formal-help main model: %d\n", nrow(CE), CE[source == "DHS" & model == "main" & outcome == "formal", uniqueN(country)]))
# ================= A2/A3 BAYESIAN HIERARCHICAL POOLING (theta_c ~ N(mu_r, tau^2); mu_r ~ N(mu, omega^2); mu ~ N(0,1); tau, omega ~ HN(0,0.5)) =================
gibbs <- function(th, s, rg, it = 6000, burn = 1500) {
  R <- sort(unique(rg)); ri <- match(rg, R); nR <- length(R); C <- length(th)
  theta <- th; mur <- tapply(th, ri, mean); mu <- mean(th); tau <- 0.2; om <- 0.2; keep <- matrix(NA, it - burn, nR + 3)
  ll_hn <- function(x, sc) -x^2 / (2 * sc^2)
  for (i in seq_len(it)) {
    v <- 1 / (1 / s^2 + 1 / tau^2); theta <- rnorm(C, v * (th / s^2 + mur[ri] / tau^2), sqrt(v))
    for (r in seq_len(nR)) { j <- ri == r; vr <- 1 / (sum(j) / tau^2 + 1 / om^2); mur[r] <- rnorm(1, vr * (sum(theta[j]) / tau^2 + mu / om^2), sqrt(vr)) }
    vm <- 1 / (nR / om^2 + 1); mu <- rnorm(1, vm * (sum(mur) / om^2), sqrt(vm))
    lt <- function(t) sum(dnorm(theta, mur[ri], t, log = TRUE)) + ll_hn(t, 0.5) + log(t)
    tp <- exp(log(tau) + rnorm(1, 0, 0.3)); if (log(runif(1)) < lt(tp) - lt(tau)) tau <- tp
    lo <- function(o) sum(dnorm(mur, mu, o, log = TRUE)) + ll_hn(o, 0.5) + log(o)
    op <- exp(log(om) + rnorm(1, 0, 0.3)); if (log(runif(1)) < lo(op) - lo(om)) om <- op
    if (i > burn) keep[i - burn, ] <- c(mu, mur, tau, om) }
  colnames(keep) <- c("mu", paste0("mu_", R), "tau", "omega"); keep }
dl <- function(b, s) { w <- 1 / s^2; bf <- sum(w * b) / sum(w); Q <- sum(w * (b - bf)^2); k <- length(b); t2 <- if (k > 1) max(0, (Q - (k - 1)) / (sum(w) - sum(w^2) / sum(w))) else 0
  wr <- 1 / (s^2 + t2); c(est = sum(wr * b) / sum(wr), var = 1 / sum(wr)) }
qbetween <- function(b, s, rg) { R <- unique(rg); E <- t(sapply(R, function(r) dl(b[rg == r], s[rg == r]))); if (nrow(E) < 2) return(NA_real_)
  w <- 1 / E[, "var"]; m <- sum(w * E[, "est"]) / sum(w); Q <- sum(w * (E[, "est"] - m)^2); pchisq(Q, df = nrow(E) - 1, lower.tail = FALSE) }
set.seed(20261021)
POOL <- function(cdt, lab) rbindlist(lapply(split(cdt, by = c("source", "model", "outcome", "term")), function(x) {
  x <- x[is.finite(est) & is.finite(se)]; if (nrow(x) < 4) return(NULL)
  g <- gibbs(x$est, x$se, x$region); regs <- grep("^mu_", colnames(g), value = TRUE)
  sg <- sign(median(g[, "mu"])); agree <- mean(apply(g[, regs, drop = FALSE], 1, function(r) all(sign(r) == sg)))
  out <- data.table(scope = lab, source = x$source[1], model = x$model[1], outcome = x$outcome[1], term = x$term[1], countries = nrow(x), regions = length(regs),
                    OR = exp(median(g[, "mu"])), OR_lo = exp(quantile(g[, "mu"], .025)), OR_hi = exp(quantile(g[, "mu"], .975)), P_pos = mean(g[, "mu"] > 0),
                    P_sign_agree_regions = agree, tau = median(g[, "tau"]), omega = median(g[, "omega"]), p_Qbetween = qbetween(x$est, x$se, x$region))
  for (r in regs) { out[, (paste0("OR_", sub("mu_", "", r))) := exp(median(g[, r]))] }
  out }), fill = TRUE)
PA <- POOL(CE, "all countries")
PS <- POOL(CE[events >= 50], "countries with >= 50 events (post-hoc sensitivity)"); fwrite(PS, file.path(OUT, "A3_pooled_regional_sens50.csv"))
PA[, p_holm_Q := p.adjust(p_Qbetween, "holm"), by = .(source, model, outcome)]
PA[, class := fifelse(!is.na(p_holm_Q) & p_holm_Q < 0.05, "region-specific", fifelse(P_sign_agree_regions > 0.9 & (is.na(p_holm_Q) | p_holm_Q > 0.05), "universal", "uncertain"))]
fwrite(PA, file.path(OUT, "A3_pooled_regional.csv"))
# ---- replication: discovery vs held-out countries ----
PD <- POOL(CE[set == "discovery"], "discovery"); PH <- POOL(CE[set == "held-out"], "held-out")
REP <- merge(PD[, .(source, model, outcome, term, OR_d = OR, lo_d = OR_lo, hi_d = OR_hi, Ppos_d = P_pos)], PH[, .(source, model, outcome, term, OR_h = OR, lo_h = OR_lo, hi_h = OR_hi, Ppos_h = P_pos)],
             by = c("source", "model", "outcome", "term"))
REP[, `:=`(same_sign = sign(log(OR_d)) == sign(log(OR_h)), both_credible = (lo_d > 1 | hi_d < 1) & (lo_h > 1 | hi_h < 1) & sign(log(OR_d)) == sign(log(OR_h)))]
fwrite(REP, file.path(OUT, "A3_replication.csv"))
# ================= A4 CLIMATE MODERATION (exploratory) =================
Rv <- D[recent == 1 & !is.na(formal)]
SB <- rbindlist(lapply(unique(Rv$survey), function(sv) { f <- file.path(PIPE, paste0(sv, "_speibase.csv.gz")); if (!file.exists(f)) return(NULL)
  z <- fread(f, select = c("unit_id", "date", "spei12")); z[, `:=`(survey = sv, v001 = as.integer(unit_id), cmc = (year(as.IDate(date)) - 1900) * 12 + month(as.IDate(date)))]; z[, .(survey, v001, cmc, spei12)] }))
setkey(SB, survey, v001, cmc)
SB <- unique(SB, by = c("survey", "v001", "cmc")); setkey(SB, survey, v001, cmc)
KK <- Rv[, .(survey, v001 = as.integer(v001), v008)]
MM <- sapply(1:12, function(j) SB[.(KK$survey, KK$v001, KK$v008 - j), spei12, mult = "first"])
Rv[, d12 := fifelse(rowSums(!is.na(MM)) == 12, -rowMeans(MM), NA_real_)]
Rv[, `:=`(imon = (v008 - 1) %% 12 + 1, sec = as.integer(educ == "secondary+"), rural = 1L - urban, partner = as.integer(perp %in% c("partner", "both")), fk = 1000 * formal)]
mA4 <- feols(fk ~ d12 + d12:sec + d12:wealthq + d12:rural + d12:partner + sec + wealthq + rural + partner + agegrp + marital | survey^v024^imon, Rv[!is.na(d12)], weights = ~w, cluster = ~clus, notes = FALSE)
ks <- grep("^d12", names(coef(mA4)), value = TRUE)
A4 <- data.table(term = ks, est = coef(mA4)[ks], se = se(mA4)[ks], p = pvalue(mA4)[ks], n = nobs(mA4)); A4[term != "d12", q_BH := p.adjust(p, "BH")]
fwrite(A4, file.path(OUT, "A4_climate_moderation.csv"))
# ================= PRINT =================
cat("\nA1 DHS by region:\n"); print(P1r[, lapply(.SD, function(x) if (is.numeric(x)) round(x, 3) else x)])
cat("\nA1 VACS (pooled by type x sex):\n"); print(VP[, .(n = sum(n), tell = wm(tell, n), sought = wm(sought, n), recv = wm(recvdserv, n)), by = .(type, female)][, lapply(.SD, function(x) if (is.numeric(x)) round(x, 3) else x)])
cat("\nA2/A3 DHS main model, FORMAL help (primary):\n"); print(PA[source == "DHS" & model == "main" & outcome == "formal", .(term, countries, OR = round(OR, 2), lo = round(OR_lo, 2), hi = round(OR_hi, 2), Pagree = round(P_sign_agree_regions, 2), tau = round(tau, 2), pQ_holm = signif(p_holm_Q, 2), class, OR_West = round(OR_West, 2), OR_Central = round(OR_Central, 2), OR_East = round(OR_East, 2), OR_Southern = round(OR_Southern, 2))])
cat("\nSENSITIVITY (>= 50 events), DHS main FORMAL:\n"); print(PS[source == "DHS" & model == "main" & outcome == "formal", .(term, countries, OR = round(OR, 2), lo = round(OR_lo, 2), hi = round(OR_hi, 2), Pagree = round(P_sign_agree_regions, 2), tau = round(tau, 2))])
cat("\nDHS partner model, FORMAL:\n"); print(PA[source == "DHS" & model == "partner" & outcome == "formal" & term %in% c("severe", "injury", "alcohol", "controlling3"), .(term, countries, OR = round(OR, 2), lo = round(OR_lo, 2), hi = round(OR_hi, 2), Pagree = round(P_sign_agree_regions, 2), class)])
cat("\nVACS sought help (primary):\n"); print(PA[source == "VACS" & outcome == "sought", .(model, term, countries, OR = round(OR, 2), lo = round(OR_lo, 2), hi = round(OR_hi, 2), Pagree = round(P_sign_agree_regions, 2), class)])
cat("\nClassification counts:\n"); print(PA[, .N, by = .(source, outcome, class)][order(source, outcome)])
cat("\nReplication (discovery vs held-out), DHS main formal:\n"); print(REP[source == "DHS" & model == "main" & outcome == "formal", .(term, OR_d = round(OR_d, 2), OR_h = round(OR_h, 2), same_sign, both_credible)])
cat(sprintf("Replication summary (all DHS terms/outcomes): same sign %.0f%%; credible & same sign in both %.0f%%\n", 100 * REP[source == "DHS", mean(same_sign)], 100 * REP[source == "DHS", mean(both_credible)]))
cat("\nA4 climate moderation (exploratory):\n"); print(A4[, lapply(.SD, function(x) if (is.numeric(x)) signif(x, 3) else x)])
