# 52_disclosure_trends.R - Pre-registration 33, amendment A5 (run once): trends in help-seeking across DHS rounds, and whether drivers changed.
# Lines below up to '# ===== A5' are copied verbatim from 50_disclosure_predictors.R (data build, Gibbs pooling).
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
# ===== A5 =====
wm <- function(x, w) { ok <- !is.na(x) & !is.na(w); if (!any(ok)) NA_real_ else sum(x[ok] * w[ok]) / sum(w[ok]) }
RHS_M <- "agegrp + educ + wealthq + urban + marital + working + kids3 + media + decisions2 + justify + vtype + perp + recent + private"
RHS_P <- paste(RHS_M, "+ severe + injury + inj_na + alcohol + controlling3")
D[, syear := 1900 + mean(v008, na.rm = TRUE) / 12, by = survey]; D[, yr10 := (syear - 2015) / 10]
D[, phase := as.integer(substr(sub("^[A-Z]{2}_[0-9]{4}_", "", survey), 1, 1))]
D[, sec := as.integer(educ == "secondary+")]
MULTI <- D[, uniqueN(survey), by = country][V1 >= 2, country]; TT <- D[country %in% MULTI]
cat(sprintf("A5 sample: %d countries with >=2 rounds, %d surveys, %s women\n", length(MULTI), uniqueN(TT$survey), format(nrow(TT), big.mark = ",")))
fit <- function(d, y, rhs, terms, lpm = FALSE) {
  d <- d[!is.na(get(y))]; if (uniqueN(d$survey) < 2 || sum(d[[y]]) < 20) return(NULL)
  f <- as.formula(paste(y, "~", rhs))
  m <- tryCatch(if (lpm) feols(f, d, weights = ~w, cluster = ~clus, notes = FALSE, warn = FALSE) else feglm(f, d, family = binomial("logit"), weights = ~w, cluster = ~clus, notes = FALSE, warn = FALSE), error = function(e) NULL)
  if (is.null(m)) return(NULL); b <- coef(m); s <- se(m); k <- intersect(terms, names(b)); k <- k[is.finite(s[k]) & s[k] < 5]
  if (!length(k)) return(NULL); data.table(term = k, est = b[k], se = s[k], n = nobs(m), events = sum(d[[y]])) }
run <- function(dat, lab) {
  E <- list()
  for (k in sort(unique(dat$country))) { dk <- dat[country == k]
    for (y in c("formal", "sought_any", "told")) {
      add <- function(r, spec) if (!is.null(r)) E[[length(E) + 1]] <<- cbind(spec = spec, outcome = y, country = k, r)
      add(fit(dk, y, "yr10", "yr10"), "crude")
      add(fit(dk, y, paste("yr10 +", RHS_M), "yr10"), "adjusted")
      add(fit(dk, paste0(y, "_pp"), paste("yr10 +", RHS_M), "yr10", lpm = TRUE), "adjusted_LPM_pp")
      add(fit(dk, y, paste("yr10 +", RHS_M, "+ yr10:wealthq + yr10:sec"), c("yr10:wealthq", "yr10:sec")), "interaction_main")
      add(fit(dk[perp %in% c("partner", "both")], y, paste("yr10 +", RHS_P, "+ yr10:injury + yr10:severe"), c("yr10:injury", "yr10:severe")), "interaction_partner") } }
  E <- rbindlist(E); E[, `:=`(region = REG[country], set = SET[country], source = lab, model = spec)]
  E }
for (y in c("formal", "sought_any", "told")) TT[, (paste0(y, "_pp")) := get(y)]   # proportion scale for pooling; reported x100
set.seed(20261021)
POOLLIN <- function(x) { x <- x[is.finite(est)]; if (nrow(x) < 4) return(NULL); g <- gibbs(x$est, x$se, x$region)   # same model, linear scale (pp/decade)
  data.table(countries = nrow(x), est = 100 * median(g[, "mu"]), lo = 100 * quantile(g[, "mu"], .025), hi = 100 * quantile(g[, "mu"], .975)) }
summ <- function(E, lab) {
  lg <- E[model != "adjusted_LPM_pp"]; P <- POOL(lg, lab)
  lp <- E[model == "adjusted_LPM_pp", POOLLIN(.SD), by = .(outcome, term)]; lp[, scope := lab]
  list(P = P, L = lp) }
S0 <- run(TT, "DHS"); fwrite(S0, file.path(OUT, "A5_country_trends.csv"))
R0 <- summ(S0, "primary: ever-victims, all rounds")
R1 <- summ(run(TT[recent == 1], "DHS"), "S1: recent victims")
T2 <- TT[phase >= 6]; T2 <- T2[country %in% T2[, uniqueN(survey), by = country][V1 >= 2, country]]
R2 <- summ(run(T2, "DHS"), "S2: DHS phase >= 6")
R3d <- summ(S0[set == "discovery"], "S3: discovery"); R3h <- summ(S0[set == "held-out"], "S3: held-out")
PT <- rbindlist(list(R0$P, R1$P, R2$P, R3d$P, R3h$P), fill = TRUE)
INT <- PT[scope == "primary: ever-victims, all rounds" & grepl(":", term) & outcome == "formal"]; INT[, p_two := 2 * pmin(P_pos, 1 - P_pos)]
fwrite(PT, file.path(OUT, "A5_pooled_trends.csv")); fwrite(rbindlist(list(R0$L, R1$L, R2$L, R3d$L, R3h$L)), file.path(OUT, "A5_pooled_trends_pp.csv"))
# Holm on the four primary interactions: frequentist DL meta p-values (Bayesian pooling has no p-value)
MI <- S0[outcome == "formal" & grepl(":", term), { r <- dl(est, se); z <- r["est"] / sqrt(r["var"]); .(est = r["est"], se = sqrt(r["var"]), p = 2 * pnorm(-abs(z)), k = .N) }, by = term]
MI[, p_holm := p.adjust(p, "holm")]; fwrite(MI, file.path(OUT, "A5_interactions_holm.csv"))
DESC <- TT[, .(syear = round(syear[1], 1), n = .N, told = wm(told, w), sought_any = wm(sought_any, w), formal = wm(formal, w)), by = .(region, country, survey)][order(region, country, syear)]
fwrite(DESC, file.path(OUT, "A5_rates_by_round.csv"))
# ---- print ----
cat("\nA5a pooled trend per decade (OR, 95% CrI):\n"); print(PT[term == "yr10", .(scope, model, outcome, countries, OR = round(OR, 2), lo = round(OR_lo, 2), hi = round(OR_hi, 2), Pagree = round(P_sign_agree_regions, 2), pQ = signif(p_Qbetween, 2), West = round(OR_West, 2), Central = round(OR_Central, 2), East = round(OR_East, 2))])
cat("\nA5a adjusted, percentage points per decade:\n"); print(rbindlist(list(R0$L, R1$L, R2$L))[, lapply(.SD, function(x) if (is.numeric(x)) round(x, 2) else x)])
cat("\nA5b interactions (formal; DL meta + Holm):\n"); print(MI[, lapply(.SD, function(x) if (is.numeric(x)) signif(x, 3) else x)])
cat("\nA5b interactions (Bayesian pooled, all outcomes):\n"); print(PT[scope == "primary: ever-victims, all rounds" & grepl(":", term), .(outcome, term, OR = round(OR, 2), lo = round(OR_lo, 2), hi = round(OR_hi, 2))])
cat("\nRates by round:\n"); print(DESC[, lapply(.SD, function(x) if (is.numeric(x)) round(x, 3) else x)])
