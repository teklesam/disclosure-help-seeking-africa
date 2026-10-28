# 50b_correction3_vacs_kenya.R - Pre-registration 33, CORRECTION 3 (2026-10-27): VACS country models silently excluded Kenya.
# Bug in 50_disclosure_predictors.R: the right-hand side was built as
#   c("female", "age1824", if (rural_ok) "rural", "hh_female", "poly")[c(TRUE, TRUE, rural_ok, TRUE, TRUE)]
# When a survey has no rural variable (Kenya 2019 only), the vector has four elements but the index has five, so "hh_female" was
# dropped and a literal NA term added; every Kenya model then failed inside tryCatch and was skipped.
# Fix: build the formula without the positional index. Only VACS rows of A2 are recomputed (code otherwise verbatim from script 50,
# youth v3 settings: AGE_CONT=1, so age1824 = age in years centred at 18); DHS rows are copied unchanged.
suppressMessages({library(data.table); library(fixest)})
ROOT <- path.expand("~/Documents/OMS DHS Births Climate Policy"); OUT <- file.path(ROOT, "outputs/tables/P33_youth_v3")
REG <- c(KE = "East", MW = "East", MZ = "East", TZ = "East", ZW = "East", LS = "Southern", NM = "Southern", SZ = "Southern")
SPLIT <- fread(file.path(ROOT, "data/derived/split_countries.csv")); SET <- setNames(SPLIT$set, SPLIT$country)
V <- fread(path.expand("~/Documents/OMS VACS Climate Violence/data/processed/vacs_disclosure_harmonised.csv"))
V[, `:=`(region = REG[country], w = w / mean(w, na.rm = TRUE))]
V[, age1824 := age - 18]                                                     # A8.3 (as in script 50 with AGE_CONT=1)
CE <- list()
for (k in sort(unique(V$country))) { vk <- V[country == k]
  for (t in c("sv", "pv")) for (o in c("tell", "sought", "knewplace", "recvdserv")) { y <- paste0(t, "_", o); dd <- vk[get(paste0(t, "_victim")) == 1]
    rural_ok <- dd[, any(!is.na(rural))]
    terms <- c(if (dd[, uniqueN(female)] >= 2) "female", "age1824", if (rural_ok) "rural", "hh_female", "poly")   # CORRECTION 3
    d2 <- dd[!is.na(get(y))]; if (nrow(d2) < 150 || sum(d2[[y]]) < 15) next
    m <- tryCatch(feglm(as.formula(paste(y, "~", paste(terms, collapse = " + "))), d2, family = binomial("logit"), weights = ~w, cluster = ~psu, notes = FALSE, warn = FALSE),
                  error = function(e) { message(k, " ", y, ": ", conditionMessage(e)); NULL })
    if (is.null(m)) next; b <- coef(m)[-1]; s <- se(m)[-1]; ok <- is.finite(s) & s < 5
    CE[[length(CE) + 1]] <- data.table(source = "VACS", model = toupper(t), outcome = o, country = k, term = names(b)[ok], est = b[ok], se = s[ok], n = nobs(m)) } }
CE <- rbindlist(CE); CE[, `:=`(region = REG[country], set = SET[country])]
EV <- rbindlist(lapply(c("tell", "sought", "knewplace", "recvdserv"), function(o) V[, .(source = "VACS", outcome = o, events = sum(sv_victim == 1 & get(paste0("sv_", o)) == 1, na.rm = TRUE) +
                                                                                           sum(pv_victim == 1 & get(paste0("pv_", o)) == 1, na.rm = TRUE)), by = country]))
CE <- merge(CE, EV, by = c("source", "country", "outcome"), all.x = TRUE)
OLD <- fread(file.path(OUT, "_backup_pre_correction3/A2_country_estimates.csv"))
# check: for countries other than Kenya the recomputed VACS rows must equal the originals
chk <- merge(OLD[source == "VACS"], CE[country != "KE"], by = c("source", "model", "outcome", "country", "term"), suffixes = c("_old", "_new"))
stopifnot(nrow(chk) == nrow(OLD[source == "VACS"]), nrow(chk) == nrow(CE[country != "KE"]), max(abs(chk$est_old - chk$est_new)) < 1e-8, max(abs(chk$se_old - chk$se_new)) < 1e-8)
NEW <- rbind(OLD[source == "DHS"], CE[, names(OLD), with = FALSE])
fwrite(NEW, file.path(OUT, "A2_country_estimates.csv"))
cat("Correction 3: non-Kenya VACS rows reproduced exactly (", nrow(chk), "rows ); Kenya rows added:", nrow(CE[country == "KE"]), "\n")
print(CE[country == "KE", .(model, outcome, n, terms = paste(term, collapse = ","))][, .SD[1], by = .(model, outcome)])
