# 69_southern_africa_pooled.R - exploratory (pre-reg 33 note, 2026-10-26): one pooled model for Southern Africa's young women (LS, NM, ZA),
# because no Southern country reaches the country-model threshold. Same covariates as the youth v3 models (age in years).
suppressMessages({library(data.table); library(fixest)})
ROOT <- path.expand("~/Documents/OMS DHS Births Climate Policy")
s50 <- readLines(file.path(ROOT, "analysis/50_disclosure_predictors.R"))
Sys.setenv(P33_SCOPE = "youth", AGE_CONT = "1")
eval(parse(text = s50[2:(grep("^V <- fread", s50)[1] - 1)]))                    # data build verbatim from script 50 (youth, continuous age)
RHS_M <- "agegrp + educ + wealthq + urban + marital + working + kids3 + media + decisions2 + justify + vtype + perp + recent + private"
RHS_P <- paste(RHS_M, "+ severe + injury + inj_na + alcohol + controlling3")
S <- D[country %in% c("LS", "NM", "ZA")]
fit <- function(d, rhs) { m <- feglm(as.formula(paste("sought_any ~", rhs, "| survey")), d[!is.na(sought_any)], family = binomial("logit"), weights = ~w, cluster = ~clus, notes = FALSE, warn = FALSE)
  ci <- confint(m); data.table(term = names(coef(m)), OR = exp(coef(m)), lo = exp(ci[, 1]), hi = exp(ci[, 2]), n = nobs(m), events = sum(d$sought_any, na.rm = TRUE)) }
R <- rbind(cbind(model = "main", fit(S, RHS_M)), cbind(model = "partner", fit(S[perp %in% c("partner", "both")], RHS_P)))
fwrite(R, file.path(ROOT, "outputs/tables/P33_youth_v3/southern_africa_pooled_exploratory.csv"))
print(R[(model == "partner" & term %in% c("injury", "severe", "alcohol")) | (model == "main" & term %in% c("vtypeboth", "maritalformer", "working", "educsecondary+", "wealthq", "urban")),
        .(model, term, OR = round(OR, 2), CI = sprintf("%.2f-%.2f", lo, hi), n, events)])
