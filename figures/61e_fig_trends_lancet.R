# 61e_fig_trends_lancet.R - Figure 3: trends in disclosure and help-seeking among young women, 2004-2025 (Pre-registration 39).
# Rows: told anyone, sought any help, sought formal help. Columns: all countries and UN regions.
# Points: survey-weighted survey estimates (size = women); thin lines join a country's rounds.
# Curves and bands: posterior median and 95% CrI from the hierarchical meta-regression (84_trend_model.py).
# Text: odds ratio per decade (95% CrI) and posterior probability of decline; Southern Africa has no repeated surveys.
suppressMessages({library(data.table); library(ggplot2)})
ROOT <- path.expand("~/Documents/OMS DHS Births Climate Policy"); T <- file.path(ROOT, "outputs/tables/P38_review"); FG <- file.path(ROOT, "outputs/figures/disclosure_v3")
FONT <- "Times New Roman"; NAVY <- "#1B2A6B"; GREY <- "#8C8C8C"
md <- function(x, d = 2) gsub(".", "·", formatC(x, format = "f", digits = d), fixed = TRUE)
OUT <- c(told = "Told anyone", sought_any = "Sought any help", formal = "Sought formal help")
SC <- c("All", "West", "Central", "East", "Southern"); SCL <- c(All = "All countries", West = "West Africa", Central = "Central Africa", East = "East Africa", Southern = "Southern Africa")
E <- fread(file.path(T, "P39_survey_estimates.csv")); C <- fread(file.path(T, "P39_curves.csv")); S <- fread(file.path(T, "P39_slopes.csv")); M <- fread(file.path(T, "P39_metafor_tests.csv"))
pts <- rbind(E[, .(outcome, scope = region, country, year, pct = 100 * p, n)], E[, .(outcome, scope = "All", country, year, pct = 100 * p, n)])
C[is.na(trend_estimable), trend_estimable := TRUE]
cur <- C[scope != "Southern"]
south_ref <- C[scope == "All" & year >= E[region == "Southern", min(year)] - 1 & year <= E[region == "Southern", max(year)] + 1]
south_ref <- south_ref[, scope := "Southern"]
lab <- S[, .(outcome, scope, txt = fifelse(scope == "Southern", "No repeated\nsurveys: overall\ntrend (dashed)",
                                     sprintf("OR/decade %s\n(%s–%s)\nPr(decline) %s", md(OR_decade), md(OR_lo), md(OR_hi), md(P_decline))))]
lab <- merge(lab, M[, .(outcome, p = p_slope)], by = "outcome")
lab[scope == "All", txt := paste0(txt, "\nMeta-regression p", fifelse(p < 0.001, "<0·001", paste0("=", md(p, 2))))]
YL <- data.table(outcome = names(OUT), ymax = c(115, 76, 15.5))
lab <- merge(lab, YL, by = "outcome")
for (d in list(pts, cur, south_ref, lab, YL)) { d[, outcome := factor(OUT[outcome], levels = OUT)] }
for (d in list(pts, cur, south_ref, lab)) { d[, scope := factor(SCL[scope], levels = SCL)] }
blank <- YL[, .(outcome, ymax)][rep(1:3, each = 5)][, scope := factor(rep(SCL, 3), levels = SCL)]
g <- ggplot() +
  geom_blank(data = blank, aes(x = 2015, y = ymax)) + geom_blank(data = blank, aes(x = 2015, y = 0)) +
  geom_line(data = pts, aes(year, pct, group = country), colour = "grey80", linewidth = 0.3) +
  geom_point(data = pts, aes(year, pct, size = n), shape = 21, fill = "grey75", colour = "grey45", stroke = 0.25, alpha = 0.9) +
  geom_ribbon(data = cur, aes(year, ymin = lo, ymax = hi), fill = NAVY, alpha = 0.18) +
  geom_line(data = cur, aes(year, pct), colour = NAVY, linewidth = 0.8) +
  geom_line(data = south_ref, aes(year, pct), colour = NAVY, linewidth = 0.6, linetype = "dashed") +
  geom_text(data = lab, aes(x = 2004.3, y = ymax * 0.985, label = txt), hjust = 0, vjust = 1, family = FONT, size = 2.6, lineheight = 0.9) +
  facet_grid(outcome ~ scope, scales = "free_y", switch = "y") +
  scale_size_area(max_size = 2.6, guide = "none") +
  scale_x_continuous(breaks = c(2005, 2015, 2025), labels = c("2005", "2015", "2025"), limits = c(2003.5, 2026)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.02))) +
  labs(x = "Year of survey", y = "Young women with violence, %") +
  theme_minimal(base_family = FONT, base_size = 9) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(), strip.placement = "outside",
        strip.text = element_text(face = "bold", size = 9), axis.text = element_text(size = 7.5), panel.spacing = unit(0.5, "lines"),
        axis.line = element_line(colour = "black", linewidth = 0.3), axis.ticks = element_line(colour = "black", linewidth = 0.3))
ragg::agg_png(file.path(FG, "fig_trends_lancet.png"), width = 7.2, height = 7.8, units = "in", res = 300, background = "white"); print(g); invisible(dev.off())
cat("fig_trends_lancet.png written\n")
