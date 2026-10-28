# 61d_fig_cascade_forest_lancet.R - Disclosure cascade by country (told anyone, sought any help, sought formal help), three forests
# sharing one country column, in the style of Figure 1C and Li et al. Lancet Glob Health 2026; 14: 103889.
# Input: A8_5_dhs_cascade_ci.csv (59_A8_cascade_ci.R): survey-weighted percentages with design-based logit 95% CIs (equation 1);
# regional and overall rows pool women within region with survey weights. Each step has its own denominator: telling anyone is
# missing where a survey's question did not distinguish it from seeking help, and formal help where the source items were absent.
suppressMessages({library(data.table); library(ggplot2)})
ROOT <- path.expand("~/Documents/OMS DHS Births Climate Policy"); TB <- file.path(ROOT, "outputs/tables/P33_youth_v3")
FG <- file.path(ROOT, "outputs/figures/disclosure_v3")
FONT <- "Times New Roman"; NAVY <- "#1B2A6B"; MAROON <- "#9E2A3A"; SQ <- "#B5B5B5"
NAMES <- c(AO = "Angola", BF = "Burkina Faso", BJ = "Benin", BU = "Burundi", CD = "DR Congo", CI = "Côte d'Ivoire", CM = "Cameroon", ET = "Ethiopia", GA = "Gabon", GH = "Ghana",
           GM = "Gambia", GN = "Guinea", KE = "Kenya", KM = "Comoros", LB = "Liberia", LS = "Lesotho", MD = "Madagascar", ML = "Mali", MR = "Mauritania", MW = "Malawi", MZ = "Mozambique",
           NG = "Nigeria", NM = "Namibia", RW = "Rwanda", SL = "Sierra Leone", SN = "Senegal", TD = "Chad", TG = "Togo", TZ = "Tanzania", UG = "Uganda", ZA = "South Africa", ZM = "Zambia",
           ZW = "Zimbabwe", SZ = "Eswatini")
md <- function(x, d = 1) gsub(".", "·", formatC(x, format = "f", digits = d), fixed = TRUE)
est <- function(p, l, h) sprintf("%s (%s–%s)", md(p), md(l), md(h))
CI <- fread(file.path(TB, "A8_5_dhs_cascade_ci.csv"))
OUT <- c("told", "sought_any", "formal"); TITLE <- c("A  Told anyone", "B  Sought any help", "C  Sought formal help")
AX <- list(c(20, 80, 10), c(10, 60, 10), c(0, 16, 4))                 # min, max, tick step of each forest (%)
cty <- unique(CI[level == "country", .(country, region)]); cty[, name := NAMES[country]]
rows <- list(); y <- 0
add <- function(...) { y <<- y - 1; rows[[length(rows) + 1]] <<- data.table(y = y, ...) }
for (r in c("West", "Central", "East", "Southern")) {
  add(type = "head", lab = paste(r, "Africa"), kk = NA_character_)
  for (k in cty[region == r][order(name), country]) add(type = "country", lab = NAMES[[k]], kk = k)
  add(type = "sub", lab = "Subtotal", kk = r); y <- y - 0.5 }
add(type = "all", lab = "Overall", kk = "all")
F <- rbindlist(rows)
X0 <- c(15, 46.5, 78); XW <- 13.5; XT <- X0 + XW + 0.6                 # x layout (0-100): country | forest + estimate x 3
pos <- function(v, j) X0[j] + XW * (pmin(pmax(v, AX[[j]][1]), AX[[j]][2]) - AX[[j]][1]) / (AX[[j]][2] - AX[[j]][1])
V <- rbindlist(lapply(1:3, function(j) { o <- OUT[j]; v <- copy(F)
  v[type == "country", c("p", "l", "h", "n") := CI[level == "country" & outcome == o][match(kk, country), .(pct, lo, hi, n)]]
  v[type == "sub", c("p", "l", "h", "n") := CI[level == "region" & outcome == o][match(kk, region), .(pct, lo, hi, n)]]
  v[type == "all", c("p", "l", "h", "n") := CI[level == "all" & outcome == o, .(pct, lo, hi, n)]]
  v[, `:=`(j = j, outcome = o, xo = pos(p, j), xl = pos(l, j), xh = pos(h, j), cut_h = h > AX[[j]][2], etxt = est(p, l, h))]; v }))
V[type == "head", etxt := NA_character_]; stopifnot(!anyNA(V[type != "head", p]))
fwrite(V[, .(outcome, type, lab, kk, n, pct = p, lo = l, hi = h)], file.path(TB, "fig_cascade_forest_rows.csv"))
ALL <- V[type == "all"]; cty_ <- V[type == "country"]
dia <- V[type %in% c("sub", "all"), .(x = c(xl, xo, xh, xo), y = c(y, y + 0.42, y, y - 0.42)), by = .(j, kk)]
ybot <- min(F$y) - 0.9; top <- 1.3
TK <- rbindlist(lapply(1:3, function(j) data.table(j = j, v = seq(AX[[j]][1], AX[[j]][2], AX[[j]][3]))))[, x := pos(v, j), by = j]
g <- ggplot() +
  annotate("segment", x = 0, xend = 100, y = top - 0.75, yend = top - 0.75, linetype = "longdash", colour = "grey70", linewidth = 0.35) +
  geom_segment(data = ALL, aes(x = xo, xend = xo, y = ybot, yend = top - 0.75), colour = MAROON, linetype = "dashed", linewidth = 0.45) +
  geom_point(data = cty_, aes(x = xo, y = y, size = n), shape = 15, colour = SQ) +
  geom_segment(data = cty_, aes(x = xl, xend = xh, y = y, yend = y), colour = "black", linewidth = 0.42) +
  geom_segment(data = cty_[cut_h == TRUE], aes(x = xh - 0.01, xend = xh, y = y, yend = y), colour = "black", linewidth = 0.42, arrow = arrow(length = unit(1.5, "pt"), type = "closed")) +
  geom_point(data = cty_, aes(x = xo, y = y), shape = 18, size = 1.3, colour = "black") +
  geom_polygon(data = dia, aes(x = x, y = y, group = interaction(j, kk)), fill = "white", colour = NAVY, linewidth = 0.5) +
  geom_text(data = F, aes(x = 0, y = y, label = lab, fontface = ifelse(type %in% c("head", "all"), "bold", "plain")), hjust = 0, family = FONT, size = 2.6) +
  geom_text(data = V[type != "head"], aes(x = XT[j], y = y, label = etxt), hjust = 0, family = FONT, size = 2.45) +
  annotate("text", x = 0, y = top, label = "Country", hjust = 0, vjust = 0.2, family = FONT, size = 2.9) +
  annotate("text", x = X0 + (XW + 9) / 2, y = top + 1.25, label = TITLE, family = FONT, fontface = "bold", size = 3) +
  annotate("text", x = XT, y = top, label = "% (95% CI)", hjust = 0, vjust = 0.2, family = FONT, size = 2.9) +
  annotate("segment", x = X0, xend = X0 + XW, y = ybot, yend = ybot, linewidth = 0.45) +
  geom_segment(data = TK, aes(x = x, xend = x, y = ybot, yend = ybot - 0.3), linewidth = 0.4) +
  geom_text(data = TK, aes(x = x, y = ybot - 0.95, label = v), family = FONT, size = 2.4) +
  annotate("text", x = X0 + XW / 2, y = ybot - 2.1, label = "% of women (95% CI)", family = FONT, size = 2.8) +
  scale_size_area(max_size = 3.6, guide = "none") +
  coord_cartesian(xlim = c(0, 100), ylim = c(ybot - 2.6, top + 1.9), expand = FALSE, clip = "off") +
  theme_void(base_family = FONT) + theme(plot.margin = margin(4, 6, 4, 4))
ragg::agg_png(file.path(FG, "fig_cascade_forest_lancet.png"), width = 8.6, height = 0.19 * (top + 1.9 - ybot + 2.6) + 0.3, units = "in", res = 300, background = "white")
print(g); invisible(dev.off())
cat("cascade forest written:", paste(OUT, ALL$etxt, collapse = " | "), "\n")
