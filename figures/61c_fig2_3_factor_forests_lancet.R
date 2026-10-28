# 61c_fig2_3_factor_forests_lancet.R - Figures 2 and 3: country-level forest plots of the four main factors associated with
# seeking any help, in the style of Li et al. Lancet Glob Health 2026; 14: 103889 (appendix figures 4, 6, 7).
#   Figure 2: injury and severe physical violence (partner-violence model, women with partner violence, 21 countries)
#   Figure 3: physical and sexual violence and formerly in union (main model, 26 countries)
# Rows: each country's survey-weighted logit OR with PSU-clustered 95% CI (A2, equations 2-4); regional subtotals and the overall
# estimate are posterior medians with 95% CrI from the hierarchical model (A3, equation 5; same numbers as Table 1);
# Southern Africa is the exploratory pooled model of Lesotho, Namibia and South Africa (95% CI), not part of the overall estimate.
suppressMessages({library(data.table); library(ggplot2)})
ROOT <- path.expand("~/Documents/OMS DHS Births Climate Policy"); TB <- file.path(ROOT, "outputs/tables/P33_youth_v3")
FG <- file.path(ROOT, "outputs/figures/disclosure_v3")
FONT <- "Times New Roman"; NAVY <- "#1B2A6B"; MAROON <- "#9E2A3A"; SQ <- "#B5B5B5"; GREYD <- "#7A7A7A"
NAMES <- c(AO = "Angola", BF = "Burkina Faso", BJ = "Benin", BU = "Burundi", CD = "DR Congo", CI = "Côte d'Ivoire", CM = "Cameroon", ET = "Ethiopia", GA = "Gabon", GH = "Ghana",
           GM = "Gambia", GN = "Guinea", KE = "Kenya", KM = "Comoros", LB = "Liberia", LS = "Lesotho", MD = "Madagascar", ML = "Mali", MR = "Mauritania", MW = "Malawi", MZ = "Mozambique",
           NG = "Nigeria", NM = "Namibia", RW = "Rwanda", SL = "Sierra Leone", SN = "Senegal", TD = "Chad", TG = "Togo", TZ = "Tanzania", UG = "Uganda", ZA = "South Africa", ZM = "Zambia",
           ZW = "Zimbabwe", SZ = "Eswatini")
md <- function(x, d = 2) gsub(".", "·", formatC(x, format = "f", digits = d), fixed = TRUE)
est <- function(o, l, h) sprintf("%s (%s–%s)", md(o), md(l), md(h))
CE <- fread(file.path(TB, "A2_country_estimates.csv"))[source == "DHS" & outcome == "sought_any"]
PO <- fread(file.path(TB, "A3_pooled_numpyro.csv"))[source == "DHS" & outcome == "sought_any"]
SA <- fread(file.path(TB, "southern_africa_pooled_exploratory.csv"))
LMIN <- 0.125; LMAX <- 16; TICKS <- c(0.125, 0.25, 0.5, 1, 2, 4, 8, 16)

make <- function(mdl, terms, titles, file) {
  cc <- CE[model == mdl & term == terms[1], .(country, region, n)]
  for (t in terms) stopifnot(setequal(CE[model == mdl & term == t, country], cc$country))
  cc[, name := NAMES[country]]
  rows <- list(); y <- 0
  add <- function(...) { y <<- y - 1; rows[[length(rows) + 1]] <<- data.table(y = y, ...) }
  for (r in c("West", "Central", "East")) {
    add(type = "head", lab = paste(r, "Africa"), kk = NA_character_, n = NA_real_)
    for (k in cc[region == r][order(name), country]) add(type = "country", lab = NAMES[[k]], kk = k, n = cc[country == k, n])
    add(type = "sub", lab = "Subtotal", kk = r, n = cc[region == r, sum(n)]); y <- y - 0.5 }
  add(type = "all", lab = "Overall, West, Central and East", kk = "all", n = sum(cc$n)); y <- y - 0.5
  add(type = "head", lab = "Southern Africa", kk = NA_character_, n = NA_real_)
  add(type = "sa", lab = "Lesotho, Namibia, South Africa", kk = "SA", n = SA[model == mdl & term == terms[1], n])
  F <- rbindlist(rows)
  # x layout (arbitrary units): country | women | forest 1 | estimate 1 | forest 2 | estimate 2
  X0 <- c(27.5, 66); XW <- 19; XT <- X0 + XW + 1.2
  pos <- function(or, j) X0[j] + XW * (log(pmin(pmax(or, LMIN), LMAX)) - log(LMIN)) / (log(LMAX) - log(LMIN))
  P <- list(); D <- list(); TXT <- list(); REF <- list()
  for (j in 1:2) { t <- terms[j]; po <- PO[model == mdl & term == t]; sa <- SA[model == mdl & term == t]
    v <- copy(F)
    v[type == "country", c("o", "l", "h") := { e <- CE[model == mdl & term == t][match(kk, country)]; list(exp(e$est), exp(e$est - 1.96 * e$se), exp(e$est + 1.96 * e$se)) }]
    for (r in c("West", "Central", "East")) v[type == "sub" & kk == r, c("o", "l", "h") := list(po[[paste0("OR_", r)]], po[[paste0("OR_", r, "_lo")]], po[[paste0("OR_", r, "_hi")]])]
    v[type == "all", c("o", "l", "h") := list(po$OR, po$OR_lo, po$OR_hi)]
    v[type == "sa", c("o", "l", "h") := list(sa$OR, sa$lo, sa$hi)]
    v[, `:=`(j = j, term = t, xo = pos(o, j), xl = pos(l, j), xh = pos(h, j), cut_l = l < LMIN, cut_h = h > LMAX)]
    v[!is.na(o), etxt := est(o, l, h)]
    P[[j]] <- v; REF[[j]] <- data.table(j = j, x1 = pos(1, j), xo = pos(po$OR, j), tau = po$tau) }
  V <- rbindlist(P); R <- rbindlist(REF)
  fwrite(V[, .(term, type, lab, kk, n, OR = o, lo = l, hi = h)], file.path(TB, sub(".png", "_rows.csv", basename(file))))
  dia <- V[type %in% c("sub", "all", "sa"), .(x = c(xl, xo, xh, xo), y = c(y, y + 0.42, y, y - 0.42), col = ifelse(type == "sa", GREYD, NAVY)), by = .(j, type, kk)]
  ybot <- min(F$y) - 0.9; top <- 1.3; cty <- V[type == "country"]
  g <- ggplot() +
    annotate("segment", x = 0, xend = 100, y = top - 0.75, yend = top - 0.75, linetype = "longdash", colour = "grey70", linewidth = 0.35) +
    geom_segment(data = R, aes(x = x1, xend = x1, y = ybot, yend = top - 0.75), colour = "black", linewidth = 0.45) +
    geom_segment(data = R, aes(x = xo, xend = xo, y = ybot, yend = top - 0.75), colour = MAROON, linetype = "dashed", linewidth = 0.45) +
    geom_point(data = cty, aes(x = xo, y = y, size = n), shape = 15, colour = SQ) +
    geom_segment(data = cty, aes(x = xl, xend = xh, y = y, yend = y), colour = "black", linewidth = 0.45) +
    geom_segment(data = cty[cut_h == TRUE], aes(x = xh - 0.01, xend = xh, y = y, yend = y), colour = "black", linewidth = 0.45, arrow = arrow(length = unit(1.6, "pt"), type = "closed")) +
    geom_segment(data = cty[cut_l == TRUE], aes(x = xl + 0.01, xend = xl, y = y, yend = y), colour = "black", linewidth = 0.45, arrow = arrow(length = unit(1.6, "pt"), type = "closed")) +
    geom_point(data = cty, aes(x = xo, y = y), shape = 18, size = 1.5, colour = "black") +
    geom_polygon(data = dia, aes(x = x, y = y, group = interaction(j, kk), colour = I(col)), fill = "white", linewidth = 0.55) +
    geom_text(data = F, aes(x = 0, y = y, label = lab, fontface = ifelse(type %in% c("head", "all"), "bold", "plain")), hjust = 0, family = FONT, size = 2.75) +
    geom_text(data = F[!is.na(n)], aes(x = 25.5, y = y, label = formatC(n, format = "d", big.mark = " ")), hjust = 1, family = FONT, size = 2.75) +
    geom_text(data = V[!is.na(o)], aes(x = XT[j], y = y, label = etxt), hjust = 0, family = FONT, size = 2.75) +
    annotate("text", x = 0, y = top, label = "Country", hjust = 0, vjust = 0.2, family = FONT, size = 3) +
    annotate("text", x = 25.5, y = top, label = "Women, n", hjust = 1, vjust = 0.2, family = FONT, size = 3) +
    annotate("text", x = X0 + XW / 2, y = top + 1.25, label = titles, family = FONT, fontface = "bold", size = 3.1) +
    annotate("text", x = XT, y = top, label = "OR (95% CI)", hjust = 0, vjust = 0.2, family = FONT, size = 3) +
    annotate("segment", x = X0, xend = X0 + XW, y = ybot, yend = ybot, linewidth = 0.45) +
    annotate("segment", x = rep(sapply(1:2, function(j) pos(TICKS, j)), 1), xend = rep(sapply(1:2, function(j) pos(TICKS, j)), 1),
             y = ybot, yend = ybot - 0.3, linewidth = 0.4) +
    annotate("text", x = c(sapply(1:2, function(j) pos(TICKS, j))), y = ybot - 0.95, label = rep(c("", "0·25", "0·5", "1", "2", "4", "8", "16"), 2),
             family = FONT, size = 2.4) +
    annotate("text", x = X0 + XW / 2, y = ybot - 2.1, label = "Odds ratio (log scale)", family = FONT, size = 2.9) +
    annotate("text", x = R$x1 - 0.6, y = ybot - 3.1, label = "Less help-seeking", hjust = 1, family = FONT, size = 2.4, colour = "grey30") +
    annotate("text", x = R$x1 + 0.6, y = ybot - 3.1, label = "More help-seeking", hjust = 0, family = FONT, size = 2.4, colour = "grey30") +
    scale_size_area(max_size = 4.2, guide = "none") +
    coord_cartesian(xlim = c(0, 100), ylim = c(ybot - 3.6, top + 1.9), expand = FALSE, clip = "off") +
    theme_void(base_family = FONT) + theme(plot.margin = margin(4, 6, 4, 4))
  ragg::agg_png(file, width = 8, height = 0.205 * (top + 1.9 - ybot + 3.6) + 0.3, units = "in", res = 300, background = "white"); print(g); invisible(dev.off())
  cat(basename(file), ": countries", nrow(cc), "| tau", paste(round(R$tau, 3), collapse = ", "), "\n")
  invisible(V) }
make("partner", c("injury", "severe"), c("A  Injury from partner violence", "B  Severe physical violence"), file.path(FG, "fig2_forest_injury_severe_lancet.png"))
make("main", c("vtypeboth", "maritalformer"), c("A  Physical and sexual violence", "B  Formerly in union"), file.path(FG, "fig3_forest_both_former_lancet.png"))
