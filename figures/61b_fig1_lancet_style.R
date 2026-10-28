# 61b_fig1_lancet_style.R - Figure 1 restyled to Lancet Global Health house style, following the forest plots in
# Li et al. Lancet Glob Health 2026; 14: 103889 (appendix figures 4, 6, 7): Times font; country column on the left;
# estimate (95% CI) and weight columns on the right; grey squares sized by weight with a black point marker;
# black CI lines; maroon dashed line at the overall estimate; hollow navy diamonds for subtotals and the overall estimate.
# Decimals use the Lancet mid-dot. No caption inside the figure (the LaTeX caption carries it).
# Same inputs and numbers as 61_fig1_map_forest_v3.R; only the presentation changes.
suppressMessages({library(data.table); library(ggplot2); library(sf); library(patchwork)})
sf_use_s2(FALSE)
ROOT <- path.expand("~/Documents/OMS DHS Births Climate Policy"); TB <- file.path(ROOT, "outputs/tables/P33_youth_v3")
FG <- file.path(ROOT, "outputs/figures/disclosure_v3")
FONT <- "Times New Roman"; NAVY <- "#1B2A6B"; MAROON <- "#9E2A3A"; SQ <- "#B5B5B5"
ISO3 <- c(AO = "AGO", BF = "BFA", BJ = "BEN", BU = "BDI", CD = "COD", CI = "CIV", CM = "CMR", ET = "ETH", GA = "GAB", GH = "GHA", GM = "GMB", GN = "GIN", KE = "KEN", KM = "COM",
          LB = "LBR", LS = "LSO", MD = "MDG", ML = "MLI", MR = "MRT", MW = "MWI", MZ = "MOZ", NG = "NGA", NM = "NAM", RW = "RWA", SL = "SLE", SN = "SEN", TD = "TCD", TG = "TGO",
          TZ = "TZA", UG = "UGA", ZA = "ZAF", ZM = "ZMB", ZW = "ZWE", SZ = "SWZ")
NAMES <- c(AO = "Angola", BF = "Burkina Faso", BJ = "Benin", BU = "Burundi", CD = "DR Congo", CI = "Côte d'Ivoire", CM = "Cameroon", ET = "Ethiopia", GA = "Gabon", GH = "Ghana",
           GM = "Gambia", GN = "Guinea", KE = "Kenya", KM = "Comoros", LB = "Liberia", LS = "Lesotho", MD = "Madagascar", ML = "Mali", MR = "Mauritania", MW = "Malawi", MZ = "Mozambique",
           NG = "Nigeria", NM = "Namibia", RW = "Rwanda", SL = "Sierra Leone", SN = "Senegal", TD = "Chad", TG = "Togo", TZ = "Tanzania", UG = "Uganda", ZA = "South Africa", ZM = "Zambia",
           ZW = "Zimbabwe", SZ = "Eswatini")
md <- function(x, d = 1) gsub(".", "·", formatC(x, format = "f", digits = d), fixed = TRUE)   # Lancet mid-dot decimals
est <- function(p, l, h) sprintf("%s (%s–%s)", md(p), md(l), md(h))
CI <- fread(file.path(TB, "A8_5_dhs_cascade_ci.csv"))
cty <- CI[level == "country" & outcome == "sought_any"]; reg <- CI[level == "region" & outcome == "sought_any"]; al <- CI[level == "all" & outcome == "sought_any"]
NT <- sum(cty$n); stopifnot(NT == al$n)
rounds <- fread(file.path(ROOT, "outputs/tables/P33/A0_dhs_surveys_rounds.csv"))[survey != "SN_2019_8B", .(rounds = .N), by = country]
cty <- merge(cty, rounds, by = "country"); cty[, `:=`(iso3 = ISO3[country], name = NAMES[country])]

# ---- (a) map and (b) region key ----
VACS <- c("KE", "LS", "MW", "MZ", "NM", "SZ", "TZ", "ZW")
af <- rnaturalearth::ne_countries(scale = 50, continent = "Africa", returnclass = "sf"); af$iso3 <- ifelse(af$iso_a3 == "-99", af$adm0_a3, af$iso_a3)
m <- merge(af, as.data.frame(cty), by = "iso3", all.x = TRUE)
cent <- suppressWarnings(st_point_on_surface(m[!is.na(m$rounds), ])); vac <- m[m$iso3 %in% ISO3[VACS], ]
pA <- ggplot(m) + geom_sf(aes(fill = pct), colour = "white", linewidth = 0.15) +
  geom_sf(data = vac, fill = NA, colour = "black", linewidth = 0.55) +
  geom_sf_text(data = cent, aes(label = rounds), size = 2.4, family = FONT) +
  scale_fill_gradient(low = "#EEF0F7", high = NAVY, na.value = "#E6E6E6", limits = c(15, 50), breaks = c(20, 30, 40, 50),
                      name = "Sought any\nhelp, %") +
  coord_sf(xlim = c(-18, 52), ylim = c(-35, 25)) + theme_void(base_size = 9, base_family = FONT) +
  theme(legend.position = "right", legend.title = element_text(size = 9), plot.tag = element_text(face = "bold", size = 13)) +
  guides(fill = guide_colourbar(barwidth = 0.6, barheight = 7, title.position = "top")) + labs(tag = "A", fill = "Sought any\nhelp, %")
GREY <- c(West = "#2E2E2E", Central = "#8F8F8F", East = "#CFCFCF", Southern = "#5E5E5E")
lab <- data.frame(x = c(-5, 21, 37, 22), y = c(13, -1, 1, -29), t = c("W", "C", "E", "S"), col = c("white", "white", "black", "white"))
pB <- ggplot(m) + geom_sf(aes(fill = region), colour = "white", linewidth = 0.1, show.legend = FALSE) +
  geom_text(data = lab, aes(x, y, label = t, colour = I(col)), family = FONT, fontface = "bold", size = 3) +
  scale_fill_manual(values = GREY, na.value = "#F4F4F4") + coord_sf(xlim = c(-18, 52), ylim = c(-35, 25)) +
  theme_void(base_size = 8, base_family = FONT) + theme(plot.tag = element_text(face = "bold", size = 11), panel.border = element_rect(fill = NA, colour = "grey60")) + labs(tag = "B", x = NULL, y = NULL)

# ---- (c) forest plot in the style of Li et al. 2026 ----
ord <- c("West", "Central", "East", "Southern"); rows <- list(); y <- 0
add <- function(...) { y <<- y - 1; rows[[length(rows) + 1]] <<- data.table(y = y, ...) }
for (r in ord) {
  add(type = "head", lab = paste(r, "Africa"), pct = NA, lo = NA, hi = NA, w = NA)
  cc <- cty[region == r][order(name)]
  for (i in seq_len(nrow(cc))) add(type = "country", lab = cc$name[i], pct = cc$pct[i], lo = cc$lo[i], hi = cc$hi[i], w = 100 * cc$n[i] / NT)
  rr <- reg[region == r]; add(type = "sub", lab = "Subtotal", pct = rr$pct, lo = rr$lo, hi = rr$hi, w = 100 * rr$n / NT)
  y <- y - 0.5 }
add(type = "all", lab = "Overall", pct = al$pct, lo = al$lo, hi = al$hi, w = 100)
F <- rbindlist(rows); stopifnot(abs(sum(F[type == "country", w]) - 100) < 1e-8)
F[!is.na(pct), `:=`(etxt = est(pct, lo, hi), wtxt = md(w, 2))]
XL <- -28; XE <- 64; XW <- 101; top <- 1.2                                # x positions of the text columns
dia <- rbindlist(lapply(which(F$type %in% c("sub", "all")), function(i) { r <- F[i]
  data.table(x = c(r$lo, r$pct, r$hi, r$pct), y = c(r$y, r$y + 0.42, r$y, r$y - 0.42), id = i) }))
pC <- ggplot() +
  annotate("segment", x = XL, xend = XW + 2, y = top - 0.7, yend = top - 0.7, linetype = "longdash", colour = "grey70", linewidth = 0.35) +
  annotate("segment", x = al$pct, xend = al$pct, y = min(F$y) - 0.5, yend = top - 0.7, linetype = "dashed", colour = MAROON, linewidth = 0.45) +
  geom_point(data = F[type == "country"], aes(x = pct, y = y, size = w), shape = 15, colour = SQ) +
  geom_segment(data = F[type == "country"], aes(x = lo, xend = hi, y = y, yend = y), colour = "black", linewidth = 0.5) +
  geom_point(data = F[type == "country"], aes(x = pct, y = y), shape = 18, size = 1.6, colour = "black") +
  geom_polygon(data = dia, aes(x = x, y = y, group = id), fill = "white", colour = NAVY, linewidth = 0.6) +
  geom_text(data = F, aes(x = XL, y = y, label = lab, fontface = ifelse(type %in% c("head", "all"), "bold", "plain")), hjust = 0, family = FONT, size = 3) +
  geom_text(data = F[!is.na(pct)], aes(x = XE, y = y, label = etxt), hjust = 0, family = FONT, size = 3) +
  geom_text(data = F[!is.na(pct)], aes(x = XW, y = y, label = wtxt), hjust = 1, family = FONT, size = 3) +
  annotate("text", x = XL, y = top, label = "Country", hjust = 0, vjust = 0.3, family = FONT, size = 3.2) +
  annotate("text", x = XE, y = top, label = "Sought any help,\n% (95% CI)", hjust = 0, vjust = 0.3, family = FONT, size = 3.2, lineheight = 0.9) +
  annotate("text", x = XW, y = top, label = "Women, %", hjust = 1, vjust = 0.3, family = FONT, size = 3.2) +
  annotate("segment", x = 10, xend = 60, y = min(F$y) - 0.9, yend = min(F$y) - 0.9, colour = "black", linewidth = 0.5) +
  scale_size_area(max_size = 6, guide = "none") +
  scale_x_continuous(breaks = seq(10, 60, 10), labels = seq(10, 60, 10), expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  coord_cartesian(xlim = c(XL, XW + 2), ylim = c(min(F$y) - 0.9, top + 1.2), clip = "off") +
  labs(x = "Sought any help after violence, % (95% CI)", y = NULL, tag = "C") +
  theme_void(base_size = 9, base_family = FONT) +
  theme(axis.text.x = element_text(size = 9, margin = margin(t = 2)), axis.ticks.x = element_line(colour = "black", linewidth = 0.4),
        axis.ticks.length.x = unit(2.5, "pt"), axis.title.x = element_text(size = 9.5, margin = margin(t = 5)),
        plot.tag = element_text(face = "bold", size = 13), plot.margin = margin(4, 8, 4, 4))
fig <- ((pA | pB) + plot_layout(widths = c(2.1, 1))) / pC + plot_layout(heights = c(1, 2.35))
ragg::agg_png(file.path(FG, "fig1_map_help_seeking_lancet.png"), width = 8, height = 10.6, units = "in", res = 300, background = "white")
print(fig); invisible(dev.off())
fwrite(F, file.path(TB, "fig1_forest_rows.csv"))
cat("fig1 (Lancet style) written; women =", NT, "; overall =", est(al$pct, al$lo, al$hi), "\n")
