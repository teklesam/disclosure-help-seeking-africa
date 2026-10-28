# 61_fig1_map_forest_v3.R - Figure 1 (v3, co-author comments 31-33): where young survivors seek help.
#   (a) map: % of young women (15-24) who experienced violence and sought ANY help, by country (DHS, all rounds), number = rounds; VACS countries ringed
#   (b) small inset: UN regions used for pooling
#   (c) forest of country percentages with survey-design 95% CIs, grouped by region, with regional estimates (diamonds); point size = number of women
suppressMessages({library(data.table); library(ggplot2); library(sf); library(patchwork)})
sf_use_s2(FALSE)
ROOT <- path.expand("~/Documents/OMS DHS Births Climate Policy"); TB <- file.path(ROOT, "outputs/tables/P33_youth_v3")
FG <- file.path(ROOT, "outputs/figures/disclosure_v3"); dir.create(FG, showWarnings = FALSE)
OK <- c(West = "#0072B2", Central = "#009E73", East = "#E69F00", Southern = "#CC79A7")
ISO3 <- c(AO = "AGO", BF = "BFA", BJ = "BEN", BU = "BDI", CD = "COD", CI = "CIV", CM = "CMR", ET = "ETH", GA = "GAB", GH = "GHA", GM = "GMB", GN = "GIN", KE = "KEN", KM = "COM",
          LB = "LBR", LS = "LSO", MD = "MDG", ML = "MLI", MR = "MRT", MW = "MWI", MZ = "MOZ", NG = "NGA", NM = "NAM", RW = "RWA", SL = "SLE", SN = "SEN", TD = "TCD", TG = "TGO",
          TZ = "TZA", UG = "UGA", ZA = "ZAF", ZM = "ZMB", ZW = "ZWE", SZ = "SWZ")
NAMES <- c(AO = "Angola", BF = "Burkina Faso", BJ = "Benin", BU = "Burundi", CD = "DR Congo", CI = "Côte d'Ivoire", CM = "Cameroon", ET = "Ethiopia", GA = "Gabon", GH = "Ghana",
           GM = "Gambia", GN = "Guinea", KE = "Kenya", KM = "Comoros", LB = "Liberia", LS = "Lesotho", MD = "Madagascar", ML = "Mali", MR = "Mauritania", MW = "Malawi", MZ = "Mozambique",
           NG = "Nigeria", NM = "Namibia", RW = "Rwanda", SL = "Sierra Leone", SN = "Senegal", TD = "Chad", TG = "Togo", TZ = "Tanzania", UG = "Uganda", ZA = "South Africa", ZM = "Zambia",
           ZW = "Zimbabwe", SZ = "Eswatini")
CI <- fread(file.path(TB, "A8_5_dhs_cascade_ci.csv"))
cty <- CI[level == "country" & outcome == "sought_any"]; reg <- CI[level == "region" & outcome == "sought_any"]
rounds <- fread(file.path(ROOT, "outputs/tables/P33/A0_dhs_surveys_rounds.csv"))[survey != "SN_2019_8B", .(rounds = .N), by = country]
cty <- merge(cty, rounds, by = "country"); cty[, iso3 := ISO3[country]]
VACS <- c("KE", "LS", "MW", "MZ", "NM", "SZ", "TZ", "ZW")
af <- rnaturalearth::ne_countries(scale = 50, continent = "Africa", returnclass = "sf"); af$iso3 <- ifelse(af$iso_a3 == "-99", af$adm0_a3, af$iso_a3)
m <- merge(af, as.data.frame(cty), by = "iso3", all.x = TRUE)
cent <- suppressWarnings(st_point_on_surface(m[!is.na(m$rounds), ])); vac <- m[m$iso3 %in% ISO3[VACS], ]
pA <- ggplot(m) + geom_sf(aes(fill = pct), colour = "white", linewidth = 0.15) +
  geom_sf(data = vac, fill = NA, colour = "black", linewidth = 0.55) +
  geom_sf_text(data = cent, aes(label = rounds), size = 2.2) +
  scale_fill_gradient(low = "#E8F1F8", high = "#08519C", na.value = "#E3E3E3", limits = c(15, 50), breaks = c(20, 30, 40, 50),
                      name = "Young women (15–24) who sought any help\nafter violence (%), DHS, survey-weighted") +
  coord_sf(xlim = c(-18, 52), ylim = c(-35, 25)) + theme_void(base_size = 9) +
  theme(legend.position = "bottom", legend.title = element_text(size = 8), plot.tag = element_text(face = "bold", size = 13)) +
  guides(fill = guide_colourbar(barwidth = 11, barheight = 0.5, title.position = "top")) + labs(tag = "a")
m$regcat <- ifelse(is.na(m$region), NA, m$region)
pB <- ggplot(m) + geom_sf(aes(fill = regcat), colour = "white", linewidth = 0.1, show.legend = FALSE) +
  scale_fill_manual(values = OK, na.value = "#E3E3E3") + coord_sf(xlim = c(-18, 52), ylim = c(-35, 25)) +
  theme_void(base_size = 8) + theme(plot.tag = element_text(face = "bold", size = 11), panel.border = element_rect(fill = NA, colour = "grey70")) + labs(tag = "b")
ord <- c("West", "Central", "East", "Southern"); rows <- list(); y <- 0
for (r in rev(ord)) { y <- y + 1; rr <- reg[region == r]
  rows[[length(rows) + 1]] <- data.table(y = y, type = "region", region = r, lab = sprintf("%s Africa: %.1f%% (%.1f–%.1f)", r, rr$pct, rr$lo, rr$hi), pct = rr$pct, lo = rr$lo, hi = rr$hi, n = rr$n)
  cc <- cty[region == r][order(-pct)]
  for (i in seq_len(nrow(cc))) { y <- y + 1; rows[[length(rows) + 1]] <- data.table(y = y, type = "country", region = r, lab = sprintf("%s  %.0f%% (%.0f–%.0f)", NAMES[cc$country[i]], cc$pct[i], cc$lo[i], cc$hi[i]),
                                                                                 pct = cc$pct[i], lo = cc$lo[i], hi = cc$hi[i], n = cc$n[i]) }
  y <- y + 0.7 }
F <- rbindlist(rows); al <- CI[level == "all" & outcome == "sought_any"]
dia <- rbindlist(lapply(seq_len(nrow(F[type == "region"])), function(i) { r <- F[type == "region"][i]; data.table(x = c(r$lo, r$pct, r$hi, r$pct), y = c(r$y, r$y + 0.4, r$y, r$y - 0.4), id = i, region = r$region) }))
pC <- ggplot() + geom_vline(xintercept = al$pct, linetype = "dashed", colour = "grey50", linewidth = 0.4) +
  geom_segment(data = F[type == "country"], aes(x = lo, xend = hi, y = y, yend = y, colour = region), linewidth = 0.45) +
  geom_point(data = F[type == "country"], aes(x = pct, y = y, size = n, colour = region), shape = 15) +
  geom_polygon(data = dia, aes(x = x, y = y, group = id, fill = region)) +
  scale_colour_manual(values = OK, guide = "none") + scale_fill_manual(values = OK, guide = "none") + scale_size_area(max_size = 3.2, name = "Women (n)", breaks = c(500, 2000, 6000)) +
  scale_y_continuous(breaks = F$y, labels = F$lab, expand = expansion(add = 0.8)) + scale_x_continuous(limits = c(10, 60), breaks = seq(10, 60, 10)) +
  labs(x = sprintf("Sought any help after violence (%%), 95%% CI (survey design)\ndashed line = all countries, %.1f%% (%.1f–%.1f)", al$pct, al$lo, al$hi), y = NULL, tag = "c") +
  theme_minimal(base_size = 8.5) + theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(), axis.text.y = element_text(size = 6.6),
                                       legend.position = "bottom", plot.tag = element_text(face = "bold", size = 13))
left <- pA + inset_element(pB, left = 0.0, bottom = 0.05, right = 0.33, top = 0.42, align_to = "panel")
fig <- (left | pC) + plot_layout(widths = c(1.25, 1)) +
  plot_annotation(caption = paste0("DHS 2004–2025, 64 surveys, 32 countries; young women aged 15–24 who experienced physical or sexual violence (n = 49 313). (a) Numbers = DHS rounds; black outlines = VACS countries. (b) UN regions used for pooling: blue = West, green = Central, orange = East, pink = Southern Africa.\n",
                                   "(c) Squares = countries (size = number of women); diamonds = regional estimates with 95% CI, pooling women within region. Basemap: Natural Earth 1:50m."),
                  theme = theme(plot.caption = element_text(size = 7.2, hjust = 0)))
ggsave(file.path(FG, "fig1_map_help_seeking.png"), fig, width = 13, height = 8.2, dpi = 300, bg = "white")
cat("fig1 written\n")
