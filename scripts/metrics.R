
library(dplyr)
library(ggplot2)

tday <- lubridate::today()

metrics <- readr::read_csv(
  "external/results/metrics/combined_metrics_20251009_221229.csv"
) %>%
  distinct() %>%
  select(-train_checkpoint) %>%
  select(source_file, pixel_level_iou:object_level_recall) %>%
  rename(experiment = source_file) %>%
  arrange(experiment) %>%
  filter(grepl("mappingafrica-catalog.csv|ftw-catalog2.csv", experiment)) %>% #View()
  mutate(
    model = case_when(
      grepl("ma-approximate", experiment) ~ "MA approx baseline",
      grepl("ftwbaseline", experiment) ~ "FTW Baseline",
      TRUE ~ experiment
    )
  ) %>%
  mutate(
    train = case_when(
      grepl("^ma-", experiment) ~ "MA",
      grepl("^ftw-|^ftwbaseline", experiment) ~ "FTW",
      grepl("^fullcat", experiment) ~ "Combined",
      TRUE ~ experiment
    )
  ) %>%
  mutate(
    validate = case_when(
      grepl("-mappingafrica-catalog", experiment) ~ "MA",
      grepl("-ftw-catalog", experiment) ~ "FTW",
      TRUE ~ experiment
    )
  ) %>%
  mutate(
    settings = case_when(
      grepl("tversky-exp", experiment) ~ "Tversky focal loss",
      grepl("ftwbaseline-localtversky-minmax_gab-exp8", experiment) ~
        "Tversky, Min-max global",
      grepl("minmax_gab", experiment) ~ "Min-max global",
      grepl("minmax_lab-exp3-", experiment) ~ "Min-max local",
      grepl("exp3a-", experiment) ~ "Min-max local, 1% clip",
      grepl("zvalue", experiment) ~ "Z-value, global per band",
      grepl("photometric", experiment) ~ "Combined photometric augs",
      grepl("rescale", experiment) ~ "Random resize crop",
      grepl("satslide", experiment) ~ "SatSlideMix",
      grepl("fullcat-ftwbaseline-exp2", experiment) ~ "Min-max local",
      grepl("fullcat-ftwbaseline-exp3", experiment) ~ "Tversky focal loss",
      grepl("fullcat-ftwbaseline-exp4", experiment) ~ "Tversky, Min-max local",
      grepl("ma-ftwbaseline-exp2", experiment) ~ "Min-max local",
      TRUE ~ "Default"
    )
  ) %>%
  select(experiment, model, train, validate, settings,
         pixel_level_iou:object_level_recall) # %>%
  # View()

metrics %>% readr::write_csv(
  glue::glue("external/results/summary_metrics_{tday}.csv")
)


# metrics %>% View()
# metrics$settings
metrics$settings <- factor(
  metrics$settings,
  levels = c("Default", "Z-value, global per band", "Min-max local",
             "Min-max local, 1% clip", "Min-max global",
             "SatSlideMix", "Random resize crop",
             "Combined photometric augs", "Tversky focal loss",
             "Tversky, Min-max local", "Tversky, Min-max global")
)

p <- metrics %>%
  # select(-experiment) %>%
  tidyr::pivot_longer(
    cols = pixel_level_iou:object_level_recall,
    names_to = "metric",
    values_to = "value"
  ) %>%
  filter(model == "FTW Baseline" & validate == "FTW" & train == "FTW") %>%
  # View()
  ggplot() +
  geom_bar(aes(x = settings, y = value, fill = settings),
           stat = "identity", position = position_dodge()) +
  geom_text(aes(x = settings, y = value * 0.75, label = round(value, 3)),
            angle = 90) +
  xlab("") +
  scale_fill_manual(
    values = RColorBrewer::brewer.pal(
      length(unique(metrics$settings)), "Spectral"
    )
  ) +
  facet_wrap(~metric, scales = "free_y") +
  theme_linedraw() +
  # theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
  theme(axis.text.x = element_blank(),
        strip.background = element_rect(fill = "lightgrey"),
        strip.text = element_text(color = "black")) +
  ggtitle("FTW Baseline with different settings")

ggsave("external/results/figures/ftwbaseline-tests.png", p, width = 10,
       height = 6, dpi = 300, units = "in")

# metrics <- readr::read_csv(
#   glue::glue("external/results/summary_metrics_{tday}.csv")
# )

p2 <- metrics %>%
  select(-experiment) %>%
  mutate(modelvariant = paste0(model, ", ", settings)) %>%
  tidyr::pivot_longer(
    cols = pixel_level_iou:object_level_recall,
    names_to = "metric",
    values_to = "value"
  ) %>%
  filter(train == "Combined") %>% #View()
  ggplot() +
  geom_bar(
    aes(x = validate, y = value, fill = modelvariant),
    stat = "identity",
    position = position_dodge(width = 0.9)
  ) +
  geom_text(
    aes(x = validate, y = value * 0.75, label = round(value, 3),
        group = modelvariant),    # <- group by variant
    angle = 90,
    position = position_dodge(width = 0.9), # <- match dodging to bars
    size = 3
  ) +
  xlab("Validation set") +
  scale_fill_manual(values = RColorBrewer::brewer.pal(5, "Spectral")) +
  facet_wrap(~metric, scales = "free_y") +
  theme_linedraw() +
  theme(#axis.text.x = element_blank(),
        strip.background = element_rect(fill = "lightgrey"),
        strip.text = element_text(color = "black")) +
  ggtitle("FTW and MA models trained on full catalog")

ggsave("external/results/figures/model-comparisons.png", p2, width = 10,
       height = 6, dpi = 300, units = "in")

p3 <- metrics %>%
  # select(-experiment) %>%
  mutate(modelvariant = paste0(model, ", ", settings)) %>%
  tidyr::pivot_longer(
    cols = pixel_level_iou:object_level_recall,
    names_to = "metric",
    values_to = "value"
  ) %>%
  filter(train == "MA") %>%
  # View()
  ggplot() +
  geom_bar(
    aes(x = validate, y = value, fill = modelvariant),
    stat = "identity",
    position = position_dodge(width = 0.9)
  ) +
  geom_text(
    aes(x = validate, y = value * 0.75, label = round(value, 3),
        group = modelvariant),  # <- tell ggplot to dodge by model
    angle = 90,
    position = position_dodge(width = 0.9),
    size = 3
  ) +
  xlab("Validation set") +
  scale_fill_manual(values = RColorBrewer::brewer.pal(3, "Spectral")) +
  facet_wrap(~metric, scales = "free_y") +
  theme_linedraw() +
  theme(
    strip.background = element_rect(fill = "lightgrey"),
    strip.text = element_text(color = "black")
  ) +
  ggtitle("FTW and MA models trained on MA catalog")

ggsave("external/results/figures/model-comparisons-ma.png", p3, width = 10,
       height = 6, dpi = 300, units = "in")

p4 <- metrics %>%
  # select(-experiment) %>%
  mutate(modelvariant = paste0(model, ", ", settings)) %>%
  tidyr::pivot_longer(
    cols = pixel_level_iou:object_level_recall,
    names_to = "metric",
    values_to = "value"
  ) %>% #View()
  filter(train == "FTW" &
           modelvariant %in% c("MA approx baseline, Default",
                              "FTW Baseline, Default",
                              "FTW Baseline, Tversky focal loss")) %>%
  ggplot() +
  geom_bar(
    aes(x = validate, y = value, fill = modelvariant),
    stat = "identity",
    position = position_dodge(width = 0.9)
  ) +
  geom_text(
    aes(x = validate, y = value * 0.75, label = round(value, 3),
        group = modelvariant),  # <- tell ggplot to dodge by model
    angle = 90,
    position = position_dodge(width = 0.9),
    size = 3
  ) +
  xlab("Validation set") +
  scale_fill_manual(values = RColorBrewer::brewer.pal(3, "Spectral")) +
  facet_wrap(~metric, scales = "free_y") +
  theme_linedraw() +
  theme(
    strip.background = element_rect(fill = "lightgrey"),
    strip.text = element_text(color = "black")
  ) +
  ggtitle("FTW and MA models trained on FTW catalog")

ggsave("external/results/figures/model-comparisons-ftw.png", p4, width = 10,
       height = 6, dpi = 300, units = "in")
