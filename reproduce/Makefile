# The article's tables and figures, regenerated with the lvc command line:
#
#     make -C reproduce all
#
# Paths in the configurations are relative to the working folder (the repository's root), so every command runs
# from there. The dataset and the ranking go to build/, the tables and figures to results/.
#
# The segmentation of the dataset's pages isn't part of `all`: it needs the segment extra, the models in models/ and
# the page images from their servers.
#
#     make -C reproduce segment-dataset
#
# The features, the geometry ranker and their tables and figures are part of `all` and are made from that
# segmentation (build/segment/dataset), which they make first if it's missing.

LVC ?= ../.venv/bin/lvc

TOP := $(abspath ..)
COMMAND := $(if $(findstring /,$(LVC)),$(abspath $(LVC)),$(LVC))
RUN := cd $(TOP) && $(COMMAND)
CONFIGS := reproduce/configs

# A contact for the servers the manifests and thumbnails come from, if .env gives one.
-include ../.env
export LVC_CONTACT

DATASET := $(TOP)/build/dataset/judgments.tsv
RANKING := $(TOP)/build/ranking/scores.tsv
REPORT := $(RUN) judgments report --config $(CONFIGS)/report.yaml --config $(CONFIGS)

.PHONY: all dataset ranking segment-dataset table-dataset-categories table-reliability figure-comparison-graph \
	figure-elo-violins features-dataset evaluate-geometry evaluate-geometry-layout table-geometry-by-elo-gap \
	table-geometry-feature-hierarchy figure-geometry-feature-correlations figure-geometry-features-on-page \
	figure-geometry-features-on-schematic-page figure-geometry-feature-families clean

all: table-dataset-categories table-reliability figure-comparison-graph figure-elo-violins table-geometry-by-elo-gap \
	table-geometry-feature-hierarchy figure-geometry-feature-correlations figure-geometry-features-on-page \
	figure-geometry-features-on-schematic-page figure-geometry-feature-families

# The dataset comes from configs/dataset.yaml's source: a folder in the published structure, or the Zenodo
# record's DOI. When it is a folder, a change to its files makes the dataset again.
SOURCE := $(shell sed -n 's/^source:[[:space:]]*//p' configs/dataset.yaml | tr -d "\"'")
dataset: $(DATASET)
$(DATASET): configs/dataset.yaml $(wildcard $(TOP)/$(SOURCE)/*.tsv $(TOP)/$(SOURCE)/README.md)
	$(RUN) judgments import --config $(CONFIGS)/dataset.yaml

ranking: $(RANKING)
$(RANKING): $(DATASET) configs/ranking.yaml
	$(RUN) judgments rank --config $(CONFIGS)/ranking.yaml

SEGMENT := $(TOP)/build/segment/dataset/pages.parquet
segment-dataset: $(SEGMENT)
$(SEGMENT): $(DATASET) configs/segment_dataset.yaml
	$(RUN) segment --config $(CONFIGS)/segment_dataset.yaml

table-dataset-categories: $(TOP)/results/tables/dataset_categories.tsv
$(TOP)/results/tables/dataset_categories.tsv: $(RANKING) configs/report.yaml configs/table_dataset_categories.yaml
	$(REPORT)/table_dataset_categories.yaml

table-reliability: $(TOP)/results/tables/reliability_by_elo_gap.tsv
$(TOP)/results/tables/reliability_by_elo_gap.tsv: $(RANKING) configs/report.yaml configs/table_reliability.yaml
	$(REPORT)/table_reliability.yaml

figure-comparison-graph: $(TOP)/results/figures/comparison_graph.svg
$(TOP)/results/figures/comparison_graph.svg: $(RANKING) configs/report.yaml configs/figure_comparison_graph.yaml
	$(REPORT)/figure_comparison_graph.yaml

figure-elo-violins: $(TOP)/results/figures/elo_score_distributions.svg
$(TOP)/results/figures/elo_score_distributions.svg: $(RANKING) configs/report.yaml configs/figure_elo_violins.yaml
	$(REPORT)/figure_elo_violins.yaml

# The features of the dataset's pages, the geometry ranker with and without the Layout features, and their tables
# and figures.
FEATURES := $(TOP)/build/features/dataset/features.parquet
features-dataset: $(FEATURES)
$(FEATURES): $(SEGMENT) configs/features_dataset.yaml
	$(RUN) features --config $(CONFIGS)/features_dataset.yaml

RELIABILITY := $(TOP)/results/tables/reliability_by_elo_gap.tsv
EVALUATE := $(RUN) evaluate --config $(CONFIGS)/evaluate.yaml --config $(CONFIGS)
GEOMETRY := $(TOP)/build/evaluate/geometry/model.json
GEOMETRY_LAYOUT := $(TOP)/build/evaluate/geometry_layout/model.json

evaluate-geometry: $(GEOMETRY)
$(GEOMETRY): $(FEATURES) $(RANKING) $(RELIABILITY) configs/evaluate.yaml configs/evaluate_geometry.yaml
	$(EVALUATE)/evaluate_geometry.yaml

evaluate-geometry-layout: $(GEOMETRY_LAYOUT)
$(GEOMETRY_LAYOUT): $(FEATURES) $(RANKING) $(RELIABILITY) configs/evaluate.yaml configs/evaluate_geometry_layout.yaml
	$(EVALUATE)/evaluate_geometry_layout.yaml

FEATURE_REPORT := $(RUN) report --config $(CONFIGS)/report_features.yaml --config $(CONFIGS)

table-geometry-by-elo-gap: $(TOP)/results/tables/geometry_by_elo_gap.tsv
$(TOP)/results/tables/geometry_by_elo_gap.tsv: $(GEOMETRY) $(GEOMETRY_LAYOUT) configs/report_features.yaml \
		configs/table_geometry_by_elo_gap.yaml
	$(FEATURE_REPORT)/table_geometry_by_elo_gap.yaml

table-geometry-feature-hierarchy: $(TOP)/results/tables/geometry_feature_hierarchy.tsv
$(TOP)/results/tables/geometry_feature_hierarchy.tsv: configs/report_features.yaml \
		configs/table_geometry_feature_hierarchy.yaml
	$(FEATURE_REPORT)/table_geometry_feature_hierarchy.yaml

figure-geometry-feature-correlations: $(TOP)/results/figures/geometry_feature_correlations.svg
$(TOP)/results/figures/geometry_feature_correlations.svg: $(FEATURES) $(RANKING) configs/report_features.yaml \
		configs/figure_geometry_feature_correlations.yaml
	$(FEATURE_REPORT)/figure_geometry_feature_correlations.yaml

figure-geometry-features-on-page: $(TOP)/results/figures/geometry_features_on_page.svg
$(TOP)/results/figures/geometry_features_on_page.svg: $(FEATURES) configs/report_features.yaml \
		configs/figure_geometry_features_on_page.yaml
	$(FEATURE_REPORT)/figure_geometry_features_on_page.yaml

figure-geometry-features-on-schematic-page: $(TOP)/results/figures/geometry_features_on_schematic_page.svg
$(TOP)/results/figures/geometry_features_on_schematic_page.svg: configs/report_features.yaml \
		configs/figure_geometry_features_on_schematic_page.yaml
	$(FEATURE_REPORT)/figure_geometry_features_on_schematic_page.yaml

figure-geometry-feature-families: $(TOP)/results/figures/geometry_feature_families.svg
$(TOP)/results/figures/geometry_feature_families.svg: configs/report_features.yaml \
		configs/figure_geometry_feature_families.yaml
	$(FEATURE_REPORT)/figure_geometry_feature_families.yaml

clean:
	rm -rf $(TOP)/build/dataset $(TOP)/build/ranking $(TOP)/build/features/dataset $(TOP)/build/evaluate
	rm -rf $(TOP)/results/tables $(TOP)/results/figures $(TOP)/results/runs
