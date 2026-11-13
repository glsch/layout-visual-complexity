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
#
# The ViT ranker needs the vit extra; its tokens (embed-dataset) come from DINOv2 ViT-S/14, downloaded once into the
# Hugging Face cache, and are kept in the user cache. Its evaluation with its learning curves takes about 25 minutes
# on an Apple M4 Pro. The head that comes with the package is trained on every judgment (head-dataset) and copied into
# the package by `make -C reproduce package-head`.
#
# The baselines' features are measured with the others, the generic image measures with the patch codebook learned
# on the dataset's pages (codebook-dataset), which `make -C reproduce package-codebook` copies into the package. The
# linear model on the mean token needs the ViT's tokens.

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
	figure-geometry-features-on-schematic-page figure-geometry-feature-families embed-dataset evaluate-vit \
	evaluate-ensembles head-dataset package-head table-methods-by-elo-gap figure-accuracy-by-elo-gap \
	table-paired-differences table-ensembles-by-elo-gap figure-learning-curves-and-calibration \
	figure-local-complexity-maps codebook-dataset package-codebook features-mean-token evaluate-baselines \
	evaluate-measures table-baselines-by-elo-gap table-baseline-measures table-baseline-differences \
	figure-baselines-by-information figure-baselines-by-information-each-measure \
	figure-generic-measure-correlations clean

all: table-dataset-categories table-reliability figure-comparison-graph figure-elo-violins table-geometry-by-elo-gap \
	table-geometry-feature-hierarchy figure-geometry-feature-correlations figure-geometry-features-on-page \
	figure-geometry-features-on-schematic-page figure-geometry-feature-families table-methods-by-elo-gap \
	figure-accuracy-by-elo-gap table-paired-differences table-ensembles-by-elo-gap \
	figure-learning-curves-and-calibration figure-local-complexity-maps table-baselines-by-elo-gap \
	table-baseline-measures table-baseline-differences figure-baselines-by-information \
	figure-baselines-by-information-each-measure figure-generic-measure-correlations

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
CODEBOOK := $(TOP)/build/codebook/dataset/codebook.npz
codebook-dataset: $(CODEBOOK)
$(CODEBOOK): $(SEGMENT) configs/codebook_dataset.yaml
	$(RUN) codebook --config $(CONFIGS)/codebook_dataset.yaml

package-codebook: $(CODEBOOK)
	mkdir -p $(TOP)/src/layout_visual_complexity/codebooks
	cp $(CODEBOOK) $(TOP)/src/layout_visual_complexity/codebooks/patch_colours.npz

FEATURES := $(TOP)/build/features/dataset/features.parquet
features-dataset: $(FEATURES)
$(FEATURES): $(SEGMENT) $(CODEBOOK) configs/features_dataset.yaml
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

# The ViT ranker, the ensembles, and the methods' tables and figures.
TOKENS := $(TOP)/build/embed/dataset/tokens.json
embed-dataset: $(TOKENS)
$(TOKENS): $(SEGMENT) configs/embed_dataset.yaml
	$(RUN) embed --config $(CONFIGS)/embed_dataset.yaml

VIT := $(TOP)/build/evaluate/vit/head.npz
evaluate-vit: $(VIT)
$(VIT): $(TOKENS) $(RANKING) $(RELIABILITY) configs/evaluate.yaml configs/evaluate_vit.yaml
	$(EVALUATE)/evaluate_vit.yaml

ENSEMBLE := $(TOP)/build/evaluate/ensemble/ensemble.json
ENSEMBLE_LAYOUT := $(TOP)/build/evaluate/ensemble_layout/ensemble.json
ENSEMBLE_FITTED := $(TOP)/build/evaluate/ensemble_fitted/ensemble.json
ENSEMBLE_LAYOUT_FITTED := $(TOP)/build/evaluate/ensemble_layout_fitted/ensemble.json
evaluate-ensembles: $(ENSEMBLE) $(ENSEMBLE_LAYOUT) $(ENSEMBLE_FITTED) $(ENSEMBLE_LAYOUT_FITTED)
$(ENSEMBLE): $(GEOMETRY) $(VIT) configs/evaluate.yaml configs/evaluate_ensemble.yaml
	$(EVALUATE)/evaluate_ensemble.yaml
$(ENSEMBLE_LAYOUT): $(GEOMETRY_LAYOUT) $(VIT) configs/evaluate.yaml configs/evaluate_ensemble_layout.yaml
	$(EVALUATE)/evaluate_ensemble_layout.yaml
$(ENSEMBLE_FITTED): $(GEOMETRY) $(VIT) configs/evaluate.yaml configs/evaluate_ensemble_fitted.yaml
	$(EVALUATE)/evaluate_ensemble_fitted.yaml
$(ENSEMBLE_LAYOUT_FITTED): $(GEOMETRY_LAYOUT) $(VIT) configs/evaluate.yaml configs/evaluate_ensemble_layout_fitted.yaml
	$(EVALUATE)/evaluate_ensemble_layout_fitted.yaml

HEAD := $(TOP)/build/train/vit_head/head.npz
head-dataset: $(HEAD)
$(HEAD): $(TOKENS) $(DATASET) configs/train_vit_head.yaml
	$(RUN) train --config $(CONFIGS)/train_vit_head.yaml

package-head: $(HEAD)
	cp $(HEAD) $(TOP)/src/layout_visual_complexity/heads/dinov2_vits14.npz

METHODS := $(GEOMETRY) $(VIT) $(ENSEMBLE)

# The baselines: each family's measures together, each measure alone (the pairwise logistic model on the columns of
# its prefix), and the linear model on the mean token.
MEAN_TOKEN_FEATURES := $(TOP)/build/features/mean_token/features.parquet
features-mean-token: $(MEAN_TOKEN_FEATURES)
$(MEAN_TOKEN_FEATURES): $(TOKENS) configs/features_mean_token.yaml
	$(RUN) features --config $(CONFIGS)/features_mean_token.yaml

GROUPS := trivial generic kyle_davidson ink_distribution
GROUP_MODELS := $(foreach group,$(GROUPS),$(TOP)/build/evaluate/$(group)/model.json)
GENERIC := $(TOP)/build/evaluate/generic/model.json
MEAN_TOKEN := $(TOP)/build/evaluate/mean_token/model.json
$(GROUP_MODELS): $(TOP)/build/evaluate/%/model.json: $(FEATURES) $(RANKING) $(RELIABILITY) configs/evaluate.yaml \
		configs/evaluate_%.yaml
	$(EVALUATE)/evaluate_$*.yaml
$(MEAN_TOKEN): $(MEAN_TOKEN_FEATURES) $(RANKING) $(RELIABILITY) configs/evaluate.yaml configs/evaluate_mean_token.yaml
	$(EVALUATE)/evaluate_mean_token.yaml

COLUMNS_ink_coverage := Trivial__InkCoverage
COLUMNS_text_lines := Trivial__TextLines
COLUMNS_edge_orientation_entropy := Generic__EdgeOrientationEntropy
COLUMNS_mssc := Generic__Mssc
COLUMNS_compressibility := Generic__Compressibility
COLUMNS_feature_congestion := Generic__FeatureCongestion
COLUMNS_colourfulness := Generic__Colourfulness
COLUMNS_codebook_variety := Generic__CodebookVariety
COLUMNS_clutter := KyleDavidson__Clutter
COLUMNS_symmetry := KyleDavidson__Symmetry
COLUMNS_entropy := KyleDavidson__Entropy
COLUMNS_lacunarity := Ink__Lacunarity
COLUMNS_fragmentation := Ink__Fragmentation
COLUMNS_chromatic_share := Ink__ChromaticShare
COLUMNS_hue_classes := Ink__HueClasses
MEASURES := $(patsubst COLUMNS_%,%,$(filter COLUMNS_%,$(.VARIABLES)))
MEASURE_MODELS := $(foreach measure,$(MEASURES),$(TOP)/build/evaluate/measures/$(measure)/model.json)
$(MEASURE_MODELS): $(TOP)/build/evaluate/measures/%/model.json: $(FEATURES) $(RANKING) $(RELIABILITY) \
		configs/evaluate.yaml configs/evaluate_measure.yaml
	$(EVALUATE)/evaluate_measure.yaml --columns $(COLUMNS_$*) --output build/evaluate/measures/$*

evaluate-baselines: $(GROUP_MODELS) $(MEAN_TOKEN)
evaluate-measures: $(MEASURE_MODELS)
BASELINES := $(GROUP_MODELS) $(MEAN_TOKEN) $(MEASURE_MODELS) $(GEOMETRY) $(GEOMETRY_LAYOUT) $(VIT) $(ENSEMBLE)

table-baselines-by-elo-gap: $(TOP)/results/tables/baselines_by_elo_gap.tsv
$(TOP)/results/tables/baselines_by_elo_gap.tsv: $(GROUP_MODELS) $(MEASURE_MODELS) configs/report_features.yaml \
		configs/table_baselines_by_elo_gap.yaml
	$(FEATURE_REPORT)/table_baselines_by_elo_gap.yaml

table-baseline-measures: $(TOP)/results/tables/baseline_measures.tsv
$(TOP)/results/tables/baseline_measures.tsv: $(FEATURES) configs/report_features.yaml \
		configs/table_baseline_measures.yaml
	$(FEATURE_REPORT)/table_baseline_measures.yaml

table-baseline-differences: $(TOP)/results/tables/baseline_differences.tsv
$(TOP)/results/tables/baseline_differences.tsv: $(GROUP_MODELS) $(MEAN_TOKEN) $(GEOMETRY) $(ENSEMBLE) \
		configs/report_features.yaml configs/table_baseline_differences.yaml
	$(FEATURE_REPORT)/table_baseline_differences.yaml

figure-baselines-by-information: $(TOP)/results/figures/baselines_by_information.svg
$(TOP)/results/figures/baselines_by_information.svg: $(BASELINES) configs/report_features.yaml \
		configs/figure_baselines_by_information.yaml
	$(FEATURE_REPORT)/figure_baselines_by_information.yaml

figure-baselines-by-information-each-measure: $(TOP)/results/figures/baselines_by_information_each_measure.svg
$(TOP)/results/figures/baselines_by_information_each_measure.svg: $(BASELINES) configs/report_features.yaml \
		configs/figure_baselines_by_information_each_measure.yaml
	$(FEATURE_REPORT)/figure_baselines_by_information_each_measure.yaml

figure-generic-measure-correlations: $(TOP)/results/figures/generic_measure_correlations.svg
$(TOP)/results/figures/generic_measure_correlations.svg: $(FEATURES) $(GENERIC) configs/report_features.yaml \
		configs/figure_generic_measure_correlations.yaml
	$(FEATURE_REPORT)/figure_generic_measure_correlations.yaml

table-methods-by-elo-gap: $(TOP)/results/tables/methods_by_elo_gap.tsv
$(TOP)/results/tables/methods_by_elo_gap.tsv: $(GENERIC) $(METHODS) configs/report_features.yaml \
		configs/table_methods_by_elo_gap.yaml
	$(FEATURE_REPORT)/table_methods_by_elo_gap.yaml

figure-accuracy-by-elo-gap: $(TOP)/results/figures/accuracy_by_elo_gap.svg
$(TOP)/results/figures/accuracy_by_elo_gap.svg: $(METHODS) configs/report_features.yaml \
		configs/figure_accuracy_by_elo_gap.yaml
	$(FEATURE_REPORT)/figure_accuracy_by_elo_gap.yaml

table-paired-differences: $(TOP)/results/tables/paired_differences.tsv
$(TOP)/results/tables/paired_differences.tsv: $(METHODS) configs/report_features.yaml \
		configs/table_paired_differences.yaml
	$(FEATURE_REPORT)/table_paired_differences.yaml

table-ensembles-by-elo-gap: $(TOP)/results/tables/ensembles_by_elo_gap.tsv
$(TOP)/results/tables/ensembles_by_elo_gap.tsv: $(ENSEMBLE) $(ENSEMBLE_LAYOUT) $(ENSEMBLE_FITTED) \
		$(ENSEMBLE_LAYOUT_FITTED) configs/report_features.yaml configs/table_ensembles_by_elo_gap.yaml
	$(FEATURE_REPORT)/table_ensembles_by_elo_gap.yaml

figure-learning-curves-and-calibration: $(TOP)/results/figures/learning_curves_and_calibration.svg
$(TOP)/results/figures/learning_curves_and_calibration.svg: $(METHODS) configs/report_features.yaml \
		configs/figure_learning_curves_and_calibration.yaml
	$(FEATURE_REPORT)/figure_learning_curves_and_calibration.yaml

figure-local-complexity-maps: $(TOP)/results/figures/local_complexity_maps.svg
$(TOP)/results/figures/local_complexity_maps.svg: $(HEAD) $(TOKENS) configs/report_features.yaml \
		configs/figure_local_complexity_maps.yaml
	$(FEATURE_REPORT)/figure_local_complexity_maps.yaml

clean:
	rm -rf $(TOP)/build/dataset $(TOP)/build/ranking $(TOP)/build/features/dataset $(TOP)/build/evaluate
	rm -rf $(TOP)/build/embed/dataset $(TOP)/build/train $(TOP)/build/codebook/dataset $(TOP)/build/features/mean_token
	rm -rf $(TOP)/results/tables $(TOP)/results/figures $(TOP)/results/runs
