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
#
# The colour level (line texture as the article publishes it, and ink colour) is measured with the others too; its
# ensemble with the geometry and ViT rankers needs the ViT's evaluation.
#
# The rankers that come with the package (package-models) are the geometry ranker and the ViT head trained on every
# judgment, with their scales of win probabilities and the Elo scale from the evaluations. Crux of Telč's case study
# scores its 23 manuscripts with the rankers trained without its manuscripts in the dataset (score-crux, not part of
# `all`), and draws its figures and tables from that run and the catalogue's facts in crux_manuscripts.tsv.
#
# The difficulty experiment measures how hard CREMMA's pages, and the held-out pages of CATMuS's test split, are for
# the regions model, Kraken's lines and Kraken's recognition with CATMuS's model, beside their predicted complexity.
# Its sources come at fixed commits and a fixed revision, checked (sources); its runs and outputs aren't part of
# `all` (difficulty).

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

.PHONY: all dataset ranking segment-dataset table-dataset-categories table-reliability figure-position-bias-funnel \
	figure-comparison-graph figure-elo-score-bands figure-elo-score-bands-trios features-dataset evaluate-geometry \
	evaluate-geometry-layout table-vit-backbones \
	table-geometry-by-elo-gap table-geometry-feature-hierarchy figure-geometry-feature-correlations \
	figure-geometry-features-on-page figure-geometry-feature-families \
	embed-dataset evaluate-vit evaluate-ensembles head-dataset package-head train-geometry models-dataset \
	package-models models-without-crux score-crux table-methods-by-elo-gap \
	figure-accuracy-by-elo-gap table-paired-differences table-ensembles-by-elo-gap figure-learning-curves-and-calibration \
	figure-local-complexity-maps codebook-dataset package-codebook features-mean-token evaluate-baselines \
	evaluate-measures table-baselines-by-elo-gap table-baseline-measures table-baseline-differences \
	figure-baselines-by-information figure-baselines-by-information-each-measure figure-generic-measure-correlations \
	evaluate-colour evaluate-colour-measures table-colour-by-elo-gap table-colour-differences table-line-texture-families \
	table-colour-measures table-within-and-across-manuscripts figure-ink-colour-on-page figure-manuscript-profiles \
	figure-manuscript-strips figure-crux-overview table-crux-manuscripts table-crux-readings sources difficulty \
	difficulty-cremma difficulty-catmus-test figure-difficulty-against-complexity table-difficulty-regression \
	table-difficulty-regression-across-and-within table-difficulty-held-out table-difficulty-correlations \
	table-difficulty-main-text-and-margins table-difficulty-ground-truth-coverage table-difficulty-by-ranker \
	table-difficulty-by-quarter table-difficulty-material difficulty-pages clean

all: table-dataset-categories table-reliability figure-position-bias-funnel figure-comparison-graph figure-elo-score-bands \
	figure-elo-score-bands-trios table-geometry-by-elo-gap table-geometry-feature-hierarchy \
	figure-geometry-feature-correlations figure-geometry-features-on-page figure-geometry-feature-families \
	table-methods-by-elo-gap figure-accuracy-by-elo-gap table-paired-differences table-ensembles-by-elo-gap \
	figure-learning-curves-and-calibration figure-local-complexity-maps table-baselines-by-elo-gap \
	table-baseline-measures table-baseline-differences figure-baselines-by-information \
	figure-baselines-by-information-each-measure figure-generic-measure-correlations table-colour-by-elo-gap \
	table-colour-differences table-line-texture-families table-colour-measures table-within-and-across-manuscripts \
	figure-ink-colour-on-page figure-manuscript-profiles figure-manuscript-strips figure-crux-overview \
	table-crux-manuscripts table-crux-readings

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

figure-position-bias-funnel: $(TOP)/results/figures/position_bias_funnel.svg
$(TOP)/results/figures/position_bias_funnel.svg: $(RANKING) configs/report.yaml configs/figure_position_bias_funnel.yaml
	$(REPORT)/figure_position_bias_funnel.yaml

figure-comparison-graph: $(TOP)/results/figures/comparison_graph.svg
$(TOP)/results/figures/comparison_graph.svg: $(RANKING) configs/report.yaml configs/figure_comparison_graph.yaml
	$(REPORT)/figure_comparison_graph.yaml

# The score distributions as bands, with thumbnails of ten pages.
figure-elo-score-bands: $(TOP)/results/figures/elo_score_bands.svg
$(TOP)/results/figures/elo_score_bands.svg: $(RANKING) configs/report.yaml configs/figure_elo_score_bands.yaml
	$(REPORT)/figure_elo_score_bands.yaml

# The same bands with three pages at each place: similar scores can belong to quite different layouts.
figure-elo-score-bands-trios: $(TOP)/results/figures/elo_score_bands_trios.svg
$(TOP)/results/figures/elo_score_bands_trios.svg: $(RANKING) configs/report.yaml \
		configs/figure_elo_score_bands_trios.yaml
	$(REPORT)/figure_elo_score_bands_trios.yaml

# The features of the dataset's pages, the geometry ranker with and without Layout (Kyle–Davidson), and their tables
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

# The rankers that come with the package, which lvc score and lvc compare use: the geometry ranker and the head, each
# trained on every judgment, and their ensemble with equal weights. Their scales of win probabilities are fitted on
# the evaluations' out-of-fold scores, and the ensemble's Elo scale is the regression of the Elo scores on its
# evaluation's out-of-fold scores. `make -C reproduce package-models` copies them into the package.
TRAINED_GEOMETRY := $(TOP)/build/train/geometry/model.json
MODELS := $(TOP)/build/train/models/ensemble.json
train-geometry: $(TRAINED_GEOMETRY)
$(TRAINED_GEOMETRY): $(FEATURES) $(DATASET) configs/train_geometry.yaml
	$(RUN) train --config $(CONFIGS)/train_geometry.yaml

models-dataset: $(MODELS)
$(MODELS): $(TRAINED_GEOMETRY) $(HEAD) $(GEOMETRY) $(VIT) $(ENSEMBLE) configs/train_models.yaml
	$(RUN) train --config $(CONFIGS)/train_models.yaml

package-models: $(MODELS)
	cp $(addprefix $(TOP)/build/train/models/,ensemble.json geometry.json dinov2_vits14.npz) \
		$(TOP)/src/layout_visual_complexity/heads/

# Crux of Telč's case study, scored out of sample: the rankers trained without the four of its manuscripts that gave
# the dataset pages. Scoring its 14,024 leaves isn't part of `all` (it needs the segment and vit extras, the models
# in models/, the network, and about four hours on an Apple M4 Pro), but the case study's figures and tables are,
# and make it first if it's missing.
GEOMETRY_WITHOUT_CRUX := $(TOP)/build/train/geometry_without_crux/model.json
HEAD_WITHOUT_CRUX := $(TOP)/build/train/vit_head_without_crux/head.npz
MODELS_WITHOUT_CRUX := $(TOP)/build/train/models_without_crux/ensemble.json
$(GEOMETRY_WITHOUT_CRUX): $(FEATURES) $(DATASET) configs/train_geometry_without_crux.yaml
	$(RUN) train --config $(CONFIGS)/train_geometry_without_crux.yaml
$(HEAD_WITHOUT_CRUX): $(TOKENS) $(DATASET) configs/train_vit_head_without_crux.yaml
	$(RUN) train --config $(CONFIGS)/train_vit_head_without_crux.yaml
models-without-crux: $(MODELS_WITHOUT_CRUX)
$(MODELS_WITHOUT_CRUX): $(GEOMETRY_WITHOUT_CRUX) $(HEAD_WITHOUT_CRUX) $(GEOMETRY) $(VIT) $(ENSEMBLE) \
		configs/train_models_without_crux.yaml
	$(RUN) train --config $(CONFIGS)/train_models_without_crux.yaml

SCORE_CRUX := $(TOP)/build/score/crux/scores.tsv
score-crux: $(SCORE_CRUX)
$(SCORE_CRUX): $(MODELS_WITHOUT_CRUX) configs/score_crux.yaml
	$(RUN) score --config $(CONFIGS)/score_crux.yaml

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
COLUMNS_colour_chromatic_share := Colour__ChromaticShare
COLUMNS_colour_hue_classes := Colour__HueClasses
COLUMNS_colour_hue_entropy := Colour__HueEntropy
COLUMNS_colour_saturation := Colour__Saturation
COLUMNS_colour_zones := Colour__Zones
COLUMNS_colour_lines := Colour__Lines
COLUMNS_colour_coloured_initials := Colour__ColouredInitials
MEASURES := $(patsubst COLUMNS_%,%,$(filter COLUMNS_%,$(.VARIABLES)))
MEASURE_MODELS := $(foreach measure,$(MEASURES),$(TOP)/build/evaluate/measures/$(measure)/model.json)
$(MEASURE_MODELS): $(TOP)/build/evaluate/measures/%/model.json: $(FEATURES) $(RANKING) $(RELIABILITY) \
		configs/evaluate.yaml configs/evaluate_measure.yaml
	$(EVALUATE)/evaluate_measure.yaml --columns $(COLUMNS_$*) --output build/evaluate/measures/$*

evaluate-baselines: $(GROUP_MODELS) $(MEAN_TOKEN)
evaluate-measures: $(MEASURE_MODELS)

# The colour level: line texture and ink colour alone, each with geometry in one model, and the ensemble with ink
# colour.
COLOUR_GROUPS := line_texture colour geometry_line_texture geometry_colour
COLOUR_MODELS := $(foreach group,$(COLOUR_GROUPS),$(TOP)/build/evaluate/$(group)/model.json)
COLOUR := $(TOP)/build/evaluate/colour/model.json
LINE_TEXTURE := $(TOP)/build/evaluate/line_texture/model.json
GEOMETRY_COLOUR := $(TOP)/build/evaluate/geometry_colour/model.json
GEOMETRY_LINE_TEXTURE := $(TOP)/build/evaluate/geometry_line_texture/model.json
ENSEMBLE_COLOUR := $(TOP)/build/evaluate/ensemble_colour/ensemble.json
$(COLOUR_MODELS): $(TOP)/build/evaluate/%/model.json: $(FEATURES) $(RANKING) $(RELIABILITY) configs/evaluate.yaml \
		configs/evaluate_%.yaml
	$(EVALUATE)/evaluate_$*.yaml
$(ENSEMBLE_COLOUR): $(GEOMETRY) $(VIT) $(COLOUR) configs/evaluate.yaml configs/evaluate_ensemble_colour.yaml
	$(EVALUATE)/evaluate_ensemble_colour.yaml
COLOUR_MEASURE_MODELS := $(foreach measure,$(filter colour_%,$(MEASURES)),$(TOP)/build/evaluate/measures/$(measure)/model.json)
evaluate-colour: $(COLOUR_MODELS) $(ENSEMBLE_COLOUR)
evaluate-colour-measures: $(COLOUR_MEASURE_MODELS)

BASELINES := $(GROUP_MODELS) $(MEAN_TOKEN) $(MEASURE_MODELS) $(GEOMETRY) $(GEOMETRY_LAYOUT) $(VIT) $(ENSEMBLE) \
	$(COLOUR) $(LINE_TEXTURE)

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
$(TOP)/results/tables/methods_by_elo_gap.tsv: $(GENERIC) $(COLOUR) $(METHODS) configs/report_features.yaml \
		configs/table_methods_by_elo_gap.yaml
	$(FEATURE_REPORT)/table_methods_by_elo_gap.yaml

figure-accuracy-by-elo-gap: $(TOP)/results/figures/accuracy_by_elo_gap.svg
$(TOP)/results/figures/accuracy_by_elo_gap.svg: $(LINE_TEXTURE) $(COLOUR) $(METHODS) configs/report_features.yaml \
		configs/figure_accuracy_by_elo_gap.yaml
	$(FEATURE_REPORT)/figure_accuracy_by_elo_gap.yaml

table-paired-differences: $(TOP)/results/tables/paired_differences.tsv
$(TOP)/results/tables/paired_differences.tsv: $(METHODS) configs/report_features.yaml \
		configs/table_paired_differences.yaml
	$(FEATURE_REPORT)/table_paired_differences.yaml

# DINOv3 ViT-S/16 against DINOv2 ViT-S/14: the same head on the same folds, their accuracy and SROCC, and the
# difference (table-vit-backbones). Not part of `all`: DINOv3's repository on Hugging Face is gated, so its tokens
# need access granted to an account and that account's token, HF_TOKEN in ../.env, which only the tokens' recipe sees
# (and the segmentation's, if it has to be made first). Its evaluation takes about 15 minutes on an Apple M4 Pro.
TOKENS_DINOV3 := $(TOP)/build/embed/dataset_dinov3/tokens.json
VIT_DINOV3 := $(TOP)/build/evaluate/vit_dinov3/head.npz
$(TOKENS_DINOV3): export HF_TOKEN := $(HF_TOKEN)
$(TOKENS_DINOV3): $(SEGMENT) configs/embed_dataset_dinov3.yaml
	$(RUN) embed --config $(CONFIGS)/embed_dataset_dinov3.yaml
$(VIT_DINOV3): $(TOKENS_DINOV3) $(RANKING) $(RELIABILITY) configs/evaluate.yaml configs/evaluate_vit_dinov3.yaml
	$(EVALUATE)/evaluate_vit_dinov3.yaml

table-vit-backbones: $(TOP)/results/tables/vit_backbones.tsv
$(TOP)/results/tables/vit_backbones.tsv: $(VIT) $(VIT_DINOV3) configs/report_features.yaml \
		configs/table_vit_backbones.yaml
	$(FEATURE_REPORT)/table_vit_backbones.yaml

table-ensembles-by-elo-gap: $(TOP)/results/tables/ensembles_by_elo_gap.tsv
$(TOP)/results/tables/ensembles_by_elo_gap.tsv: $(ENSEMBLE) $(ENSEMBLE_LAYOUT) $(ENSEMBLE_FITTED) \
		$(ENSEMBLE_LAYOUT_FITTED) $(ENSEMBLE_COLOUR) configs/report_features.yaml configs/table_ensembles_by_elo_gap.yaml
	$(FEATURE_REPORT)/table_ensembles_by_elo_gap.yaml

figure-learning-curves-and-calibration: $(TOP)/results/figures/learning_curves_and_calibration.svg
$(TOP)/results/figures/learning_curves_and_calibration.svg: $(METHODS) configs/report_features.yaml \
		configs/figure_learning_curves_and_calibration.yaml
	$(FEATURE_REPORT)/figure_learning_curves_and_calibration.yaml

figure-local-complexity-maps: $(TOP)/results/figures/local_complexity_maps.svg
$(TOP)/results/figures/local_complexity_maps.svg: $(HEAD) $(TOKENS) configs/report_features.yaml \
		configs/figure_local_complexity_maps.yaml
	$(FEATURE_REPORT)/figure_local_complexity_maps.yaml

table-colour-by-elo-gap: $(TOP)/results/tables/colour_by_elo_gap.tsv
$(TOP)/results/tables/colour_by_elo_gap.tsv: $(LINE_TEXTURE) $(COLOUR) $(GEOMETRY_COLOUR) $(GEOMETRY) \
		configs/report_features.yaml configs/table_colour_by_elo_gap.yaml
	$(FEATURE_REPORT)/table_colour_by_elo_gap.yaml

table-colour-differences: $(TOP)/results/tables/colour_differences.tsv
$(TOP)/results/tables/colour_differences.tsv: $(GEOMETRY_LINE_TEXTURE) $(GEOMETRY_COLOUR) $(GEOMETRY) \
		$(ENSEMBLE_COLOUR) $(ENSEMBLE) configs/report_features.yaml configs/table_colour_differences.yaml
	$(FEATURE_REPORT)/table_colour_differences.yaml

table-line-texture-families: $(TOP)/results/tables/line_texture_families.tsv
$(TOP)/results/tables/line_texture_families.tsv: configs/report_features.yaml configs/table_line_texture_families.yaml
	$(FEATURE_REPORT)/table_line_texture_families.yaml

table-colour-measures: $(TOP)/results/tables/colour_measures.tsv
$(TOP)/results/tables/colour_measures.tsv: $(FEATURES) configs/report_features.yaml configs/table_colour_measures.yaml
	$(FEATURE_REPORT)/table_colour_measures.yaml

ALL_METHODS := $(BASELINES) $(COLOUR_MODELS) $(ENSEMBLE_COLOUR) $(ENSEMBLE_LAYOUT) $(ENSEMBLE_FITTED) \
	$(ENSEMBLE_LAYOUT_FITTED)
table-within-and-across-manuscripts: $(TOP)/results/tables/within_and_across_manuscripts.tsv
$(TOP)/results/tables/within_and_across_manuscripts.tsv: $(ALL_METHODS) configs/report_features.yaml \
		configs/table_within_and_across_manuscripts.yaml
	$(FEATURE_REPORT)/table_within_and_across_manuscripts.yaml

figure-ink-colour-on-page: $(TOP)/results/figures/ink_colour_on_page.svg
$(TOP)/results/figures/ink_colour_on_page.svg: $(FEATURES) configs/report_features.yaml \
		configs/figure_ink_colour_on_page.yaml
	$(FEATURE_REPORT)/figure_ink_colour_on_page.yaml

# Crux of Telč's case study: the profiles of I D 18 and XI C 1, every manuscript's strip, the overview, the table of
# the 23 manuscripts and the article's readings, from the score run (score-crux) and the catalogue's facts.
CRUX_REPORT := $(RUN) report --config $(CONFIGS)/report_crux.yaml --config $(CONFIGS)
CATALOGUE := crux_manuscripts.tsv

figure-manuscript-profiles: $(TOP)/results/figures/manuscript_profile_id18.svg
$(TOP)/results/figures/manuscript_profile_id18.svg: $(SCORE_CRUX) $(CATALOGUE) configs/report_crux.yaml \
		configs/figure_manuscript_profiles.yaml
	$(CRUX_REPORT)/figure_manuscript_profiles.yaml

figure-manuscript-strips: $(TOP)/results/figures/manuscript_strips.svg
$(TOP)/results/figures/manuscript_strips.svg: $(SCORE_CRUX) $(CATALOGUE) configs/report_crux.yaml \
		configs/figure_manuscript_strips.yaml
	$(CRUX_REPORT)/figure_manuscript_strips.yaml

figure-crux-overview: $(TOP)/results/figures/crux_overview.svg
$(TOP)/results/figures/crux_overview.svg: $(SCORE_CRUX) $(CATALOGUE) configs/report_crux.yaml \
		configs/figure_crux_overview.yaml
	$(CRUX_REPORT)/figure_crux_overview.yaml

table-crux-manuscripts: $(TOP)/results/tables/crux_manuscripts.tsv
$(TOP)/results/tables/crux_manuscripts.tsv: $(SCORE_CRUX) $(CATALOGUE) configs/report_crux.yaml \
		configs/table_crux_manuscripts.yaml
	$(CRUX_REPORT)/table_crux_manuscripts.yaml

table-crux-readings: $(TOP)/results/tables/crux_readings.tsv
$(TOP)/results/tables/crux_readings.tsv: $(SCORE_CRUX) $(CATALOGUE) configs/report_crux.yaml \
		configs/table_crux_readings.yaml
	$(CRUX_REPORT)/table_crux_readings.yaml

# The difficulty experiment's sources, in build/sources/: CREMMA Medieval and CREMMA Medii Aevi (Latin) at their
# commits, each fetched shallowly at that commit, and the files of CATMuS Medieval Segmentation's test split the
# experiment uses (its metadata and the images of 15 manuscripts), at a fixed revision, each checked against its
# SHA-256 from the Hub's listing (catmus_segmentation_files.tsv). A repository already there is checked, not fetched.
SOURCES := $(TOP)/build/sources
CREMMA_MEDIEVAL := bc4a105f6e8316a9b30d9ce5d98dda2ba947d7d4
CREMMA_LATIN := 292525969ad98380b398e6606a9c2a36d51913ae
CATMUS_REVISION := e8fb686763d061627fd78f9b516a4db37ac25d5a
CATMUS := https://huggingface.co/datasets/CATMuS/medieval-segmentation/resolve/$(CATMUS_REVISION)
CATMUS_FILES := catmus_segmentation_files.tsv

# $(call at_commit,repository URL,folder,commit)
at_commit = if [ ! -e $(2)/.git ]; then git init -q $(2) && git -C $(2) fetch -q --depth 1 $(1) $(3) \
	&& git -C $(2) -c advice.detachedHead=false checkout -q FETCH_HEAD; fi; \
	test "$$(git -C $(2) rev-parse HEAD)" = $(3) || { echo "$(2) is not at $(3)"; exit 1; }

sources: $(SOURCES)/cremma.checked $(SOURCES)/catmus.checked

$(SOURCES)/cremma.checked:
	mkdir -p $(SOURCES)/cremma
	$(call at_commit,https://github.com/HTR-United/cremma-medieval.git,$(SOURCES)/cremma/cremma-medieval,$(CREMMA_MEDIEVAL))
	$(call at_commit,https://github.com/HTR-United/CREMMA-Medieval-LAT.git,$(SOURCES)/cremma/cremma-medieval-lat,$(CREMMA_LATIN))
	touch $@

$(SOURCES)/catmus.checked: $(CATMUS_FILES)
	tail -n +2 $(CATMUS_FILES) | while IFS="$$(printf '\t')" read -r path size sha; do \
		target=$(SOURCES)/catmus/$$path; \
		if ! echo "$$sha  $$target" | shasum -a 256 -c --status 2>/dev/null; then \
			mkdir -p "$$(dirname "$$target")"; \
			curl -sSfL --retry 3 -o "$$target.part" "$(CATMUS)/$$path" && mv "$$target.part" "$$target" || exit 1; \
			echo "$$sha  $$target" | shasum -a 256 -c --status || { echo "$$path: its SHA-256 differs"; exit 1; }; \
		fi; \
	done
	touch $@

# The difficulty experiment: CREMMA's 400 pages and the held-out pages of CATMuS's test split measured against their
# ground truth (lvc difficulty), and its figure and tables. None of it is part of `all`: the runs need the segment and
# vit extras, the models in models/ and the sources; on an Apple M4 Pro CREMMA's run takes about 45 minutes, the
# test split's a few. `make -C reproduce difficulty` makes them all.
DIFFICULTY_CREMMA := $(TOP)/build/difficulty/cremma/difficulty.tsv
DIFFICULTY_CATMUS := $(TOP)/build/difficulty/catmus_test/difficulty.tsv
MANUSCRIPT_TABLE := difficulty_manuscripts.tsv
difficulty-cremma: $(DIFFICULTY_CREMMA)
$(DIFFICULTY_CREMMA): $(SOURCES)/cremma.checked configs/difficulty_cremma.yaml $(MANUSCRIPT_TABLE)
	$(RUN) difficulty --config $(CONFIGS)/difficulty_cremma.yaml
difficulty-catmus-test: $(DIFFICULTY_CATMUS)
$(DIFFICULTY_CATMUS): $(SOURCES)/catmus.checked configs/difficulty_catmus_test.yaml $(MANUSCRIPT_TABLE)
	$(RUN) difficulty --config $(CONFIGS)/difficulty_catmus_test.yaml

DIFFICULTY_REPORT := $(RUN) report --config $(CONFIGS)/report_difficulty.yaml --config $(CONFIGS)
DIFFICULTY_RUNS := $(DIFFICULTY_CREMMA) $(DIFFICULTY_CATMUS) $(MANUSCRIPT_TABLE) configs/report_difficulty.yaml
DIFFICULTY_OUTPUTS := figure-difficulty-against-complexity table-difficulty-regression \
	table-difficulty-regression-across-and-within table-difficulty-held-out table-difficulty-correlations \
	table-difficulty-main-text-and-margins table-difficulty-ground-truth-coverage table-difficulty-by-ranker \
	table-difficulty-by-quarter table-difficulty-material difficulty-pages
difficulty: $(DIFFICULTY_OUTPUTS)

figure-difficulty-against-complexity: $(TOP)/results/figures/difficulty_against_complexity.svg
$(TOP)/results/figures/difficulty_against_complexity.svg: $(DIFFICULTY_RUNS) configs/figure_difficulty_against_complexity.yaml
	$(DIFFICULTY_REPORT)/figure_difficulty_against_complexity.yaml

table-difficulty-regression: $(TOP)/results/tables/difficulty_regression.tsv
$(TOP)/results/tables/difficulty_regression.tsv: $(DIFFICULTY_RUNS) configs/table_difficulty_regression.yaml
	$(DIFFICULTY_REPORT)/table_difficulty_regression.yaml

table-difficulty-regression-across-and-within: $(TOP)/results/tables/difficulty_regression_across_and_within.tsv
$(TOP)/results/tables/difficulty_regression_across_and_within.tsv: $(DIFFICULTY_RUNS) \
		configs/table_difficulty_regression_across_and_within.yaml
	$(DIFFICULTY_REPORT)/table_difficulty_regression_across_and_within.yaml

table-difficulty-held-out: $(TOP)/results/tables/difficulty_held_out.tsv
$(TOP)/results/tables/difficulty_held_out.tsv: $(DIFFICULTY_RUNS) configs/table_difficulty_held_out.yaml
	$(DIFFICULTY_REPORT)/table_difficulty_held_out.yaml

table-difficulty-correlations: $(TOP)/results/tables/difficulty_correlations.tsv
$(TOP)/results/tables/difficulty_correlations.tsv: $(DIFFICULTY_RUNS) configs/table_difficulty_correlations.yaml
	$(DIFFICULTY_REPORT)/table_difficulty_correlations.yaml

table-difficulty-main-text-and-margins: $(TOP)/results/tables/difficulty_main_text_and_margins.tsv
$(TOP)/results/tables/difficulty_main_text_and_margins.tsv: $(DIFFICULTY_RUNS) \
		configs/table_difficulty_main_text_and_margins.yaml
	$(DIFFICULTY_REPORT)/table_difficulty_main_text_and_margins.yaml

table-difficulty-ground-truth-coverage: $(TOP)/results/tables/difficulty_ground_truth_coverage.tsv
$(TOP)/results/tables/difficulty_ground_truth_coverage.tsv: $(DIFFICULTY_RUNS) \
		configs/table_difficulty_ground_truth_coverage.yaml
	$(DIFFICULTY_REPORT)/table_difficulty_ground_truth_coverage.yaml

table-difficulty-by-ranker: $(TOP)/results/tables/difficulty_by_ranker.tsv
$(TOP)/results/tables/difficulty_by_ranker.tsv: $(DIFFICULTY_RUNS) configs/table_difficulty_by_ranker.yaml
	$(DIFFICULTY_REPORT)/table_difficulty_by_ranker.yaml

table-difficulty-by-quarter: $(TOP)/results/tables/difficulty_by_quarter.tsv
$(TOP)/results/tables/difficulty_by_quarter.tsv: $(DIFFICULTY_RUNS) configs/table_difficulty_by_quarter.yaml
	$(DIFFICULTY_REPORT)/table_difficulty_by_quarter.yaml

table-difficulty-material: $(TOP)/results/tables/difficulty_material.tsv
$(TOP)/results/tables/difficulty_material.tsv: $(DIFFICULTY_RUNS) configs/table_difficulty_material.yaml
	$(DIFFICULTY_REPORT)/table_difficulty_material.yaml

difficulty-pages: $(TOP)/results/difficulty/cremma.tsv
$(TOP)/results/difficulty/cremma.tsv: $(DIFFICULTY_RUNS) configs/difficulty_pages.yaml
	$(DIFFICULTY_REPORT)/difficulty_pages.yaml

clean:
	rm -rf $(TOP)/build/dataset $(TOP)/build/ranking $(TOP)/build/features/dataset $(TOP)/build/evaluate
	rm -rf $(TOP)/build/embed/dataset $(TOP)/build/train $(TOP)/build/codebook/dataset $(TOP)/build/features/mean_token
	rm -rf $(TOP)/results/tables $(TOP)/results/figures $(TOP)/results/runs
