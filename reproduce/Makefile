# The article's tables and figures, regenerated with the lvc command line:
#
#     make -C reproduce all
#
# Paths in the configurations are relative to the working folder (the repository's root), so every command runs
# from there. The dataset and the ranking go to build/, the tables and figures to results/.

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

.PHONY: all dataset ranking table-dataset-categories table-reliability figure-comparison-graph figure-elo-violins \
	all-sources clean

all: table-dataset-categories table-reliability figure-comparison-graph figure-elo-violins

dataset: $(DATASET)
$(DATASET): configs/dataset.yaml
	$(RUN) judgments import --config $(CONFIGS)/dataset.yaml

ranking: $(RANKING)
$(RANKING): $(DATASET) configs/ranking.yaml
	$(RUN) judgments rank --config $(CONFIGS)/ranking.yaml

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

# Every source kept, the exhibition sessions included, as the article computed its numbers; into build/ only.
ALL_SOURCES := build/all_sources
all-sources:
	$(RUN) judgments import --config $(CONFIGS)/dataset.yaml --config $(CONFIGS)/dataset_all_sources.yaml
	$(RUN) judgments rank --config $(CONFIGS)/ranking.yaml --dataset $(ALL_SOURCES)/dataset \
		--output $(ALL_SOURCES)/ranking
	for table in table_dataset_categories table_reliability figure_comparison_graph figure_elo_violins; do \
		$(REPORT)/$$table.yaml --dataset $(ALL_SOURCES)/dataset --ranking $(ALL_SOURCES)/ranking \
			--output $(ALL_SOURCES)/results || exit 1; \
	done

clean:
	rm -rf $(TOP)/build/dataset $(TOP)/build/ranking $(TOP)/$(ALL_SOURCES)
	rm -rf $(TOP)/results/tables $(TOP)/results/figures $(TOP)/results/runs
