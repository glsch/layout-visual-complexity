# Changelog

## Unreleased

- The command is now `lvc`; `layout-visual-complexity` is gone. Settings come from YAML files and options.
- `lvc judgments import` turns the annotation app's export into an anonymised dataset: judgments with pseudonymous
  annotators, the images with their IIIF canvases, and the manuscripts.
- `lvc judgments rank` computes Elo scores over seeded permutations of the judgments' order, in parallel, and
  Bradley-Terry scores beside them.
- `lvc judgments report` writes reliability tables (consistency and agreement by Elo gap, position bias, the
  consistency reference per fold, the split-half ceiling), the comparison graph and the score distributions.
- The evaluation protocol: folds by manuscript, annotator or category; the pairwise logistic model on any table of
  page features; accuracy by Elo gap, SROCC, log-loss and calibration, bootstrap intervals over manuscripts, paired
  comparisons and learning curves.
- `reproduce/` regenerates the article's Table 1, Table 2 and Figures 2 and 3.
- `lvc judgments import` takes tables of manuscript records corrected by hand (`manuscript_overrides`; a corrected
  manifest carries the manuscript's pages with it) and of canvases checked by hand (`canvas_overrides`); it finds a
  page whose file name gives its canvas's identifier (`00055-ID0025r`) by that identifier, and adds `notes` to the
  dataset's README. `manuscripts.tsv` has a `category_confidence` column.
- The folds by manuscript are drawn ten times by default (10 × 10), so the consistency reference depends less on
  the seed.
