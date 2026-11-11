# Changelog

## Unreleased

- The command is `lvc`. Settings come from YAML files and options; every run writes its configuration and a manifest
  of its inputs, their checksums and the versions it used.
- `lvc judgments import` reads a dataset of judgments in its published structure (judgments, pages and manuscripts as
  TSV, with a README), from a folder or from its Zenodo record by DOI, each file checked against the record's MD5,
  and checks the dataset before copying it: every column, every judgment's pages, every page's manuscript, canvas and
  image link, dates in EDTF and languages in ISO 639-3.
- `lvc judgments rank` computes Elo scores over seeded permutations of the judgments' order, in parallel, and
  Bradley-Terry scores beside them.
- `lvc judgments report` writes reliability tables (consistency and agreement by Elo gap, position bias, the
  consistency reference per fold, the split-half ceiling), the comparison graph and the score distributions.
- The evaluation protocol: folds by manuscript (10 × 10 by default), annotator or category; the pairwise logistic
  model on any table of page features; accuracy by Elo gap, SROCC, log-loss and calibration, bootstrap intervals over
  manuscripts, paired comparisons and learning curves.
- `reproduce/` regenerates the article's Table 1, Table 2 and Figures 2 and 3, and imports the dataset again when the
  folder it reads from changes.
