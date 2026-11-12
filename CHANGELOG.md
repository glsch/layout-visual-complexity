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
- In the published structure, a manuscript's identifier in the Handschriftencensus is `hscensus_id`.
- `lvc segment` finds the folio crop, the regions and the text lines of pages from IIIF manifests and image services,
  image URLs, local images, folders, lists and page tables, into a layout store in Parquet (and ALTO 4 with
  `--alto`); its steps run at once in worker processes, YOLO and Kraken's network on Apple's GPU where there is one,
  and a rerun resumes. It needs the new `segment` extra (Kraken, PyTorch, ultralytics).
  - The folio crop is Otsu's threshold on lightness, the whole image where its quadrilateral is implausible; a YOLO
    folio model can seed it, for pages photographed on a bright ground.
  - The regions model's line classes (headings, interlinear glosses, music) are kept beside its zones.
  - Kraken keeps up to 500 lines a page instead of its own 200 (`lines.max_lines`).
- Downloads now fetch from several servers at once, with up to two requests in flight to each.
- `reproduce/` segments the dataset's pages (`make -C reproduce segment-dataset`).
