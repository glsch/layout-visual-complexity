# Changelog

## Unreleased

- Two new feature families, the colour level: `line_texture`, the article's 38 line-texture measures as its Appendix
  B defines them and under its names, and `colour`, real colour on the ink layer (chromatic share, hue classes, hue
  entropy, saturation, the colour by kind of zone and across the text lines, coloured initials), the ink within
  stamps left out. It shares the ink family's colour code.
- The ink family's and the colour family's chromatic ink leaves out the hues from 60° to 110° (yellow, gold and
  olive), where brown writing reaches the chroma threshold against the parchment, whatever their chroma (a setting,
  `excluded_hues`); the hue classes are red, green and blue, and `Ink__HueClasses__gold` is gone. The ink family's
  chromatic share and hue classes change on the pages with such ink; its version is 2, so earlier feature tables are
  measured again.
- Every evaluation now gives its accuracy on the test judgments within one manuscript and across manuscripts, apart,
  and its SROCC within manuscripts, each with its interval (`metrics.tsv`); `lvc report` computes them from the hits
  and scores of any evaluation, an older one too, and paired differences give them on the same resamples.
- The methods' tables and figures name the consistency reference and the ceiling for SROCC as one "Reference"; their
  TSVs call its column `reference` (was `consistency`). Values don't change.
- `lvc report` writes the colour level's tables and figure (`colour_by_elo_gap`, `colour_differences`,
  `line_texture_families`, `colour_measures`, `ink_colour_on_page`) and `within_and_across_manuscripts`, and takes
  `shown`, the evaluations a table shows as columns while the others serve its comparisons.
- Kyle-Davidson's clutter no longer depends on the pages a worker measured before: the normalised cut solves its
  eigenvectors densely instead of with ARPACK, which keeps state between calls. Its values move on about a quarter of
  the dataset's pages, by four regions at most; the family's version is 2, so earlier feature tables are measured
  again.
- `reproduce/` regenerates Appendix B's successor, the colour level's tables and figure, the table of every method
  within and across manuscripts, and Table 4, the accuracy figure, the ensembles and the information figures with
  the colour level.
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
- `lvc features` measures feature families on a segmentation run's pages, in worker processes and resumably, into a
  table with one row per page that records each column's family and settings. Families keep a common contract and
  are chosen by short name or Python path, so other projects can add their own. Two come with the package:
  `geometry`, the article's 83 geometric features of the written space and the text lines, and `layout`, 22 features
  adapting Kyle-Davidson's clutter, order and entropy to the page, with its blank space.
- `lvc evaluate` runs the evaluation protocol on a feature table: accuracy by Elo gap beside the consistency reference,
  SROCC beside the split-half ceiling, log-loss and calibration, with intervals from resamples of the manuscripts (a
  judgment between two pages of one manuscript counts as often as that manuscript is drawn); and keeps the model
  trained on every judgment, which refuses features computed with other settings.
- `lvc report` writes the features' tables and figures: accuracy by Elo gap, the features' table, each feature's
  correlation with the Elo scores, and the features drawn on pages.
- Every figure has one style: a colour per method, STIX text, drawn at its print width (15 or 7.2 cm) and included in
  LaTeX at that width. The comparison graph and the score distributions are redrawn with it.
- `reproduce/` regenerates Table 4's geometry column, Appendix A's table and Figure 9's successor, and draws the
  features on pages.
- `lvc embed` gives each page's patch tokens from a vision transformer: any `transformers` model with patch tokens, at
  a pinned revision, DINOv2 ViT-S/14 by default, on the folio crop or the whole page; kept in the user cache. Needs
  the new `vit` extra (PyTorch, transformers).
- `lvc evaluate` also evaluates the local ViT head on a token set (trained per fold with RankNet's loss, in worker
  processes, with results that don't depend on their number) and ensembles of earlier evaluations, with equal, given
  or fitted weights; it now writes each fold's hits and z-scores, so that two methods compare on the same resamples,
  and learning curves on request. Feature tables are given with `--features`, which the shared configuration no
  longer has to name.
- `lvc train` trains the pairwise model or the ViT head on every judgment. The package comes with a ViT head trained
  on the dataset's 397 pages.
- `lvc report` writes the methods' tables and figures: accuracy by Elo gap (Table 4's successor, as a table and a
  figure), differences between methods, learning curves and calibration, the ensembles for the appendix, and the ViT
  head's local complexity maps.
- Feature families may need a ViT's patch tokens (`--tokens`); a family given by its Python path now loads in the
  worker processes too.
- `reproduce/` regenerates Table 4 with the ViT ranker and the ensemble, the accuracy by Elo gap as a figure, the
  methods' differences, the ensembles for the appendix, the learning curves and calibration, and the local complexity
  maps.
- In Python, `EvaluateSettings` takes the dataset, the scores and the output first; the feature table is one of three
  methods, given by name.
- The ceiling for SROCC is the square root of the Elo scores' split-half reliability, no longer the reliability
  itself: the reliability is the correlation expected between two runs of the annotation, while a method's scores are
  compared with one run and carry none of its noise. The split-half table gives the reliability and the ceiling;
  `metrics.tsv` gives both (`elo_reliability`, `srocc_ceiling`). On the dataset the ceiling moves from 0.793 to
  0.891.
- Four new feature families, the baselines, arranged by the information they need: `trivial` (ink coverage, the
  number of text lines), `generic` (six generic image measures: edge-orientation entropy, MSSC per scale, lossless
  compressibility, feature congestion, colourfulness, patch-codebook variety), `kyle_davidson` (Kyle-Davidson et
  al.'s clutter, symmetry and entropy as they publish them) and `ink` (coverage, lacunarity, fragmentation, chromatic
  share and hue classes, with scales in line heights); and `mean_token`, the mean of a page's patch tokens. The image
  families share one ink layer and one normalised folio crop, prepared once per page. Needs scikit-image, now a
  declared dependency.
- `lvc codebook` learns the generic measures' patch codebook from a segmentation run's pages; the package comes with
  the one learned on the dataset's pages.
- `lvc evaluate` takes some columns of a feature table (`--columns`, by name or by their measure's prefix), and can
  leave out the columns that only restate a reference measure (`--restating.reference`), recording the check in
  `restating.tsv`.
- `lvc report` writes the baselines' tables and figures: by Elo gap (Table 3's successor), their differences from the
  geometry ranker and the ensemble, the measures defined, what each level of information buys (with each measure
  alone, too), and the generic measures' correlations with ink coverage. `reproduce/` regenerates them, and Table 4
  has a column for the generic image baseline.
