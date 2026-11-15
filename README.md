# layout-visual-complexity

Quantifying visual complexity of medieval manuscript page layouts.

## Install

With Python 3.12 or newer:

```
pip install layout-visual-complexity
```

Segmenting pages needs the `segment` extra, which brings Kraken, PyTorch and ultralytics:

```
pip install 'layout-visual-complexity[segment]'
```

ultralytics, which runs the YOLO models, is licensed under the AGPL-3.0; the rest of the package doesn't need it.

The ViT ranker needs the `vit` extra, which brings PyTorch and Hugging Face's transformers:

```
pip install 'layout-visual-complexity[vit]'
```

For development, from a clone:

```
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,segment,vit]'
```

## Use

The command is `lvc`. A study of pairwise judgments ("which of these two pages is more visually complex?") goes
through three steps:

```
lvc judgments import --source 10.5281/zenodo.<record> --output build/dataset
lvc judgments rank --config ranking.yaml
lvc judgments report --config report.yaml
```

`import` reads a dataset of judgments in its published structure, from a folder or from its Zenodo record by DOI,
checks it, and copies it into the working folder; `rank` computes Elo scores over seeded permutations of the
judgments' order, and Bradley-Terry scores; `report` writes the reliability tables, position bias by annotator, the
comparison graph and the score distributions as bands, with thumbnails of chosen pages.

Settings come from YAML files, and any of them can be given as an option instead: `--config` can be repeated, later
files override earlier ones, and options override files. `lvc judgments rank --help` lists every setting. Each run
writes its resolved configuration (`config.yaml`) and a `manifest.json` with its inputs and their checksums, the
versions it used, and when it ran. `--jobs` sets the number of worker processes (all cores by default); results don't
depend on it.

Records, manifests and images are fetched politely: from several servers at once, but from each with at most two
requests in flight and half a second between their starts, slowing down when a server asks, with retries. Set
`LVC_CONTACT` (see `.env.example`) to tell the servers who to contact about the requests.

### Scoring and comparing pages

```
lvc score SOURCES… --output build/score/name [--models FOLDER] [--maps] [--profiles] [--families colour] [--jobs N] [--device auto|cpu|mps]
lvc compare FIRST SECOND [--first_page 12r] [--second_page CANVAS_ID]
```

`score` takes the sources `segment` takes (below) and streams each page through every stage at once: it is fetched,
cropped to its folio, segmented into regions and lines, measured, and its folio's patch tokens are scored by the ViT
head; its image leaves the bounded download cache once it is done, and the tokens are never stored. It needs both
extras. The run's folder keeps `scores.tsv` (each page's id, document, label, canvas and place in its document; the
geometry ranker's, the ViT ranker's and their ensemble's z-scores; and the ensemble's expected Elo score), the layout
store, the features (`features.parquet`), the report and the run's record, with every model's checksum. On request,
`--maps` keeps the head's patch scores as small float16 maps (`maps/<document>.npz`), and `--profiles` draws each
document's profile (its pages coloured by score, the scores along the document, its most and least complex pages)
and the overview of the documents (each one's mean score against the spread of its pages'). `--families` measures
more families beside the ranker's, which don't enter the scores (`colour`, to describe the pages' colour). A rerun
resumes; the scores don't depend on the number of workers.

The package comes with its rankers (`--models` takes others, made by `lvc train --ensemble`): the geometry ranker
and the ViT head, each trained on the dataset's 397 pages and all their judgments, and their ensemble with equal
weights. A page's expected Elo score is the regression of the dataset's Elo scores on the ensemble's out-of-fold
scores, so that it reads on the Elo scale with the spread a prediction has.

`compare` scores two pages (images, image URLs, IIIF image services, or a page of a IIIF manifest by its place, its
label or its canvas) and prints the probability that the first is judged more complex than the second, by the
ensemble and by each ranker: a sigmoid of the difference of their z-scores, by a scale fitted on the evaluations'
out-of-fold scores, so that it holds on new pages. It is symmetric: p(a, b) = 1 − p(b, a). The pages are scored into
the user cache, where they stay for later comparisons.

```
lvc models [--config score.yaml]
lvc cache [--clear images|manifests|tokens|…|all]
```

`models` lists what a scoring run uses (the regions model, Kraken's model, the ViT backbone at its revision, the
rankers), where each is, and whether it is there and the one expected; the regions model can't come with the package
(PyPI refuses files over 100 MB), and is put in `models/` until it is published. `cache` lists the user cache's parts
(page images, manifests, tokens, thumbnails, …) with their sizes, and clears them by kind.

### Segmenting pages

```
lvc segment SOURCES… --config segment.yaml --output build/segment/name [--jobs N] [--device auto|cpu|mps] [--alto] [--limit N]
```

finds on each page the leaf (the folio crop), its regions and its text lines, and keeps them in a layout store.

- **Sources**: IIIF manifests (Presentation 2 or 3, by URL or file), IIIF image services, image URLs, local images,
  folders of images, list files (one source per line, optionally followed by a tab and a selection), and a dataset's
  `images.tsv`. A selection takes pages by index (`5-12`) or canvas label (`10r-20v`); `inputs.label_filter`, a
  regular expression, keeps only some pages, such as the leaves (`\d+[a-z]*[rv]`).
- **Page ids**: a dataset's page table keeps its own; otherwise `<document>_<folio>`, the document being the
  manuscript's shelfmark or the folder's name in lower case without spaces or punctuation (`ig28`), the folio the
  canvas label in lower case without spaces or leading zeros (`ig28_115r`).
- **Images** are fetched at a 2,000 px long edge, or at their own size when smaller, into a cache of about 2 GB from
  which the oldest leave first (`images.long_edge`, `images.cache_gb`).
- **The folio crop**: Otsu's threshold on lightness; the largest bright region, fitted with a quadrilateral; the
  whole image when that quadrilateral covers under half or over 98% of the image, as when the image is cut to the
  leaf. For pages photographed on a bright ground, a YOLO folio model (`folio_box.model`) can seed it: the region
  that overlaps its box most, and the box when Otsu's quadrilateral is implausible.
- **Regions**: a YOLO model (`regions.model`), at the size it was trained at, keeping as regions the classes that
  `regions.classes` matches (its zones by default), and as YOLO's lines those `regions.line_classes` matches (its
  line classes: headings, interlinear glosses, music).
- **Lines**: Kraken's baseline segmenter (its own `blla` model by default), keeping up to 500 lines a page
  (`lines.max_lines`; Kraken alone keeps 200); each line is assigned to the region its boundary overlaps most.

Every step runs at once in worker processes, each page moving on as soon as it is ready. `--device auto` runs the
YOLO models and Kraken's network on Apple's GPU when there is one, and Kraken's vectorisation on the CPU's cores; the
CPU and the GPU give the same layout. Workers are capped to fit in the memory available. Running the same command
again skips the pages already done with the same settings and models, and retries those that failed.

The output folder holds `layout/<document>.parquet`, one row per element of a page (`page`, `kind`: `folio`,
`folio_box` with a folio model, `region`, `yolo_line` or `line`, `class`, `polygon`, `baseline`, `confidence`,
`region`, `id`), with every coordinate in the full image's pixels; `pages.parquet`, one row per page (its source,
canvas, image service, full and fetched size, the folio crop's method and, with a folio model, its IoU with the box,
its numbers of regions, YOLO's lines and Kraken's lines, each step's seconds, its status); `report.md` (pages done and
failed, each step's pace, how busy the cores and the GPU were, peak memory, the store's size, the pages cropped to the
whole image or the box); `alto/` with `--alto`, one
ALTO 4 file per page; and the run's `config.yaml` and `manifest.json`, with the models' checksums.

### Features of pages

```
lvc features --layout build/segment/name --families geometry,layout --output build/features/name [--jobs N]
```

measures feature families on a segmentation run's pages, into `features.parquet`: one row per page, `page` first,
then one column per feature, named `Family__Feature__Flavour` (`AreaFeature__LineSimilarity__dice_median`).

- **`geometry`**: 83 features of the folio's written space and of its text zones' lines, in nine groups: written
  space, rectangularity, line similarity, graphic zones and drop capitals (`AreaFeature__…`); line length, interlinear
  distance, the zigzag of line ends, and parallelism (`LineFeature__…`). Line features are measured per text zone
  (`MainZone`, `MarginTextZone`) on its lines, at least three once those shorter than 7.5% of the zone's width are
  left out, and taken to the page by the median. Lengths and offsets are relative to the zone's width, interlinear
  distances to the folio's diagonal, areas to the folio's area.
- **`layout`**: 22 features adapting Kyle–Davidson's dimensions of scene complexity to the page: clutter (zones, zone
  types, the regions model's line types), order (mirror symmetry about the text block's axes, the regularity of the
  rows and columns, the balance of the written mass), the entropy of the zone types' areas, and blank space (its
  share, the largest empty rectangle, the margins). The order measures run on the written mass (`_mass`, from the
  segmentation) and on the ink (`_ink`, Otsu's dark pixels of the folio, from the image alone).
- **`trivial`**: ink coverage (the ink layer's share of the folio) and the number of text lines.
- **`generic`**: six generic image measures, needing neither segmentation nor judgments, on the folio crop at 1,400
  pixels on its long edge with its illumination flattened: edge-orientation entropy, MSSC at each of seven scales,
  lossless compressibility (the quantised lightness's raw size over its PNG's), feature congestion (colour, contrast
  and orientation), colourfulness, and patch-codebook variety (how many kinds of patch a page has, and the entropy of
  their use, against one codebook of patch colour histograms: the package's, learned on the dataset's pages, or one
  of your own from `lvc codebook`, given as `parameters.generic.codebook`).
- **`kyle_davidson`**: Kyle-Davidson et al.'s (2023) clutter (the regions left by a normalised cut of SLIC
  superpixels), mirror symmetry in patches of three sizes, and the entropy of each HSV channel over a Gaussian pyramid,
  on the folio crop at 700 pixels, as they measure scenes.
- **`ink`**: ink distribution, on one ink layer (Sauvola's threshold of the flattened lightness, specks removed), with
  scales in line heights estimated from the page (the text lines' spacing, else the period of the ink's rows): coverage,
  lacunarity at 1 to 16 line heights, fragmentation, the share of chromatic ink against the parchment's own colour,
  and the shares of red, green and blue ink. Ink of hues between 60° and 110° (yellow, gold and olive), where brown
  writing reaches the chroma threshold, isn't counted as chromatic.
- **`mean_token`**: the mean of a page's patch tokens from a token set (`--tokens`), for a linear model on them.
- **`line_texture`**: the article's 38 line-texture measures (its Appendix B), named as the article names them
  (`LineTextureFeature__LineSuperpixels__…`): for each text line, the vertical Sobel responses of the gamma-corrected
  red, green and blue channels, partitioned into SLIC superpixels, and their number, areas, positions, channel
  statistics and position–response correlations, averaged over the folio's lines. They describe texture, not hue.
- **`colour`**: real colour, on the ink layer with the parchment made neutral: the chromatic share of the ink (hues
  60°–110° left out, as in `ink`), the shares of red, green and blue ink and the number of classes present, the
  entropy of the chromatic ink's hues, its median chroma and spread, its chromatic share within each kind of zone
  (main text, margins, numbering and running titles, decoration, heading lines, the rest), the share of rubric lines
  and the spread of the lines' chromatic shares, and the number of coloured initials. The ink within the libraries'
  stamps is left out.
- **Your own family**: a subclass of `layout_visual_complexity.features.contract.Family`, named by its Python path
  (`--families geometry,mypackage.features.Darkness`). It names its columns, declares its scope (page, region or
  line) and what it needs (the image, the folio crop, the regions, the lines, a ViT's patch tokens from a token set
  given with `--tokens`, the ink layer), says how its rows are aggregated to the page, and has a dataclass of
  settings, given under `parameters` (`--parameters "{layout: {resolution: 128}}"`).

Pages are measured in worker processes; what the families need (the ink layer among it) is prepared once per page
and shared among them.
Images, when a family needs them, are found as the segmentation run fetched them, from the download cache. The
table's metadata records each column's family and the family's settings and their hash. A rerun keeps the pages
measured before with the same families, settings, layout and image. The run's `report.md` gives the seconds per page
of each family, the pages per second, how busy the cores were, and the memory used.

```
lvc codebook --layout build/segment/name --output build/codebook/name [--codewords 32] [--jobs N]
```

learns the generic family's patch codebook from a segmentation run's pages, without any judgment: k-means, seeded,
over the colour histograms of patches drawn from every page. The codebook doesn't depend on the number of workers.

### Patch tokens of a vision transformer

```
lvc embed --layout build/segment/name --output build/embed/name [--tokens.backbone facebook/dinov2-small] [--tokens.crop folio|page] [--device auto|cpu|mps]
```

gives each page's patch tokens from a vision transformer (ViT), for the ViT ranker: any `transformers` model with
patch tokens, from its publisher's repository at a pinned revision (`--tokens.revision`); DINOv2 ViT-S/14 by default.
The page's folio crop (or the whole image) is resized to a long edge of 518 pixels (`--tokens.long_edge`), each side
rounded to whole patches, so 37 patches of 14 pixels along it; the tokens are the last block's (`--tokens.layer`)
after the model's final layer norm, without the class token. They are kept as float16 arrays, one file per page, in
the user cache (`~/.cache/layout-visual-complexity/tokens/`, under a key of the backbone, revision, block, input
size and crop); the run's folder holds the token set (`tokens.json`, `pages.tsv`). The model runs on Apple's GPU or
CUDA where there is one. A rerun keeps the pages already embedded from the same image and crop.

### Evaluating a method, and the trained model

```
lvc evaluate --features build/features/name/features.parquet --dataset build/dataset --scores build/ranking/scores.tsv --output build/evaluate/name [--families geometry]
lvc evaluate --tokens build/embed/name --dataset … --scores … --output build/evaluate/vit [--jobs N]
lvc evaluate --ensemble "{geometry: build/evaluate/geometry, vit: build/evaluate/vit}" --dataset … --scores … --output build/evaluate/ensemble [--fitted_weights]
```

evaluates a method, cross-validated on folds by manuscript (10 × 10 by default):

- **the pairwise logistic model** on a feature table: all its columns, some families' (`--families`), or some
  columns (`--columns`, by name or by their measure's prefix, `Generic__Mssc`). Each feature is clipped to the
  training pages' 1st and 99th percentiles, in training and when the model scores pages, so a page unlike any
  training page (a diagram, a page of three lines) scores no further out than the training pages' extremes would. With `--restating.reference`, a
  column whose Spearman correlation with that reference column reaches `--restating.threshold` (0.9) in absolute
  value, and which adds nothing to it on the same folds, is left out (`restating.tsv` says which);
- **the local ViT head** on a token set: a small network (the tokens' width, 128, 1, with a ReLU) gives each patch a
  value, softplus makes it a non-negative score, and the page's score is the mean of its patches'; it is trained with
  RankNet's logistic loss on each fold's training judgments (180 epochs of Adam, learning rate 0.002, weight decay
  0.001, `--head.*`), the ViT frozen. Folds train in worker processes on the CPU, each from a seeded start, so the
  results don't depend on `--jobs`;
- **an ensemble** of earlier evaluations on the same folds: their z-scored page scores combined fold by fold, with
  equal weights, the weights given (`--weights`), or weights fitted on each fold's training judgments
  (`--fitted_weights`).

It writes the out-of-fold page scores (`scores.tsv`); accuracy by Elo gap and overall beside the consistency of the
same test judgments with the Elo scores (`by_elo_gap.tsv`); SROCC beside its ceiling, the square root of the Elo
scores' split-half reliability (`--ceiling`), log-loss, and within and across manuscripts the accuracy on the test
judgments between two pages of one manuscript and on those between two manuscripts, apart, and SROCC with each
manuscript's mean removed from both scores (`metrics.tsv`); calibration (`calibration.tsv`); each fold's results (`folds.tsv`), hits (`hits.tsv`) and
z-scores (`fold_scores.tsv`), so that evaluations can be compared on the same resamples and combined; with 95%
intervals from resamples of the manuscripts. With `--learning_curves.shares`, it also trains the method on shares of
each fold's training judgments and images (`learning_curve.tsv`). It then trains the method on every judgment and
keeps it, with what its inputs must be: `model.json` refuses features computed with other settings, `head.npz`
tokens made with another backbone, revision, block, input size or crop; `ensemble.json` names its members and
weights.

```
lvc train --tokens build/embed/name --dataset build/dataset --output build/train/name
```

trains the pairwise model (`--features`) or the head (`--tokens`) on every judgment, without the evaluation, and writes
its training pages' z-scores (`scores.tsv`); `--leave_out` leaves manuscripts out, their pages and every judgment
touching them, so that their pages can be scored out of sample.

```
lvc train --ensemble "{geometry: build/train/geometry, vit: build/train/vit_head}" --calibration "{geometry: build/evaluate/geometry, vit: build/evaluate/vit, ensemble: build/evaluate/ensemble}" --dataset build/dataset --output build/train/models
```

makes the rankers `lvc score` and `lvc compare` use: the members' files beside `ensemble.json`, which holds their
weights (equal by default), the spread of their weighted sum over the training pages, the scales of win
probabilities fitted on the evaluations' out-of-fold scores (`--calibration`; on the training judgments without it),
with a table comparing their calibration to the training fit's, and the Elo scale. A page whose out-of-fold z-score
lies beyond 5 in absolute value (`--fit_bound`) is left out of those fits, so one page far from the others can't bend
them. The
package's head scores a page with numpy alone (`layout_visual_complexity.evaluation.vit_head.shipped_head()`), and its
patch scores make a map of where on the page the head finds complexity.

`lvc report` writes the tables and figures of the features and of the methods evaluated on them: accuracy by Elo gap
(as a table and a figure), the differences between methods on the same resamples, learning curves and calibration,
the local complexity maps, the features' table, each feature's correlation with the Elo scores, and the features
drawn on pages; the baselines' (by Elo gap, by the information they need, the measures defined, the generic
measures' correlations with ink coverage); the colour level's (by Elo gap with what colour adds to geometry, the
measures defined, ink colour drawn on pages); and every method within and across manuscripts, from the hits and
scores each evaluation keeps. In the methods' tables and figures, the Reference is one limit measured for accuracy
and for ρ: the consistency of the same test judgments with the final Elo scores, and the ceiling for SROCC. From a
`lvc score` run and a table of the collection's catalogue facts, it also draws the manuscripts' profiles (in two
designs), the strips of all of them, the overview (in two designs), the collection's table and the readings of the
overview checked against the catalogue (`manuscript_profiles`, `manuscript_profile_strips`, `manuscript_strips`,
`manuscript_overview`, `manuscript_overview_quadrants`, `manuscripts_table`, `manuscript_readings`); `--min_lines`
leaves out blank or nearly blank pages.

## The dataset

A dataset is a folder of four files, which `import` checks before anything else uses them:

| File | One row per | Columns |
|---|---|---|
| `judgments.tsv` | judgment | `judgment`, `session`, `annotator`, `campaign`, `date`, `left`, `right`, `chosen`, `response_ms` |
| `images.tsv` | page | `image`, `manuscript`, `folio`, `canvas`, `image_url`, `image_service`, `width`, `height`, `work`, `text`, `text_language` |
| `manuscripts.tsv` | manuscript | `manuscript`, `shelfmark`, `library`, `city`, `country`, `title`, `date`, `date_not_before`, `date_not_after`, `date_edtf`, `place_of_origin`, `material`, `height_mm`, `width_mm`, `languages`, `texts`, `category`, `secondary_categories`, `pages`, `canvases`, `iiif_manifest`, `catalogue_record`, `mirabile_id`, `biblissima_id`, `hscensus_id`, `licence` |
| `README.md` | | what the dataset is, and each column |

Every judgment names two pages of `images.tsv`, every page a manuscript of `manuscripts.tsv`; every page has its IIIF
canvas and a link to its image; dates are in EDTF (ISO 8601-2), languages in ISO 639-3. A new study writes its
judgments in this structure.

## Reproducing the article

The tables and figures of the article are regenerated with:

```
make -C reproduce all
```

The dataset isn't in this repository: `reproduce/configs/dataset.yaml` names where it comes from. The dataset and the
scores go to `build/`, the tables (TSV and LaTeX) and figures (SVG, PNG and LaTeX) to `results/`. The features and
the tables and figures made from them need the dataset's segmentation (`segment-dataset`), which they make first if
it's missing; the ViT ranker needs the `vit` extra, and downloads DINOv2 ViT-S/14 once. Its evaluation, with its
learning curves (on the folds of the first two repeats), takes about 25 minutes on an Apple M4 Pro. The targets:

| Target | What it makes |
|---|---|
| `dataset` | `build/dataset/`: the dataset, checked |
| `ranking` | `build/ranking/`: Elo scores over 1,000 permutations, and Bradley-Terry |
| `table-dataset-categories` | Table 1: the dataset by text category |
| `table-reliability` | Table 2: reliability by Elo gap, with the split-half reliability and the ceiling for SROCC as its last row, and position bias; the consistency reference per fold, with its intervals |
| `figure-position-bias-funnel` | Position bias by annotator, within the bounds of chance |
| `figure-comparison-graph` | Figure 2: the comparison graph, coloured by batch and by Elo score |
| `figure-elo-score-bands` | Figure 3: the distributions of the Elo scores, as bands |
| `segment-dataset` | `build/segment/dataset/`: the dataset's pages segmented (not part of `all`: it needs the `segment` extra, the models in `models/` and the page images from their servers) |
| `features-dataset` | `build/features/dataset/`: the geometric and Layout features of the dataset's pages, from their segmentation, the baselines' and the colour level's |
| `evaluate-geometry`, `evaluate-geometry-layout` | `build/evaluate/`: the geometry ranker, without and with the Layout features |
| `table-geometry-by-elo-gap` | Table 4's geometry column: accuracy by Elo gap and SROCC, with and without Layout, beside the Reference |
| `table-geometry-feature-hierarchy` | Appendix A's table: the features, with their code names |
| `figure-geometry-feature-correlations` | Figure 9: each feature's correlation with the Elo scores |
| `figure-geometry-features-on-page`, `figure-geometry-features-on-schematic-page` | The geometric features drawn on pages of the dataset, and on a made-up page |
| `figure-geometry-feature-families` | The plate of the features, with their code names |
| `embed-dataset` | `build/embed/dataset/`: the dataset's pages' patch tokens from DINOv2 ViT-S/14, the folio crop |
| `evaluate-vit`, `evaluate-ensembles` | `build/evaluate/`: the ViT ranker; its ensembles with the geometry ranker (without and with Layout, with equal and fitted weights) |
| `head-dataset`, `package-head` | `build/train/vit_head/`: the ViT head trained on every judgment; copied into the package |
| `train-geometry`, `models-dataset`, `package-models` | `build/train/geometry/`, `build/train/models/`: the geometry ranker trained on every judgment, and the rankers that come with the package (the two and their ensemble, with calibrated scales and the Elo scale); copied into the package |
| `table-methods-by-elo-gap` | Table 4: the generic image baseline, real colour, the geometry ranker, the ViT ranker and their ensemble by Elo gap, beside the Reference |
| `figure-accuracy-by-elo-gap` | The same as a figure, with each method's interval, and as a table |
| `table-paired-differences` | The methods' differences in accuracy and SROCC, on the same resamples |
| `table-ensembles-by-elo-gap` | For the appendix: the ensembles, without and with Layout, with equal and fitted weights, and with real colour |
| `figure-learning-curves-and-calibration` | Accuracy as the training judgments and images grow, and calibration |
| `figure-local-complexity-maps` | The ViT head's local complexity maps on three pages |
| `codebook-dataset`, `package-codebook` | `build/codebook/dataset/`: the generic measures' patch codebook, learned on the dataset's pages; copied into the package |
| `evaluate-baselines`, `evaluate-measures` | `build/evaluate/`: the baselines (the trivial measures, the generic image measures, Kyle–Davidson, ink distribution, the mean token), each family together and each measure alone |
| `table-baselines-by-elo-gap` | Table 3's successor: the baselines by Elo gap, beside the Reference |
| `table-baseline-measures` | The baseline measures defined, with their parameters and sources |
| `table-baseline-differences` | The baselines' differences from the geometry ranker and the ensemble |
| `figure-baselines-by-information`, `figure-baselines-by-information-each-measure` | What each level of information buys: the methods by the information they need, combined, and with each measure alone |
| `figure-generic-measure-correlations` | The generic image measures' correlations with each other and with ink coverage |
| `evaluate-colour`, `evaluate-colour-measures` | `build/evaluate/`: line texture and real colour alone, each with the geometric features in one model, the ensemble with colour, and each colour measure alone |
| `table-colour-by-elo-gap` | Appendix B's successor: line texture, real colour and geometry with real colour by Elo gap, within manuscripts, and what colour adds to geometry |
| `table-colour-differences` | What colour adds to geometry and to the ensemble, overall and within manuscripts |
| `table-line-texture-families`, `table-colour-measures` | Appendix B's table of the line-texture measures; the real colour measures defined |
| `table-within-and-across-manuscripts` | Every method's accuracy across and within manuscripts, and its SROCC overall and within |
| `figure-ink-colour-on-page` | Ink colour on three pages, drawn by hue class beside the classes' shares |
| `models-without-crux` | `build/train/models_without_crux/`: the rankers trained without the four of Crux of Telč's manuscripts that gave the dataset pages, for the case study |
| `score-crux` | `build/score/crux/`: Crux's 23 manuscripts' 14,138 leaf pages scored out of sample (not part of `all`: it needs both extras, the models in `models/`, the network, and three to four hours on an Apple M4 Pro) |
| `figure-manuscript-profiles` | Figures 4 and 5: the profiles of I D 18 and XI C 1, in the package's design and in the article's |
| `figure-manuscript-strips` | For the appendix: the strips of all 23 manuscripts |
| `figure-crux-overview` | Figure 6: the overview of the 23 manuscripts, in the package's design and in the article's |
| `table-crux-manuscripts`, `table-crux-readings` | The 23 manuscripts' table, the catalogue's facts beside the scores; the article's readings of the overview, checked |

The dataset leaves out the sessions of a public exhibition, whose annotators were mostly not experts, and the
calibration judgments; so its numbers differ from the article's, which kept them. Crux of Telč's catalogue facts
(`reproduce/crux_manuscripts.tsv`) come from M. Dragoun, A. Ebersonová and L. Doležalová, *Středověké knihovny
augustiniánských kanonií v Třeboni a Borovanech* (Praha: Scriptorium, 2021).

## Tests

```
.venv/bin/pytest
```

Tests marked `requires_models` run the segmentation models on a synthetic page; they are skipped without the
`segment` extra or without the regions model in `models/` (`medieval-yolov11x.pt`; `folio_yolo.pt` for the folio
model's test). One runs DINOv2 ViT-S/14 with the package's head, skipped without the `vit` extra or without the
backbone in the Hugging Face cache; the other ViT tests use a tiny random ViT and download nothing. The ALTO output is
checked against the ALTO 4.4 schema when `LVC_ALTO_SCHEMA` names the schema's file.

## Licence

See [LICENSE](LICENSE). The page images shown in the figures are the libraries' own, under their terms: those from
Manuscriptorium under CC BY-NC 4.0.
