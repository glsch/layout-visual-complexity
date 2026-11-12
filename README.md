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

For development, from a clone:

```
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,segment]'
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
judgments' order, and Bradley-Terry scores; `report` writes the reliability tables, the comparison graph and the
score distributions.

Settings come from YAML files, and any of them can be given as an option instead: `--config` can be repeated, later
files override earlier ones, and options override files. `lvc judgments rank --help` lists every setting. Each run
writes its resolved configuration (`config.yaml`) and a `manifest.json` with its inputs and their checksums, the
versions it used, and when it ran. `--jobs` sets the number of worker processes (all cores by default); results don't
depend on it.

Records, manifests and images are fetched politely: from several servers at once, but from each with at most two
requests in flight and half a second between their starts, slowing down when a server asks, with retries. Set
`LVC_CONTACT` (see `.env.example`) to tell the servers who to contact about the requests.

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
scores go to `build/`, the tables (TSV and LaTeX) and figures (SVG, PNG and LaTeX) to `results/`. The targets:

| Target | What it makes |
|---|---|
| `dataset` | `build/dataset/`: the dataset, checked |
| `ranking` | `build/ranking/`: Elo scores over 1,000 permutations, and Bradley-Terry |
| `table-dataset-categories` | Table 1: the dataset by text category |
| `table-reliability` | Table 2: reliability by Elo gap, with position bias; the consistency reference per fold; the split-half ceiling |
| `figure-comparison-graph` | Figure 2: the comparison graph |
| `figure-elo-violins` | Figure 3: the distributions of the Elo scores |
| `segment-dataset` | `build/segment/dataset/`: the dataset's pages segmented (not part of `all`: it needs the `segment` extra, the models in `models/` and the page images from their servers) |

The dataset leaves out the sessions of a public exhibition, whose annotators were mostly not experts, and the
calibration judgments; so its numbers differ from the article's, which kept them.

## Tests

```
.venv/bin/pytest
```

Tests marked `requires_models` run the segmentation models on a synthetic page; they are skipped without the
`segment` extra or without the regions model in `models/` (`medieval-yolov11x.pt`; `folio_yolo.pt` for the folio
model's test). The ALTO output is
checked against the ALTO 4.4 schema when `LVC_ALTO_SCHEMA` names the schema's file.

## Licence

See [LICENSE](LICENSE). The page images shown in the figures are the libraries' own, under their terms: those from
Manuscriptorium under CC BY-NC 4.0.
