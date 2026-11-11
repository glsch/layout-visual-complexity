# layout-visual-complexity

Quantifying visual complexity of medieval manuscript page layouts.

## Install

With Python 3.12 or newer:

```
pip install layout-visual-complexity
```

For development, from a clone:

```
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
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

Records, manifests and images are fetched politely: one request at a time per server, with retries. Set
`LVC_CONTACT` (see `.env.example`) to tell the servers who to contact about the requests.

## The dataset

A dataset is a folder of four files, which `import` checks before anything else uses them:

| File | One row per | Columns |
|---|---|---|
| `judgments.tsv` | judgment | `judgment`, `session`, `annotator`, `campaign`, `date`, `left`, `right`, `chosen`, `response_ms` |
| `images.tsv` | page | `image`, `manuscript`, `folio`, `canvas`, `image_url`, `image_service`, `width`, `height`, `work`, `text`, `text_language` |
| `manuscripts.tsv` | manuscript | `manuscript`, `shelfmark`, `library`, `city`, `country`, `title`, `date`, `date_not_before`, `date_not_after`, `date_edtf`, `place_of_origin`, `material`, `height_mm`, `width_mm`, `languages`, `texts`, `category`, `secondary_categories`, `pages`, `canvases`, `iiif_manifest`, `catalogue_record`, `mirabile_id`, `biblissima_id`, `census_id`, `licence` |
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

The dataset leaves out the sessions of a public exhibition, whose annotators were mostly not experts, and the
calibration judgments; so its numbers differ from the article's, which kept them.

## Tests

```
.venv/bin/pytest
```

## Licence

See [LICENSE](LICENSE). The page images shown in the figures are the libraries' own, under their terms: those from
Manuscriptorium under CC BY-NC 4.0.
