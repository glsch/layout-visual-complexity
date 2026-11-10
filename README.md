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
lvc judgments import --config dataset.yaml
lvc judgments rank --config ranking.yaml
lvc judgments report --config report.yaml
```

`import` turns the annotation app's export into an anonymised dataset; `rank` computes Elo scores over seeded
permutations of the judgments' order, and Bradley-Terry scores; `report` writes the reliability tables, the comparison
graph and the score distributions.

Settings come from YAML files, and any of them can be given as an option instead: `--config` can be repeated, later
files override earlier ones, and options override files. `lvc judgments rank --help` lists every setting. Each run
writes its resolved configuration (`config.yaml`) and a `manifest.json` with its inputs and their checksums, the
versions it used, and when it ran. `--jobs` sets the number of worker processes (all cores by default); results don't
depend on it.

Manifests and images are fetched politely: one request at a time per server, with retries. Set `LVC_CONTACT` (see
`.env.example`) to tell the servers who to contact about the requests.

## Reproducing the article

The tables and figures of the article are regenerated from the data with:

```
make -C reproduce all
```

The data aren't in this repository: `reproduce/configs/` names the files it expects under `data/`. The dataset and the
scores go to `build/`, the tables (TSV and LaTeX) and figures (SVG, PNG and LaTeX) to `results/`. The targets:

| Target | What it makes |
|---|---|
| `dataset` | `build/dataset/`: judgments, images and manuscripts as TSV, with a README |
| `ranking` | `build/ranking/`: Elo scores over 1,000 permutations, and Bradley-Terry |
| `table-dataset-categories` | Table 1: the dataset by text category |
| `table-reliability` | Table 2: reliability by Elo gap, with position bias; the consistency reference per fold; the split-half ceiling |
| `figure-comparison-graph` | Figure 2: the comparison graph |
| `figure-elo-violins` | Figure 3: the distributions of the Elo scores |
| `all-sources` | All of the above with the exhibition sessions kept, as the article computed them, into `build/all_sources/` |

The reference dataset leaves out the sessions of the public exhibition, whose annotators were mostly not experts, and
the calibration judgments; so its numbers differ from the article's. `make -C reproduce all-sources` keeps them.

## Tests

```
.venv/bin/pytest
```

## Licence

See [LICENSE](LICENSE). The page images shown in the figures are the libraries' own, under their terms: those from
Manuscriptorium under CC BY-NC 4.0.
