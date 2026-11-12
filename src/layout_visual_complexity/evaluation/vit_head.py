"""The local ViT head (after Celona et al.): a small network that gives each patch token of a frozen ViT a
non-negative score, the page's score being the mean of its patches'.

An MLP (the tokens' width, `hidden`, 1) with a ReLU gives each patch a value, softplus makes it a score. The tokens are
scaled to unit length first. The head is trained with RankNet's logistic loss on the difference between the two pages'
scores of each training judgment: full-batch Adam with weight decay, for a fixed number of epochs. Training runs on the
CPU in one thread from a seeded start, so the weights depend on the seed alone, not on the number of workers or the
machine's load. Folds train in worker processes that share one memory-mapped copy of the tokens.

Scoring needs numpy only: a trained head is kept as its weights in an `.npz` file, with what its tokens must be
(backbone, revision, block, input size, crop) and the settings it was trained with, and it refuses tokens made
otherwise.
"""

import json
import math
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from layout_visual_complexity.evaluation import metrics
from layout_visual_complexity.evaluation.folds import Fold
from layout_visual_complexity.features.vit import IDENTITY, TokenGrid, TokenMismatch, TokenSet
from layout_visual_complexity.logs import logger
from layout_visual_complexity.ranking.reliability import other_image

SHIPPED = "dinov2_vits14.npz"
# Memory a training worker adds beside the shared tokens, per token, in bytes: the hidden layer's activations and
# their gradients (float32), with some room.
WORKER_BYTES_PER_TOKEN = 128 * 4 * 5
WORKER_BASE_BYTES = 400 * 2**20


@dataclass(frozen=True)
class HeadSettings:
    """How the local ViT head is trained.

    Args:
        hidden: The width of its hidden layer.
        epochs: Full passes over the training judgments, one Adam step each.
        learning_rate: Adam's learning rate.
        weight_decay: Adam's weight decay (an L2 penalty added to the gradient).
        unit_length: Scale each token to unit length before the head.
        seed: The seed of the initial weights; each fold's start is drawn from it and the fold's number.
    """

    hidden: int = 128
    epochs: int = 180
    learning_rate: float = 2e-3
    weight_decay: float = 1e-3
    unit_length: bool = True
    seed: int = 1


def fold_seed(seed: int, *numbers: int) -> int:
    """A seed for one training, drawn from the settings' seed and the fold's (and learning curve share's) numbers."""
    return int(np.random.SeedSequence([seed, *numbers]).generate_state(1)[0])


@dataclass
class HeadWeights:
    """The head's weights: `first` (dim × hidden) and its bias, `second` (hidden) and its bias."""

    first: np.ndarray
    first_bias: np.ndarray
    second: np.ndarray
    second_bias: float

    def patch_scores(self, tokens: np.ndarray) -> np.ndarray:
        """Each token's score (tokens already scaled as the head was trained on them)."""
        hidden = np.maximum(np.asarray(tokens, dtype=np.float32) @ self.first + self.first_bias, 0.0)
        return np.logaddexp(0.0, hidden @ self.second + self.second_bias)


def unit(tokens: np.ndarray) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


@dataclass
class PackedTokens:
    """Pages' tokens one after another (`values`, patches × dim, float32), page n's in rows `offsets[n]` to
    `offsets[n + 1]`."""

    values: np.ndarray
    offsets: np.ndarray
    pages: list[str]

    def page_scores(self, weights: HeadWeights) -> np.ndarray:
        """Each page's score: the mean of its patches' scores."""
        patch = weights.patch_scores(self.values)
        sums = np.add.reduceat(patch, self.offsets[:-1]) if len(patch) else np.zeros(len(self.pages))
        return sums / np.maximum(np.diff(self.offsets), 1)


def pack(token_set: TokenSet, pages: list[str], unit_length: bool, path: Path | None = None) -> PackedTokens:
    """The pages' tokens packed as float32, scaled to unit length if asked; into a memory-mapped file at `path` when
    given, so that worker processes can share them."""
    grids = [token_set.grid(page) for page in pages]
    counts = np.array([len(grid.tokens) for grid in grids])
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    dim = grids[0].tokens.shape[1] if grids else 0
    shape = (int(offsets[-1]), int(dim))
    values = np.lib.format.open_memmap(path, "w+", np.float32, shape) if path else np.empty(shape, np.float32)
    for number, grid in enumerate(grids):
        part = unit(grid.tokens) if unit_length else grid.tokens.astype(np.float32)
        values[offsets[number] : offsets[number + 1]] = part
    if path:
        values.flush()  # type: ignore[union-attr]
    return PackedTokens(values, offsets, list(pages))


# Training.


def train_head(
    packed: PackedTokens, chosen: np.ndarray, other: np.ndarray, settings: HeadSettings, seed: int
) -> tuple[HeadWeights, np.ndarray]:
    """The head trained on judgments given as the positions of their chosen and other pages in `packed`, and every
    page's score under it."""
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        generator = torch.Generator().manual_seed(seed)
        dim, hidden = packed.values.shape[1], settings.hidden

        def uniform(shape: tuple[int, ...], fan_in: int) -> Any:
            bound = 1 / math.sqrt(fan_in)  # PyTorch's own start for a linear layer
            return ((torch.rand(shape, generator=generator) * 2 - 1) * bound).requires_grad_()

        first, first_bias = uniform((dim, hidden), dim), uniform((hidden,), dim)
        second, second_bias = uniform((hidden,), hidden), uniform((1,), hidden)
        softplus = torch.nn.functional.softplus

        def page_scores(tokens: Any, counts: np.ndarray) -> Any:
            page_of_patch = torch.from_numpy(np.repeat(np.arange(len(counts)), counts))
            patch = softplus(torch.relu(tokens @ first + first_bias) @ second + second_bias)
            sizes = torch.from_numpy(np.maximum(counts, 1).astype(np.float32))
            return torch.zeros(len(counts)).index_add_(0, page_of_patch, patch) / sizes

        values, counts = np.asarray(packed.values), np.diff(packed.offsets)
        # Only the pages the judgments compare count in the loss: when they hold under half the tokens, the training
        # runs on a copy of theirs alone.
        used, positions = np.unique(np.concatenate([chosen, other]), return_inverse=True)
        if counts[used].sum() < counts.sum() / 2:
            rows = np.concatenate([np.arange(packed.offsets[page], packed.offsets[page + 1]) for page in used])
            training, training_counts = torch.from_numpy(values[rows]), counts[used]
            winners = torch.from_numpy(positions[: len(chosen)])
            losers = torch.from_numpy(positions[len(chosen) :])
        else:
            training, training_counts = torch.from_numpy(values), counts
            winners, losers = torch.from_numpy(np.asarray(chosen)), torch.from_numpy(np.asarray(other))
        optimiser = torch.optim.Adam(
            [first, first_bias, second, second_bias], lr=settings.learning_rate, weight_decay=settings.weight_decay
        )
        for _ in range(settings.epochs):
            score = page_scores(training, training_counts)
            loss = softplus(score[losers] - score[winners]).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            score = page_scores(torch.from_numpy(values), counts)
        weights = HeadWeights(
            first.detach().numpy().copy(),
            first_bias.detach().numpy().copy(),
            second.detach().numpy().copy(),
            float(second_bias.detach()[0]),
        )
        return weights, score.numpy().astype(float)
    finally:
        torch.set_num_threads(threads)


_packed: PackedTokens | None = None


def start_worker(path: str, offsets: np.ndarray, pages: list[str]) -> None:
    """Open the shared tokens once in a worker process."""
    global _packed
    _packed = PackedTokens(np.load(path, mmap_mode="c"), offsets, pages)


def train_task(chosen: np.ndarray, other: np.ndarray, settings: HeadSettings, seed: int) -> dict[str, Any]:
    """A worker's training: every page's score under the trained head, the weights, the seconds and the peak
    memory."""
    from layout_visual_complexity.pipeline.step import usage

    began = time.perf_counter()
    weights, scores = train_head(_packed, chosen, other, settings, seed)  # type: ignore[arg-type]
    return {"scores": scores, "weights": weights, "seconds": time.perf_counter() - began, **usage()}


def worker_count(jobs: int, patches: int) -> int:
    """Workers that fit in three fifths of the memory the system says is available, less 1 GB; at most `jobs` (0:
    all cores). macOS counts as available memory it would rather keep: nine workers filling all of it (1.16 GB each
    beside 14.6 GB available) made it swap."""
    from layout_visual_complexity.pipeline.monitor import cores, memory_available

    wanted = jobs or cores()
    available = memory_available()
    if available is None:
        return wanted
    each = WORKER_BASE_BYTES + WORKER_BYTES_PER_TOKEN * patches
    return max(1, min(wanted, int((0.6 * available - 2**30) // each)))


@dataclass
class HeadRanker:
    """Trains the head on folds of the judgments over a token set's pages: each fold in a worker process, all of them
    sharing one memory-mapped copy of the tokens."""

    token_set: TokenSet
    pages: list[str]
    settings: HeadSettings = field(default_factory=HeadSettings)
    jobs: int = 0
    timings: list[float] = field(default_factory=list)
    peak_memory: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.folder = tempfile.TemporaryDirectory(prefix="lvc-head-")
        path = Path(self.folder.name) / "tokens.npy"
        self.packed = pack(self.token_set, self.pages, self.settings.unit_length, path)
        self.path = path
        self.position = {page: number for number, page in enumerate(self.pages)}
        self.workers = worker_count(self.jobs, len(self.packed.values))

    def close(self) -> None:
        self.packed = PackedTokens(np.empty((0, 0), np.float32), np.zeros(1, np.int64), [])
        self.folder.cleanup()

    def positions(self, judgments: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        chosen = judgments["chosen"].map(self.position).to_numpy()
        other = other_image(judgments).map(self.position).to_numpy()
        if np.isnan(chosen.astype(float)).any() or np.isnan(other.astype(float)).any():
            raise ValueError("some judgments' pages have no tokens")
        return chosen.astype(np.int64), other.astype(np.int64)

    def train_many(self, tasks: list[tuple[np.ndarray, np.ndarray, int]]) -> list[dict[str, Any]]:
        """Train once per task (chosen, other, seed), in worker processes when there are several."""
        if self.workers <= 1 or len(tasks) <= 1:
            found = []
            for chosen, other, seed in tasks:
                began = time.perf_counter()
                weights, scores = train_head(self.packed, chosen, other, self.settings, seed)
                found.append({"scores": scores, "weights": weights, "seconds": time.perf_counter() - began})
            self.timings += [item["seconds"] for item in found]
            return found
        import multiprocessing

        arguments = (str(self.path), self.packed.offsets, self.pages)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            min(self.workers, len(tasks)), context, initializer=start_worker, initargs=arguments
        ) as pool:
            futures = [pool.submit(train_task, chosen, other, self.settings, seed) for chosen, other, seed in tasks]
            step = max(1, len(futures) // 10)
            for done, _ in enumerate(as_completed(futures), start=1):
                if done % step == 0 and done < len(futures):
                    logger.info("the head: %d of %d trainings done", done, len(futures))
            found = [future.result() for future in futures]
        self.timings += [item["seconds"] for item in found]
        self.peak_memory += [item["peak_memory"] for item in found]
        return found

    def fold_scores(self, folds: list[Fold], judgments: pd.DataFrame, share: int = 0) -> list[pd.Series | None]:
        """Each fold's raw scores of its training and test pages, the head trained on its training judgments.
        `share` numbers a learning curve's share, for the folds' seeds."""
        tasks, kept = [], []
        for number, fold in enumerate(folds):
            if not len(fold.train) or not len(fold.test):
                continue
            chosen, other = self.positions(judgments.iloc[fold.train])
            numbers = (number,) if not share else (number, share)
            tasks.append((chosen, other, fold_seed(self.settings.seed, *numbers)))
            kept.append(number)
        results = dict(zip(kept, self.train_many(tasks), strict=True))
        found: list[pd.Series | None] = []
        for number, fold in enumerate(folds):
            if number not in results:
                found.append(None)
                continue
            scores = pd.Series(results[number]["scores"], index=self.pages)
            found.append(scores.reindex(np.concatenate([fold.train_pages, fold.test_pages])))
        return found

    def train(self, judgments: pd.DataFrame, trained_on: dict[str, Any]) -> "TrainedHead":
        """The head trained on every judgment, with its scores' mean and spread over the pages and the scale of win
        probabilities."""
        chosen, other = self.positions(judgments)
        (result,) = self.train_many([(chosen, other, fold_seed(self.settings.seed))])
        raw = pd.Series(result["scores"], index=self.pages)
        mean, sd = float(raw.mean()), float(raw.std(ddof=0))
        z = (raw - mean) / (sd if sd > 0 else 1.0)
        scale = metrics.fit_scale(z[judgments["chosen"]].to_numpy() - z[other_image(judgments)].to_numpy())
        return TrainedHead(
            result["weights"], self.token_set.identity(), asdict(self.settings), mean, sd, scale, trained_on
        )


# A trained head.


@dataclass
class TrainedHead:
    """The head trained on every judgment, with what its tokens must be (`tokens`: backbone, revision, block, input
    size, crop, patch size and width), its settings, and its scores' mean and spread over the training pages, by
    which a page's score is z-scored; `scale` turns a difference of z-scores into the log-odds that the first page is
    chosen."""

    weights: HeadWeights
    tokens: dict[str, Any]
    settings: dict[str, Any]
    score_mean: float
    score_sd: float
    scale: float
    trained_on: dict[str, Any]

    def check(self, identity: dict[str, Any]) -> None:
        """Refuse tokens made otherwise than the head's."""
        differ = [name for name in IDENTITY if identity.get(name) != self.tokens.get(name)]
        if differ:
            said = ", ".join(f"{name} {identity.get(name)!r}, not {self.tokens.get(name)!r}" for name in differ)
            raise TokenMismatch(f"the tokens were made otherwise than the head's: {said}")

    def patch_scores(self, grid: TokenGrid) -> np.ndarray:
        """Each of a page's patches' score, as the head was trained to give it."""
        tokens = unit(grid.tokens) if self.settings.get("unit_length", True) else grid.tokens
        return self.weights.patch_scores(tokens)

    def scores(self, token_set: TokenSet, pages: list[str] | None = None) -> pd.Series:
        """Each page's z-scored score, after checking the tokens."""
        self.check(token_set.identity())
        pages = list(token_set.pages.index) if pages is None else pages
        raw = pd.Series([float(self.patch_scores(token_set.grid(page)).mean()) for page in pages], index=pages)
        return (raw - self.score_mean) / (self.score_sd if self.score_sd > 0 else 1.0)

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "local ViT head",
            "tokens": self.tokens,
            "settings": self.settings,
            "score_mean": self.score_mean,
            "score_sd": self.score_sd,
            "scale": self.scale,
            "trained_on": self.trained_on,
        }

    def save(self, path: Path) -> Path:
        """The weights as float32 arrays and the rest as JSON, in one `.npz` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez(
                stream,
                first=self.weights.first.astype(np.float32),
                first_bias=self.weights.first_bias.astype(np.float32),
                second=self.weights.second.astype(np.float32),
                second_bias=np.array([self.weights.second_bias], np.float32),
                metadata=np.array(json.dumps(self.metadata(), sort_keys=True)),
            )
        return path

    @classmethod
    def load(cls, path: Path | Any) -> "TrainedHead":
        with np.load(path, allow_pickle=False) as stored:
            weights = HeadWeights(
                stored["first"], stored["first_bias"], stored["second"], float(stored["second_bias"][0])
            )
            data = json.loads(str(stored["metadata"]))
        return cls(
            weights,
            data["tokens"],
            data["settings"],
            float(data["score_mean"]),
            float(data["score_sd"]),
            float(data["scale"]),
            data.get("trained_on", {}),
        )


def local_map(head: TrainedHead, grid: TokenGrid) -> np.ndarray:
    """A page's local complexity map: its patches' scores on the grid (rows × columns), the patch at row r and column
    c covering `grid.patch_boxes()[r * columns + c]` in the full image's pixels."""
    return head.patch_scores(grid).reshape(grid.rows, grid.columns)


def shipped_head() -> TrainedHead:
    """The head that comes with the package: trained on the dataset's 397 pages and all their judgments, on DINOv2
    ViT-S/14's last block at a 518-pixel long edge, the folio crop."""
    with resources.as_file(resources.files("layout_visual_complexity") / "heads" / SHIPPED) as path:
        return TrainedHead.load(path)
