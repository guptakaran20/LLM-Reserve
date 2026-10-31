"""Day-4 features: what a gateway can compute from a request before it is sent.

A request carries its prompt text, its length in tokens (`input_size`) and the target model, so
every feature comes from those three and nothing else:

- Hand-built prompt features (PROMPT_FEATURES): size and shape of the text, code markers, cue
  words and explicit length hints ("in 100 words", "one sentence", "list 10").
- log1p(input_size) and a one-hot of the model, its vocabulary fitted on train.
- MiniLM sentence embeddings, computed once per unique prompt. 128-token input is the serving
  default (the PRD latency check); a 256-token version is built to measure what truncation costs.

The label, its summaries, the source dataset and the category never become features. `check_no_leaks`
enforces that every time the matrix is built or loaded, and tests/test_features.py checks it again.

Writes data/features/:

- features_{split}.parquet: one row per completion of the original config, with META_COLUMNS (ids,
  the label, `is_degenerate`, and `prompt_row`, the row of the prompt's embedding) next to the features.
- embeddings_{128,256}.npy: float32, one L2-normalized row per prompt, in prompt_index.parquet order.
- manifest.json: feature columns, model vocabulary and embedding provenance.

The embeddings are cached: `make features` recomputes them only when the prompts, the encoder
or the input length change. It also writes docs/figures/features_validate.json, a quick check of
how much signal each feature set carries on validation (ridge regression of log length).

Run with `make features` (or `python -m llm_reserve.features`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

from llm_reserve.data import LEAKY_COLUMNS, ROOT, SPLITS, load_completions, load_prompts

FEATURES_DIR = ROOT / "data" / "features"
FIG_DIR = ROOT / "docs" / "figures"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # pinned, like the data
EMBED_LENGTHS = (128, 256)
SERVING_LENGTH = 128  # the PRD latency check: 256-token input took 13.4 ms of the 15 ms p99 budget
EMBED_DIM = 384

# Columns that may sit next to the features in the stored matrix, but are never features.
META_COLUMNS = ("prompt_id", "sample_id", "prompt_row", "model", "completion_idx", "is_degenerate", "output_tokens")
# Never features: the label and everything derived from it, the source dataset and category (error
# analysis only), ids, and the sampling parameters (fixed per model, so the one-hot carries them).
NEVER_FEATURES = (
    *LEAKY_COLUMNS, "output_tokens", "output_sizes", "is_degenerate", "completion_idx",
    "dataset", "category", "split", "sample_id", "prompt_id", "prompt_row", "temp", "top_p", "top_k",
)

# Written-out numbers that show up in length hints ("in two sentences", "list five ideas").
NUMBER_WORDS = {
    "one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
    "fifty": 50, "hundred": 100,
}
_NUM = r"(\d{1,5}|" + "|".join(NUMBER_WORDS) + r")"
_ITEMS = r"(?:bullet points|points|items|examples|ideas|tips|ways|reasons|steps|questions|names|things|facts|lines)"
# Explicit length hints: a count next to a unit. Each feature is log1p of the largest count found, 0 if none.
LENGTH_HINTS = {
    "hint_words": [rf"\b{_NUM}[\s-]+words?\b"],
    "hint_sentences": [rf"\b{_NUM}[\s-]+sentences?\b"],
    "hint_paragraphs": [rf"\b{_NUM}[\s-]+paragraphs?\b"],
    "hint_items": [rf"\b{_NUM}[\s-]+{_ITEMS}\b", rf"\blist\s+(?:the\s+)?(?:top\s+)?{_NUM}\b", rf"\btop\s+{_NUM}\b"],
}
# Cue words: 1 if the prompt contains any of the patterns.
CUES = {
    "cue_explain": r"\bexplain\w*|\bexplanation",
    "cue_list": r"\blist\b|\benumerate\b",
    "cue_write": r"\bwrite\b|\bcompose\b|\bdraft\b",
    "cue_summarize": r"\bsummar\w*|\btl;?dr\b",
    "cue_describe": r"\bdescri\w*",
    "cue_compare": r"\bcompar\w*|\bdifferences?\b|\bversus\b|\bvs\.?\s",
    "cue_rewrite": r"\btranslat\w*|\brewrite\b|\bparaphras\w*|\brephras\w*",
    "cue_extract": r"\bclassif\w*|\bextract\w*|\bidentify\b|\bcategori[sz]\w*",
    "cue_code": r"\bfunction\b|\bcode\b|\bprogram\b|\bimplement\w*|\bscript\b|\bpython\b",
    "cue_creative": r"\bstory\b|\bpoem\b|\bessay\b|\bletter\b|\barticle\b|\bblog\b",
    "cue_brief": (r"\bbrief(?:ly)?\b|\bshort\b|\bconcise(?:ly)?\b|\b(?:one|single)[- ]word\b|\byes or no\b"
                  r"|\b(?:only|just) (?:the|return|output|answer)\b"),
    "cue_detail": (r"\bdetail(?:ed|s)?\b|\bin depth\b|\bcomprehensive\w*|\belaborat\w*|\bthorough\w*"
                   r"|\bstep[- ]by[- ]step\b"),
    "starts_wh": r"^\s*(?:what|who|when|where|which|why|how|is|are|do|does|can)\b",
}
_HINT_RES = {name: [re.compile(p, re.I) for p in pats] for name, pats in LENGTH_HINTS.items()}
_CUE_RES = {name: re.compile(p, re.I) for name, p in CUES.items()}
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s", re.M)
_CODE_CHARS = set("{}()[];=<>_")

PROMPT_FEATURES = (
    "log_chars", "log_words", "log_lines", "log_bullets", "code_fences", "code_char_share", "digit_share",
    "upper_share", "non_ascii_share", "question_marks", "has_url",
    *LENGTH_HINTS, *CUES,
)


def parse_count(token: str) -> int:
    token = token.lower()
    return NUMBER_WORDS[token] if token in NUMBER_WORDS else int(token)


def length_hints(text: str) -> dict[str, float]:
    """log1p of the largest count each hint asks for; 0 when the prompt has none."""
    out = {}
    for name, patterns in _HINT_RES.items():
        counts = [parse_count(m.group(1)) for rx in patterns for m in rx.finditer(text)]
        out[name] = float(np.log1p(max(counts))) if counts else 0.0
    return out


def text_features(text: str) -> dict[str, float]:
    """Every hand-built feature of one prompt, in PROMPT_FEATURES order."""
    n = max(len(text), 1)
    f = {
        "log_chars": np.log1p(len(text)),
        "log_words": np.log1p(len(text.split())),
        "log_lines": np.log1p(text.count("\n") + 1),
        "log_bullets": np.log1p(len(_BULLET.findall(text))),
        "code_fences": float((text.count("```") + 1) // 2),  # fenced blocks, an unclosed one included
        "code_char_share": sum(c in _CODE_CHARS for c in text) / n,
        "digit_share": sum(c.isdigit() for c in text) / n,
        "upper_share": sum(c.isupper() for c in text) / n,
        "non_ascii_share": sum(ord(c) > 127 for c in text) / n,
        "question_marks": float(min(text.count("?"), 10)),
        "has_url": float("http://" in text or "https://" in text or "www." in text),
        **length_hints(text),
        **{name: float(bool(rx.search(text))) for name, rx in _CUE_RES.items()},
    }
    return {k: float(f[k]) for k in PROMPT_FEATURES}


def prompt_features(texts: pd.Series) -> pd.DataFrame:
    """One row of hand-built features per text, keeping the index of `texts`."""
    return pd.DataFrame([text_features(t) for t in texts], index=texts.index, columns=list(PROMPT_FEATURES),
                        dtype=np.float32)


def model_columns(models: Sequence[str]) -> list[str]:
    return [f"model={m}" for m in sorted(set(models))]


def feature_columns(models: Sequence[str]) -> list[str]:
    """The feature matrix columns: hand-built, log input size, then the model one-hot."""
    return [*PROMPT_FEATURES, "log_input_size", *model_columns(models)]


def check_no_leaks(columns: Sequence[str]) -> None:
    """Raise if any column is, or is named after, one that must never be a feature."""
    leaked = [c for c in columns if any(c == bad or c.startswith(f"{bad}=") for bad in NEVER_FEATURES)]
    if leaked:
        raise ValueError(f"never-feature columns in the feature matrix: {leaked}")


def build_matrix(completions: pd.DataFrame, by_prompt: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    """Join each completion to its prompt's features; return META_COLUMNS then the feature columns.

    `by_prompt` is indexed by prompt_id and holds PROMPT_FEATURES plus `prompt_row`. `models` is the
    train vocabulary; a model outside it is an error, not an all-zero one-hot.
    """
    unknown = set(completions.model.astype(str)) - set(models)
    if unknown:
        raise ValueError(f"models not in the train vocabulary: {sorted(unknown)}")
    missing = set(completions.prompt_id) - set(by_prompt.index)
    if missing:
        raise ValueError(f"{len(missing)} prompts have no features")

    joined = by_prompt.loc[completions.prompt_id.to_numpy()].reset_index(drop=True)
    model = completions.model.astype(str).to_numpy()
    onehot = {c: (model == c.removeprefix("model=")).astype(np.float32) for c in model_columns(models)}
    out = pd.concat([
        completions[[c for c in META_COLUMNS if c != "prompt_row"]].reset_index(drop=True),
        joined[["prompt_row", *PROMPT_FEATURES]],
        pd.DataFrame({"log_input_size": np.log1p(completions.input_size.to_numpy()).astype(np.float32), **onehot}),
    ], axis=1)
    cols = feature_columns(models)
    check_no_leaks(cols)
    return out[[*META_COLUMNS, *cols]]


def embed(texts: Sequence[str], max_length: int, batch_size: int = 64) -> np.ndarray:
    """MiniLM sentence embeddings (L2-normalized float32), truncating each text to `max_length` word pieces.

    Runs on CPU, the same as serving, so training and serving see the same numbers.
    """
    from sentence_transformers import SentenceTransformer  # imported here: torch is slow to load

    model = SentenceTransformer(EMBED_MODEL, revision=EMBED_REVISION, device="cpu")
    model.max_seq_length = max_length
    vectors = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    return vectors.astype(np.float32)


def truncated_share(texts: Sequence[str], max_length: int) -> float:
    """Share of texts longer than `max_length` MiniLM word pieces (special tokens included)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION)
    tok.model_max_length = 10**6  # only counting, never running the model, so no length warning
    lengths = np.array([len(ids) for ids in tok(list(texts), truncation=False)["input_ids"]])
    return float(np.mean(lengths > max_length))


def cached_embeddings(prompts: pd.DataFrame, max_length: int, manifest: dict) -> np.ndarray:
    """Embeddings for `prompts` in order, reused from data/features when nothing they depend on changed."""
    path = FEATURES_DIR / f"embeddings_{max_length}.npy"
    key = {"model": EMBED_MODEL, "revision": EMBED_REVISION, "max_length": max_length,
           "prompts": len(prompts), "prompt_ids_hash": pd.util.hash_pandas_object(prompts.prompt_id, index=False)
           .sum().item(), "texts_hash": pd.util.hash_pandas_object(prompts.prompt_text, index=False).sum().item()}
    if path.exists() and manifest.get("embeddings", {}).get(str(max_length)) == key:
        print(f"embeddings_{max_length}.npy: cached")
        return np.load(path)
    print(f"embedding {len(prompts):,} prompts at {max_length} tokens")
    vectors = embed(prompts.prompt_text.tolist(), max_length)
    np.save(path, vectors)
    manifest.setdefault("embeddings", {})[str(max_length)] = key
    return vectors


def load_features(split: str, config: str = "sanitized") -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """(X, y, meta) for one split: X holds only feature columns, y is output tokens, meta the ids.

    Join embeddings with `load_embeddings(n)[meta.prompt_row]`.
    """
    manifest = json.loads((FEATURES_DIR / "manifest.json").read_text())
    df = pd.read_parquet(FEATURES_DIR / f"features_{split}.parquet")
    if config == "sanitized":
        df = df[~df.is_degenerate].reset_index(drop=True)
    elif config != "original":
        raise ValueError("config must be 'sanitized' or 'original'")
    cols = manifest["feature_columns"]
    check_no_leaks(cols)
    return df[cols], df.output_tokens.to_numpy(), df[list(META_COLUMNS)]


def load_embeddings(max_length: int = SERVING_LENGTH) -> np.ndarray:
    return np.load(FEATURES_DIR / f"embeddings_{max_length}.npy")


# ---- Validation check: how much signal does each feature set carry? ----------------------------
# Not a model (Day 5 trains those): a ridge regression of each sample's mean log length, fitted on
# train, scored as the share of completion-level log-length variance it explains on validation.

def sample_level(X: np.ndarray, y_log: np.ndarray, sample_ids: np.ndarray):
    """Collapse completions to samples: first feature row, mean log length and completion count."""
    codes = pd.factorize(sample_ids)[0]
    _, first, counts = np.unique(codes, return_index=True, return_counts=True)
    means = np.bincount(codes, weights=y_log) / counts
    return X[first], means, counts, codes


def fit_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float):
    """Weighted ridge on standardized X with an unpenalized intercept; returns a predict function."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    y_mean = np.average(y, weights=w)
    sw = np.sqrt(w)[:, None]
    A, b = Z * sw, (y - y_mean) * sw[:, 0]
    coef = np.linalg.solve(A.T @ A + lam * np.eye(Z.shape[1]), A.T @ b)
    return lambda Xn: ((Xn - mu) / sd) @ coef + y_mean


def variance_explained(y_log: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - np.sum((y_log - pred) ** 2) / np.sum((y_log - y_log.mean()) ** 2))


def compare_feature_sets(lambdas: Sequence[float] = (1.0, 10.0, 1e2, 1e3, 1e4, 1e5)) -> dict:
    """Validation R² of log length per feature set, best ridge λ for each, plus the in-sample ceiling."""
    Xtr, ytr, mtr = load_features("train")
    Xva, yva, mva = load_features("validate")
    emb = {n: load_embeddings(n) for n in EMBED_LENGTHS}
    model_cols = [c for c in Xtr.columns if c.startswith("model=")]
    hand_cols = [*PROMPT_FEATURES, "log_input_size"]

    def block(X, meta, parts):
        mats = []
        for p in parts:
            if p == "model":
                mats.append(X[model_cols].to_numpy())
            elif p == "hand":
                mats.append(X[hand_cols].to_numpy())
            else:
                mats.append(emb[p][meta.prompt_row.to_numpy()])
        return np.hstack(mats).astype(np.float64)

    sets = {
        "model": ["model"],
        "model + hand": ["model", "hand"],
        f"model + MiniLM-{SERVING_LENGTH}": ["model", SERVING_LENGTH],
        **{f"model + hand + MiniLM-{n}": ["model", "hand", n] for n in EMBED_LENGTHS},
    }
    ytr_log, yva_log = np.log1p(ytr), np.log1p(yva)
    results = {}
    for name, parts in sets.items():
        Xs, ys, ws, _ = sample_level(block(Xtr, mtr, parts), ytr_log, mtr.sample_id.to_numpy())
        Xv, _, _, codes = sample_level(block(Xva, mva, parts), yva_log, mva.sample_id.to_numpy())
        scores = {lam: variance_explained(yva_log, fit_ridge(Xs, ys, ws, lam)(Xv)[codes]) for lam in lambdas}
        best = max(scores, key=scores.get)
        results[name] = {"r2_validate": scores[best], "lambda": best, "n_features": Xs.shape[1]}
        print(f"{name:32} R² {scores[best]:.3f}  (λ = {best:g}, {Xs.shape[1]} features)")
    # Ceiling: each validation sample's own mean, in-sample. No prompt-only predictor can beat it.
    _, means, _, codes = sample_level(np.zeros((len(yva_log), 1)), yva_log, mva.sample_id.to_numpy())
    return {"feature_sets": results, "ceiling_r2_validate": variance_explained(yva_log, means[codes])}


def dataset_recall(prompts: pd.DataFrame) -> dict:
    """How well each embedding names a prompt's source dataset: nearest train centroid, on validation.

    The dataset is never a feature, but if the text gives it away the model learns it anyway.
    """
    out = {}
    tr, va = prompts.split.eq("train").to_numpy(), prompts.split.eq("validate").to_numpy()
    for n in EMBED_LENGTHS:
        e = load_embeddings(n)
        names = sorted(prompts.dataset.unique())
        cent = np.stack([e[tr & prompts.dataset.eq(d).to_numpy()].mean(0) for d in names])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True)
        pred = np.asarray(names)[np.argmax(e[va] @ cent.T, axis=1)]
        out[str(n)] = float(np.mean(pred == prompts.dataset.to_numpy()[va]))
    return out


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = FEATURES_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    prompts = load_prompts().reset_index(drop=True)
    pd.DataFrame({"prompt_id": prompts.prompt_id, "prompt_row": np.arange(len(prompts))}).to_parquet(
        FEATURES_DIR / "prompt_index.parquet", index=False)
    by_prompt = prompt_features(prompts.prompt_text).assign(prompt_row=np.arange(len(prompts), dtype=np.int32))
    by_prompt.index = prompts.prompt_id

    # Vocabulary fitted on train only (leakage rule 4).
    models = sorted(load_completions("train", "original").model.astype(str).unique())
    rows = {}
    for split in SPLITS:
        matrix = build_matrix(load_completions(split, "original"), by_prompt, models)
        matrix.to_parquet(FEATURES_DIR / f"features_{split}.parquet", index=False)
        rows[split] = len(matrix)
        print(f"features_{split}.parquet: {len(matrix):,} rows")

    for n in EMBED_LENGTHS:
        cached_embeddings(prompts, n, manifest)
    manifest |= {"feature_columns": feature_columns(models), "models": models, "rows": rows,
                 "embed_model": EMBED_MODEL, "embed_revision": EMBED_REVISION, "serving_length": SERVING_LENGTH}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    train_prompts = prompts[prompts.split == "train"]
    summary = {
        "split": "validate",
        "n_features_hand": len(PROMPT_FEATURES) + 1, "n_models": len(models), "embed_dim": EMBED_DIM,
        # Hint coverage on train prompts: how often each hand-built hint or cue fires.
        "fire_rate_train": {c: float((by_prompt.loc[train_prompts.prompt_id, c] > 0).mean())
                            for c in [*LENGTH_HINTS, *CUES]},
        "truncated_share": {str(n): truncated_share(prompts.prompt_text.tolist(), n) for n in EMBED_LENGTHS},
        "dataset_recall": dataset_recall(prompts),
        **compare_feature_sets(),
    }
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "features_validate.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("truncated_share", "dataset_recall", "ceiling_r2_validate")}, indent=2))
    print(f"wrote {FEATURES_DIR.relative_to(ROOT)}/ and {FIG_DIR.relative_to(ROOT)}/features_validate.json")


if __name__ == "__main__":
    main()
