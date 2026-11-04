"""Day-11 serving runtime: request -> 95% output-token bound, on ONNX Runtime alone.

A request is what a gateway sees before it sends a call: the prompt text, the target model and the
prompt length in that model's tokens (`input_size`). The bound comes from the frozen Day-6 winner:

    text ──► hand-built features (features.text_features) ─┐
         └─► tokenizer ─► MiniLM-128 (ONNX) ─► frozen PCA ──┼─► XGBoost (ONNX) ─┐
    model ─► one-hot ; input_size ─► log1p ─────────────────┘   MLP (ONNX) ─────┴─► mean + θ ─► ceil(expm1(·))

Neither torch nor xgboost is imported here (the parity stage of llm_reserve.export asserts it), so
the Day-12 service image needs only onnxruntime, tokenizers, numpy and pandas. The bundle in
data/models/onnx/ is self-contained: the three graphs, tokenizer.json, the PCA and serving.json,
written by `make onnx`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llm_reserve.data import ROOT
from llm_reserve.features import text_features

BUNDLE_DIR = ROOT / "data" / "models" / "onnx"
SPEC = "serving.json"
THREADS = 1  # per session: one request at a time, and room for the rest of the gateway


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pca_project(embeddings: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """The frozen PCA (sklearn's transform formula, as predict.pca_block), accumulated in float64.

    In float32 the result depends on the batch shape: BLAS sums one row and a thousand rows in different
    orders, ~1e-7 apart, enough to move an XGBoost row across a split (findings 36). In float64 the cast
    back to float32 is the same for one request or a batch, so serving and the parity reference agree.
    """
    c = np.asarray(components, np.float64)
    return (np.asarray(embeddings, np.float64) @ c.T - np.asarray(mean, np.float64)[None, :] @ c.T).astype(np.float32)


def bounds(log_pred: np.ndarray, shift: float) -> np.ndarray:
    """Whole-token bounds, at least 1: models.shifted_bounds(log_pred)(shift), without importing xgboost."""
    return np.maximum(np.ceil(np.expm1(np.asarray(log_pred, dtype=np.float64) + shift)), 1.0)


@dataclass(frozen=True)
class Prediction:
    bound: int  # output tokens to reserve
    log_pred: float  # winner's log1p quantile before the shift
    members: dict[str, float]  # each member's log1p quantile


class Predictor:
    """Loads the ONNX bundle once; `predict` is one request, `predict_many` loops it for evaluation."""

    def __init__(self, bundle: Path = BUNDLE_DIR, threads: int = THREADS, verify: bool = True):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.bundle = Path(bundle)
        self.spec = json.loads((self.bundle / SPEC).read_text())
        if verify:
            for name, digest in self.spec["sha256"].items():
                if sha256(self.bundle / name) != digest:
                    raise ValueError(f"{name} does not match serving.json: re-run `make onnx`")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1

        def session(file: str):
            return ort.InferenceSession(str(self.bundle / file), opts, providers=["CPUExecutionProvider"])

        enc = self.spec["encoder"]
        self.tokenizer = Tokenizer.from_file(str(self.bundle / enc["tokenizer"]))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(enc["max_length"])
        self.encoder = session(enc["file"])
        self.heads = {name: session(m["file"]) for name, m in self.spec["members"].items()}
        pca = np.load(self.bundle / self.spec["pca"]["file"])
        self.pca_mean, self.pca_components = pca["mean"], pca["components"]
        self.columns: list[str] = self.spec["columns"]
        self.models: list[str] = self.spec["models"]
        self.shift: float = self.spec["shift"]
        self._model_index = {m: self.columns.index(f"model={m}") for m in self.models}
        self._text_columns = [c for c in self.columns if not c.startswith(("model=", "minilm"))]

    def embed(self, text: str) -> np.ndarray:
        """MiniLM-128 sentence embedding of one prompt (384 floats, L2-normalized), unpadded."""
        ids = np.asarray([self.tokenizer.encode(text).ids], dtype=np.int64)
        return self.encoder.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]

    def design_rows(self, text: str, models: list[str], input_sizes: list[int],
                    embedding: np.ndarray | None = None) -> np.ndarray:
        """The 74-column design matrix, one row per (model, input_size), in the frozen column order."""
        unknown = sorted(set(models) - set(self.models))
        if unknown:
            raise ValueError(f"models the predictor was not trained on: {unknown}")
        feats = text_features(text) | {"log_input_size": 0.0}
        row = np.zeros(len(self.columns), dtype=np.float32)
        for c in self._text_columns:
            row[self.columns.index(c)] = feats[c]
        emb = self.embed(text) if embedding is None else embedding
        pcs = pca_project(emb[None, :], self.pca_mean, self.pca_components)[0]
        row[len(self.columns) - len(pcs):] = pcs
        X = np.repeat(row[None, :], len(models), axis=0)
        X[:, self.columns.index("log_input_size")] = np.log1p(np.asarray(input_sizes, dtype=np.float64))
        for i, m in enumerate(models):
            X[i, self._model_index[m]] = 1.0
        return X

    def member_logs(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {name: s.run(None, {"x": X})[0].reshape(-1) for name, s in self.heads.items()}

    def predict_rows(self, text: str, models: list[str], input_sizes: list[int]) -> dict[str, np.ndarray]:
        """Every member's and the winner's log predictions plus the bounds, for one prompt and many models."""
        members = self.member_logs(self.design_rows(text, models, input_sizes))
        # float32, member order mlp then xgb, as holdout.winner_log_pred computes the Day-7 test predictions
        log_pred = sum(members[n] for n in self.spec["members"]) / np.float32(len(members))
        return {**members, "winner": log_pred, "bound": bounds(log_pred, self.shift)}

    def predict(self, text: str, model: str, input_size: int) -> Prediction:
        out = self.predict_rows(text, [model], [input_size])
        return Prediction(int(out["bound"][0]), float(out["winner"][0]),
                          {n: float(out[n][0]) for n in self.spec["members"]})
