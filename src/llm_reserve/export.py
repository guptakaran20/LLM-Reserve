"""Day-11 ONNX export of the frozen winner, and the parity check of the serving path.

The winner (Day 6, scored once on Day 7) is the log-space average of two regressors on the same 74
inputs: the MLP and XGBoost on hand-built features plus the PCA-32 of the MiniLM-128 embedding. Three
graphs are exported to data/models/onnx/ with what they need around them:

- minilm128.onnx: token ids -> BERT -> mean pooling over the mask -> L2 normalization, the same three
  steps as the sentence-transformers model, so the graph returns the Day-4 embedding. tokenizer.json is
  the pinned revision's fast tokenizer, truncating at 128 word pieces.
- mlp.onnx: torch.onnx.export (dynamo) of the frozen QuantileMLP, standardization included.
- xgb_minilm128.onnx: onnxmltools conversion of the booster sliced to its best iteration (Day-5
  inference stops there) and stripped of feature names (the converter only reads f0, f1, ...).
- pca_minilm128_32.npz (copied) and serving.json: column order, model vocabulary, shift, file hashes.

torch and xgboost can't share a process on macOS (findings 20), so this runs as three processes, in order:

    python -m llm_reserve.export torch   # export MiniLM and the MLP; check them vs sentence-transformers and torch
    python -m llm_reserve.export xgb     # convert XGBoost; check ONNX vs xgboost
    python -m llm_reserve.export check   # ONNX Runtime only: write serving.json, run text -> bound end to end

The parity set is 1,000 test prompts drawn with a fixed seed, with every model each was answered by
(12,446 requests). The gate (PRD: within 1e-4 on 1,000 prompts) compares each ONNX stage with its
Python model on identical inputs: tokenizer ids, encoder vs sentence-transformers, each regressor on
the Day-7 design rows and on the serving rows, and `check`'s raw text -> log prediction vs the Python
models fed the same embeddings. Every one must agree within TOL in log1p tokens.

Not gated, reported: the serving path vs the Day-7 saved test predictions. Those used embeddings encoded
in padded batches; served ones are encoded one prompt at a time and differ by ~1e-7, which moves a few
XGBoost rows across a split threshold. xgboost itself does the same when fed per-request PyTorch
embeddings, so it is not an export error (findings 36). `check` also times the single-request path and
asserts that neither torch nor xgboost was imported. Writes docs/figures/onnx_parity.json. Run with
`make onnx` (after `make eval`, whose predictions it compares against).
"""

from __future__ import annotations

import json
import shutil
import sys
import time

import numpy as np
import pandas as pd

from llm_reserve.data import ROOT, load_completions, load_prompts
from llm_reserve.features import EMBED_MODEL, EMBED_REVISION, SERVING_LENGTH, load_features
from llm_reserve.serving import BUNDLE_DIR, SPEC, Predictor, bounds, pca_project, sha256

MODELS_DIR = ROOT / "data" / "models"
FIG_DIR = ROOT / "docs" / "figures"
PRED_DIR = ROOT / "data" / "predictions"
CHECK_DIR = ROOT / "data" / "onnx_parity"  # stage results passed between processes; not part of the bundle
SPLIT = "test"
N_PROMPTS = 1000
SEED = 0
TOL = 1e-4  # PRD: ONNX and Python agree within 1e-4 (log1p tokens) on 1,000 prompts
OPSET = 18  # torch exports
XGB_OPSET = 15  # the highest onnxmltools 1.16 supports
WINNER = "mlp_xgb_avg"
MLP, XGB = "mlp", f"xgb_minilm{SERVING_LENGTH}"
FILES = {"encoder": f"minilm{SERVING_LENGTH}.onnx", "tokenizer": "tokenizer.json", MLP: "mlp.onnx",
         XGB: f"{XGB}.onnx"}
WARMUP = 20
NOISE = 3e-7  # PC perturbation for the sensitivity check: the largest encoder or PCA rounding difference seen


def manifest() -> dict:
    return json.loads((MODELS_DIR / "manifest.json").read_text())


def check_path(stage: str):
    return CHECK_DIR / f"check_{stage}.json"


def parity_rows(n_prompts: int = N_PROMPTS, seed: int = SEED) -> pd.DataFrame:
    """One row per (prompt, model) sample of `n_prompts` random test prompts.

    `row` is the sample's first row in load_features(SPLIT, "original"), so it indexes the Day-7
    predictions; input_size comes from the completions table, which has the same row order.
    """
    _, _, meta = load_features(SPLIT, "original")
    sizes = load_completions(SPLIT, "original").input_size.to_numpy()
    first = meta.drop_duplicates("sample_id").index.to_numpy()
    samples = pd.DataFrame({"row": first, "prompt_row": meta.prompt_row.to_numpy()[first],
                            "model": meta.model.astype(str).to_numpy()[first], "input_size": sizes[first]})
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.sort(samples.prompt_row.unique()), size=n_prompts, replace=False)
    out = samples[samples.prompt_row.isin(chosen)].sort_values(["prompt_row", "model"]).reset_index(drop=True)
    texts = load_prompts().prompt_text.to_numpy()
    return out.assign(text=texts[out.prompt_row.to_numpy()])


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "n": int(d.size)}


def design_rows(rows: pd.DataFrame):
    """The Day-7 design matrix for the parity rows (cached embeddings, frozen PCA), as predict.py builds it."""
    from llm_reserve.models import design
    from llm_reserve.predict import pca_block

    X, _, meta = load_features(SPLIT, "original")
    block = pca_block(manifest()["models"][XGB]["pca_file"])
    idx = rows.row.to_numpy()
    return design(X.iloc[idx].reset_index(drop=True), meta.iloc[idx].reset_index(drop=True), block)


def ort_session(path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


# ---- Inputs shared by the stages -------------------------------------------------------------------------
# A tree splits at float32 thresholds, so an embedding that moves by 1e-7 can send a row down another
# branch. The regressors are therefore checked on two input sets: the Day-7 design rows (Day-4 cached
# embeddings, encoded in padded batches of 64) and the serving rows (the same rows with the PCA block
# recomputed from the ONNX encoder, one unpadded prompt at a time, as `Predictor` does).

def parity_file(stage: str):
    return CHECK_DIR / f"parity_{stage}.npz"


def with_embeddings(A: np.ndarray, rows: pd.DataFrame, prompt_rows: np.ndarray, emb: np.ndarray) -> np.ndarray:
    """A copy of the design rows with the PCA block recomputed from `emb` (one row per entry of prompt_rows)."""
    pca = np.load(MODELS_DIR / manifest()["models"][MLP]["pca_file"])
    pcs = pca_project(emb, pca["mean"], pca["components"])
    pos = pd.Index(prompt_rows).get_indexer(rows.prompt_row.to_numpy())
    out = A.copy()
    out[:, A.shape[1] - pcs.shape[1]:] = pcs[pos]
    return out


# ---- Stage 1: MiniLM and the MLP (torch, no xgboost) ------------------------------------------------------

def mean_pooled_encoder(bert):
    """BERT plus sentence-transformers' Pooling(mean) and Normalize, as one module for export."""
    import torch

    class MeanPooledEncoder(torch.nn.Module):
        def __init__(self, bert):
            super().__init__()
            self.bert = bert

        def forward(self, input_ids, attention_mask):
            hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=torch.zeros_like(input_ids)).last_hidden_state
            w = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * w).sum(1) / w.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    return MeanPooledEncoder(bert).eval()


def export_encoder(encoder, path) -> None:
    import torch

    ids = torch.ones(2, 16, dtype=torch.long)
    dims = {0: "batch", 1: "tokens"}
    # The output is not named "embedding": the dynamo exporter reuses that name inside BERT (duplicate definition).
    torch.onnx.export(encoder, (ids, torch.ones_like(ids)), str(path), dynamo=True, opset_version=OPSET,
                      input_names=["input_ids", "attention_mask"], output_names=["sentence_vector"],
                      dynamic_shapes={"input_ids": dims, "attention_mask": dims}, external_data=False)


def export_mlp_graph(model, n_features: int, path) -> None:
    import torch

    torch.onnx.export(model.eval(), (torch.zeros(2, n_features),), str(path), dynamo=True, opset_version=OPSET,
                      input_names=["x"], output_names=["log_pred"], dynamic_shapes={"x": {0: "batch"}},
                      external_data=False)


def serving_tokenizer(path):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    tok.no_padding()
    tok.enable_truncation(SERVING_LENGTH)
    return tok


def onnx_embeddings(texts: list[str]) -> np.ndarray:
    """The serving path's embeddings: the bundle's tokenizer and encoder, one unpadded prompt per run."""
    tok, enc = serving_tokenizer(BUNDLE_DIR / FILES["tokenizer"]), ort_session(BUNDLE_DIR / FILES["encoder"])
    out = []
    for t in texts:
        ids = np.asarray([tok.encode(t).ids], np.int64)
        out.append(enc.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0])
    return np.stack(out)


def export_torch() -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer

    from llm_reserve.features import load_embeddings
    from llm_reserve.mlp import THREADS, predict_mlp
    from llm_reserve.predict import load_mlp

    torch.set_num_threads(THREADS)
    m = manifest()["models"]
    st = SentenceTransformer(EMBED_MODEL, revision=EMBED_REVISION, device="cpu")
    st.max_seq_length = SERVING_LENGTH
    export_encoder(mean_pooled_encoder(st[0].auto_model.eval()), BUNDLE_DIR / FILES["encoder"])
    shutil.copy(hf_hub_download(EMBED_MODEL, "tokenizer.json", revision=EMBED_REVISION), BUNDLE_DIR / FILES["tokenizer"])
    shutil.copy(MODELS_DIR / m[MLP]["pca_file"], BUNDLE_DIR / m[MLP]["pca_file"])

    model, columns = load_mlp(m[MLP]["file"])
    if columns != m[MLP]["columns"] or columns != m[XGB]["columns"]:
        raise AssertionError("mlp: columns differ from the manifest or from XGBoost's")
    export_mlp_graph(model, len(columns), BUNDLE_DIR / FILES[MLP])

    rows = parity_rows()
    prompts = rows.drop_duplicates("prompt_row")
    prompt_rows, texts = prompts.prompt_row.to_numpy(), prompts.text.tolist()
    # Tokenizer: the standalone fast tokenizer the service uses vs the one sentence-transformers runs.
    ours = [e.ids for e in serving_tokenizer(BUNDLE_DIR / FILES["tokenizer"]).encode_batch(texts)]
    hf_ids = st.tokenizer(texts, truncation=True, max_length=SERVING_LENGTH)["input_ids"]
    # Embeddings: ONNX and sentence-transformers both one prompt at a time, as served; the Day-4 cache was batched.
    emb_onnx = onnx_embeddings(texts)
    emb_torch = st.encode(texts, batch_size=1, normalize_embeddings=True, convert_to_numpy=True)
    emb_cache = load_embeddings(SERVING_LENGTH)[prompt_rows]

    A, cols = design_rows(rows)
    A_serving = with_embeddings(A, rows, prompt_rows, emb_onnx)
    mlp = ort_session(BUNDLE_DIR / FILES[MLP])
    torch_day7, torch_serving = predict_mlp(model, A), predict_mlp(model, A_serving)
    result = {
        "prompts": len(prompts), "token_mismatches": int(sum(a != b for a, b in zip(ours, hf_ids))),
        "truncated_prompts": int(sum(len(i) == SERVING_LENGTH for i in ours)),
        "embedding_onnx_vs_sentence_transformers": diff_stats(emb_onnx, emb_torch),
        "embedding_onnx_vs_day4_cache": diff_stats(emb_onnx, emb_cache),
        "embedding_sentence_transformers_vs_day4_cache": diff_stats(emb_torch, emb_cache),
        "mlp_onnx_vs_torch_day7_inputs": diff_stats(mlp.run(None, {"x": A})[0].reshape(-1), torch_day7),
        "mlp_onnx_vs_torch_serving_inputs": diff_stats(mlp.run(None, {"x": A_serving})[0].reshape(-1), torch_serving),
    }
    np.savez(parity_file("torch"), prompt_rows=prompt_rows, emb_onnx=emb_onnx, emb_torch=emb_torch,
             mlp_python_serving=torch_serving)
    check_path("torch").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


# ---- Stage 2: XGBoost (xgboost, no torch) ----------------------------------------------------------------

def xgb_to_onnx(booster, n_features: int):
    """ONNX model of the booster's first best_iteration + 1 rounds, the ones predict_xgb uses."""
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    served = booster[: booster.best_iteration + 1]  # a copy: the caller's booster keeps its names and rounds
    served.feature_names = None
    served.feature_types = None
    return convert_xgboost(served, initial_types=[("x", FloatTensorType([None, n_features]))],
                           target_opset=XGB_OPSET)


def changed(a: np.ndarray, b: np.ndarray) -> dict:
    """How many predictions move by more than TOL, and in how many prompts' rows."""
    d = np.abs(a - b)
    return {**diff_stats(a, b), "over_tol": int((d > TOL).sum())}


def noise_sensitivity(booster, A: np.ndarray, cols: list[str], scale: float = NOISE, seeds=(0, 1, 2)) -> dict:
    """How many XGBoost predictions move by more than TOL when the PCA block gets float-rounding-sized noise.

    Uniform noise of ±`scale` on every PC (the size of the encoder and BLAS differences above), 3 draws.
    """
    from llm_reserve.models import predict_xgb

    base = predict_xgb(booster, A, cols)
    k = sum(c.startswith("minilm") for c in cols)
    out = []
    for s in seeds:
        noisy = A.copy()
        noisy[:, -k:] += np.random.default_rng(s).uniform(-scale, scale, (len(A), k)).astype(np.float32)
        out.append(changed(predict_xgb(booster, noisy, cols), base))
    return {"scale": scale, "draws": out, "over_tol_mean": float(np.mean([d["over_tol"] for d in out])),
            "max_abs": max(d["max_abs"] for d in out)}


def export_xgb() -> None:
    import xgboost as xgb

    from llm_reserve.models import predict_xgb

    m = manifest()["models"][XGB]
    booster = xgb.Booster()
    booster.load_model(MODELS_DIR / m["file"])
    if booster.best_iteration != m["best_iteration"]:
        raise AssertionError(f"{XGB}: saved best_iteration differs from the manifest")
    (BUNDLE_DIR / FILES[XGB]).write_bytes(xgb_to_onnx(booster, len(m["columns"])).SerializeToString())

    rows = parity_rows()
    A, cols = design_rows(rows)
    if cols != m["columns"]:
        raise AssertionError(f"{XGB}: design columns differ from the frozen model's")
    emb = np.load(parity_file("torch"))  # written by the torch stage
    A_serving = with_embeddings(A, rows, emb["prompt_rows"], emb["emb_onnx"])
    A_torch = with_embeddings(A, rows, emb["prompt_rows"], emb["emb_torch"])
    session = ort_session(BUNDLE_DIR / FILES[XGB])
    py = {k: predict_xgb(booster, X, cols) for k, X in (("day7", A), ("serving", A_serving), ("torch", A_torch))}
    result = {
        "rounds": booster.best_iteration + 1, "rounds_saved": booster.num_boosted_rounds(),
        "onnx_vs_xgboost_day7_inputs": diff_stats(session.run(None, {"x": A})[0].reshape(-1), py["day7"]),
        "onnx_vs_xgboost_serving_inputs": diff_stats(session.run(None, {"x": A_serving})[0].reshape(-1),
                                                     py["serving"]),
        # Not ONNX at all: xgboost itself, on inputs that differ from Day 7's by float rounding only.
        "xgboost_serving_vs_day7_inputs": changed(py["serving"], py["day7"]),
        "xgboost_torch_per_request_vs_day7_inputs": changed(py["torch"], py["day7"]),
        "xgboost_day7_inputs_with_pc_noise": noise_sensitivity(booster, A, cols),
    }
    np.savez(parity_file("xgb"), xgb_python_serving=py["serving"])
    check_path("xgb").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


# ---- Stage 3: ONNX Runtime only -----------------------------------------------------------------------

def write_spec() -> dict:
    """serving.json: everything the Predictor needs besides the graphs, and a hash of each file."""
    m = manifest()
    winner = m["models"][WINNER]
    if m["winner"] != WINNER or winner["members"] != [MLP, XGB]:
        raise AssertionError("the frozen winner is not the MLP + XGBoost average this bundle serves")
    features = json.loads((ROOT / "data" / "features" / "manifest.json").read_text())
    pca_file = m["models"][MLP]["pca_file"]
    spec = {
        "winner": WINNER, "combine": winner["combine"], "shift": winner["shift"],
        "train_quantile": m["train_quantile"], "columns": m["models"][MLP]["columns"], "models": features["models"],
        "encoder": {"file": FILES["encoder"], "tokenizer": FILES["tokenizer"], "max_length": SERVING_LENGTH,
                    "model": EMBED_MODEL, "revision": EMBED_REVISION},
        "pca": {"file": pca_file},
        "members": {MLP: {"file": FILES[MLP]}, XGB: {"file": FILES[XGB]}},
        "sha256": {f: sha256(BUNDLE_DIR / f) for f in (*FILES.values(), pca_file)},
    }
    (BUNDLE_DIR / SPEC).write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def bound_effect(rows: pd.DataFrame, bound_sets: dict[str, np.ndarray]) -> dict:
    """Coverage and reservation ratio of each set of per-request bounds on the parity requests' sanitized completions."""
    from llm_reserve.evaluate import coverage, reservation_ratio

    _, y, meta = load_features(SPLIT, "original")
    sample = meta.sample_id.to_numpy()
    pos = pd.Index(sample[rows.row.to_numpy()]).get_indexer(sample)  # each completion's parity request, or -1
    keep = (pos >= 0) & ~meta.is_degenerate.to_numpy()
    return {"completions": int(keep.sum()),
            **{name: {"coverage": coverage(y[keep], b[pos[keep]]), "ratio": reservation_ratio(y[keep], b[pos[keep]])}
               for name, b in bound_sets.items()}}


def percentiles_ms(seconds: list[float]) -> dict:
    p = np.percentile(np.asarray(seconds) * 1e3, [50, 95, 99])
    return {"p50": float(p[0]), "p95": float(p[1]), "p99": float(p[2]), "n": len(seconds)}


def check() -> None:
    spec = write_spec()
    predictor = Predictor()
    rows = parity_rows()
    idx = rows.row.to_numpy()
    day7 = {name: np.load(PRED_DIR / f"{name}_{SPLIT}.npz")["log_pred"][idx] for name in (MLP, XGB)}
    python = {MLP: np.load(parity_file("torch"))["mlp_python_serving"], XGB: np.load(parity_file("xgb"))["xgb_python_serving"]}
    for ref in (day7, python):
        ref[WINNER] = (ref[MLP] + ref[XGB]) / 2  # holdout.winner_log_pred: float32, mlp first
    got = {k: np.empty(len(rows), np.float64) for k in (MLP, XGB, WINNER, "bound")}
    for _, g in rows.groupby("prompt_row", sort=False):
        out = predictor.predict_rows(g.text.iat[0], g.model.tolist(), g.input_size.tolist())
        for k, key in ((MLP, MLP), (XGB, XGB), (WINNER, "winner"), ("bound", "bound")):
            got[k][g.index.to_numpy()] = out[key]
    vs_python = {k: diff_stats(got[k], python[k]) for k in (MLP, XGB, WINNER)}
    vs_day7 = {k: {**diff_stats(got[k], day7[k]), "over_tol": int((np.abs(got[k] - day7[k]) > TOL).sum()),
                   "prompts_over_tol": int(rows.prompt_row[np.abs(got[k] - day7[k]) > TOL].nunique())}
               for k in (MLP, XGB, WINNER)}
    bound_day7 = bounds(day7[WINNER], spec["shift"])
    bound_python = bounds(python[WINNER], spec["shift"])
    effect = bound_effect(rows, {"served": got["bound"], "day7": bound_day7})

    # Latency of one request (one prompt, one model), single-threaded, after a warm-up.
    rng = np.random.default_rng(SEED)
    picks = rows.groupby("prompt_row").sample(1, random_state=SEED).reset_index(drop=True)
    for r in picks.head(WARMUP).itertuples():
        predictor.predict(r.text, r.model, r.input_size)
    t_embed, t_rest, t_total = [], [], []
    for r in picks.iloc[rng.permutation(len(picks))].itertuples():
        t0 = time.perf_counter()
        emb = predictor.embed(r.text)
        t1 = time.perf_counter()
        predictor.member_logs(predictor.design_rows(r.text, [r.model], [r.input_size], embedding=emb))
        t2 = time.perf_counter()
        predictor.predict(r.text, r.model, r.input_size)
        t3 = time.perf_counter()
        t_embed.append(t1 - t0)
        t_rest.append(t2 - t1)
        t_total.append(t3 - t2)

    imported = sorted(m for m in ("torch", "xgboost", "sentence_transformers", "transformers") if m in sys.modules)
    stages = {s: json.loads(check_path(s).read_text()) for s in ("torch", "xgb")}
    gated = {
        "tokens_identical": stages["torch"]["token_mismatches"] == 0,
        "encoder": stages["torch"]["embedding_onnx_vs_sentence_transformers"]["max_abs"],
        "mlp_day7_inputs": stages["torch"]["mlp_onnx_vs_torch_day7_inputs"]["max_abs"],
        "mlp_serving_inputs": stages["torch"]["mlp_onnx_vs_torch_serving_inputs"]["max_abs"],
        "xgb_day7_inputs": stages["xgb"]["onnx_vs_xgboost_day7_inputs"]["max_abs"],
        "xgb_serving_inputs": stages["xgb"]["onnx_vs_xgboost_serving_inputs"]["max_abs"],
        **{f"end_to_end_{k}": v["max_abs"] for k, v in vs_python.items()},
    }
    result = {
        "split": SPLIT, "prompts": int(rows.prompt_row.nunique()), "requests": len(rows), "seed": SEED, "tol": TOL,
        "gated": gated,
        "passed": bool(gated["tokens_identical"] and all(v <= TOL for k, v in gated.items() if k != "tokens_identical")
                       and not imported),
        "end_to_end_vs_python_same_embedding": vs_python,
        "bound_mismatches_vs_python": int((got["bound"] != bound_python).sum()),
        "end_to_end_vs_day7_predictions": vs_day7,
        "bound_mismatches_vs_day7": int((got["bound"] != bound_day7).sum()),
        "bound_mismatch_max_tokens_vs_day7": float(np.abs(got["bound"] - bound_day7).max()),
        "coverage_and_ratio_on_parity_rows": effect,
        "components": stages,
        "imported_in_serving_process": imported,
        "latency_ms_single_thread": {"embed": percentiles_ms(t_embed), "features_and_heads": percentiles_ms(t_rest),
                                     "predict": percentiles_ms(t_total)},
        "bundle_mb": {f: round((BUNDLE_DIR / f).stat().st_size / 1e6, 2) for f in spec["sha256"]},
    }
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "onnx_parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("gated", "passed", "end_to_end_vs_day7_predictions",
                                              "bound_mismatches_vs_day7", "imported_in_serving_process",
                                              "latency_ms_single_thread")}, indent=2))
    if imported:
        raise AssertionError(f"the serving path imported {imported}")
    if not result["passed"]:
        raise SystemExit(f"parity failed: an ONNX stage differs from its Python model by more than {TOL}")
    print(f"wrote {(FIG_DIR / 'onnx_parity.json').relative_to(ROOT)}: parity holds on {result['prompts']:,} prompts")


STAGES = {"torch": export_torch, "xgb": export_xgb, "check": check}  # in this order


def main(argv: list[str]) -> None:
    if len(argv) != 1 or argv[0] not in STAGES:
        raise SystemExit(f"usage: python -m llm_reserve.export {{{','.join(STAGES)}}}")
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    STAGES[argv[0]]()


if __name__ == "__main__":
    main(sys.argv[1:])
