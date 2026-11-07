"""Day-13 int8 encoder: the PRD's fallback when /predict misses its 15 ms p99, with its own coverage check.

Why (findings 42): on the laptop's macOS, ONNX Runtime runs the fp32 MiniLM on the M5's SME matrix units;
the Linux VM that Docker runs the service in does not expose SME, and the same graph is about 4x slower
there (28 ms for a 128-token prompt on one thread). The VM does expose the int8 dot-product instructions
(asimddp, i8mm), so a dynamically quantized encoder runs at 12 ms on one thread and 7 ms on two.

What: `quantize_dynamic` on the Day-11 encoder, the 36 weight MatMuls only, signed int8 weights scaled
**per output channel**; activations are quantized per call. The token embedding table (a Gather) stays fp32,
so the file is 58.5 MB, not 23. It is added to serving.json (encoder.int8, with its sha256), next to the
fp32 encoder, which stays.

Per channel is the second variant tried (findings 43): the first, per tensor, failed the check below
(validation coverage −0.51 point; embeddings at cosine 0.964 to fp32). Before re-running the check, per
channel was picked over three partial quantizations by embedding cosine to fp32 on 500 validation prompts
(no labels; 0.9986) and by latency in the container (as fast as per tensor; the partial ones were 40–90%
slower). Both variants are checked here, so the rejected one stays reproducible; only per channel is served.

The check, decided before its first run: every validation and test request (sanitized config) is
predicted by the served pipeline with each encoder, and scored with the frozen shift θ on its real completions.
**An int8 encoder is accepted if, on validation, its coverage is at most 0.1 point below fp32's and its
reservation ratio at most 0.5% above.** The shift is not re-tuned. Test is reported for the record, not used to decide.
Writes docs/figures/int8_check.json; run after `export check` (`make onnx` does both).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from llm_reserve.data import ROOT, load_completions, load_prompts
from llm_reserve.export import CHECK_DIR
from llm_reserve.evaluate import coverage, reservation_ratio
from llm_reserve.features import load_features
from llm_reserve.serving import BUNDLE_DIR, SPEC, Predictor, sha256

FIG_DIR = ROOT / "docs" / "figures"
INT8_FILE = "minilm128_int8.onnx"
PER_TENSOR_FILE = CHECK_DIR / "minilm128_int8_per_tensor.onnx"  # the rejected variant; not in the bundle
TEXTS_FILE = CHECK_DIR / "int8_texts.json"  # stage `texts` -> the service image
SERVED_FILE = CHECK_DIR / "int8_served_embeddings.npz"  # stage `embed`, inside the image -> stage `served`
SPLITS = ("validate", "test")
MAX_COVERAGE_DROP = 0.001  # 0.1 point
MAX_RATIO_RISE = 0.005  # 0.5%


def quantize(bundle=BUNDLE_DIR) -> dict:
    """Write the int8 encoder next to the fp32 one and register it in serving.json."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    spec = json.loads((bundle / SPEC).read_text())
    src = bundle / spec["encoder"]["file"]
    if sha256(src) != spec["sha256"][spec["encoder"]["file"]]:
        raise ValueError("the fp32 encoder does not match serving.json: re-run `make onnx`")
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    for path, per_channel in ((bundle / INT8_FILE, True), (PER_TENSOR_FILE, False)):
        quantize_dynamic(src, path, weight_type=QuantType.QInt8, per_channel=per_channel,
                         op_types_to_quantize=["MatMul"])
    import onnxruntime

    spec["encoder"]["int8"] = {"file": INT8_FILE, "from": spec["encoder"]["file"],
                               "method": "onnxruntime.quantization.quantize_dynamic: MatMul weights, signed int8, "
                                         "per output channel; activations per call",
                               "onnxruntime": onnxruntime.__version__}
    spec["sha256"][INT8_FILE] = sha256(bundle / INT8_FILE)
    (bundle / SPEC).write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def split_requests(split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One row per (prompt, model) sample with its text and input size; each completion's sample; y."""
    _, y, meta = load_features(split, "sanitized")
    sizes = load_completions(split, "sanitized").input_size.to_numpy()
    first = meta.drop_duplicates("sample_id").index.to_numpy()
    samples = pd.DataFrame({"sample_id": meta.sample_id.to_numpy()[first], "prompt_row": meta.prompt_row.to_numpy()[first],
                            "model": meta.model.astype(str).to_numpy()[first], "input_size": sizes[first]})
    samples["text"] = load_prompts().prompt_text.to_numpy()[samples.prompt_row.to_numpy()]
    pos = pd.Index(samples.sample_id).get_indexer(meta.sample_id.to_numpy())
    return samples, pos, y


def predict_samples(predictor: Predictor, samples: pd.DataFrame,
                    given: dict[int, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Winner log prediction and bound per sample, plus one embedding per prompt (in group order).

    `given` maps prompt_row to an embedding computed elsewhere (the service image), used instead of the encoder.
    """
    log_pred = np.empty(len(samples), np.float64)
    bound = np.empty(len(samples), np.int64)
    embeddings = []
    for prompt_row, g in samples.groupby("prompt_row", sort=True):
        text = g.text.iat[0]
        emb = predictor.embed(text) if given is None else given[prompt_row]
        embeddings.append(emb)
        out = predictor.predict_rows(text, g.model.tolist(), g.input_size.tolist(), embedding=emb)
        log_pred[g.index.to_numpy()] = out["winner"]
        bound[g.index.to_numpy()] = out["bound"]
    return log_pred, bound, np.stack(embeddings)


def drift(q: dict, ref: dict, emb: np.ndarray, emb_ref: np.ndarray) -> dict:
    """How far an int8 encoder moves the served pipeline from fp32's: coverage, ratio, embeddings, bounds."""
    d_log = np.abs(q["log_pred"] - ref["log_pred"])
    cos = np.sum(emb * emb_ref, axis=1) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(emb_ref, axis=1))
    moved = q["bound"] != ref["bound"]
    rel = q["bound"] / ref["bound"] - 1
    return {"coverage_change_points": 100 * (q["coverage"] - ref["coverage"]),
            "ratio_change_pct": 100 * (q["ratio"] / ref["ratio"] - 1),
            "embedding_cosine": {"min": float(cos.min()), "p01": float(np.percentile(cos, 1)), "mean": float(cos.mean())},
            "log_pred_abs_diff": {"mean": float(d_log.mean()), "p99": float(np.percentile(d_log, 99)),
                                  "max": float(d_log.max())},
            "bounds_moved_share": float(moved.mean()),
            "bound_rel_change": {"p01": float(np.percentile(rel, 1)), "p99": float(np.percentile(rel, 99)),
                                 "min": float(rel.min()), "max": float(rel.max())}}


def accepted(val: dict, variant: str = "int8") -> bool:
    fp32, int8 = val["fp32"], val[variant]
    return bool(int8["coverage"] >= fp32["coverage"] - MAX_COVERAGE_DROP
                and int8["ratio"] <= fp32["ratio"] * (1 + MAX_RATIO_RISE))


def per_tensor_predictor() -> Predictor:
    """The served pipeline with the rejected per-tensor encoder swapped in."""
    import onnxruntime as ort

    p = Predictor(encoder="int8")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 1, 1
    p.encoder = ort.InferenceSession(str(PER_TENSOR_FILE), opts, providers=["CPUExecutionProvider"])
    return p


def prompt_texts() -> None:
    """Every validation and test prompt the check scores, for the service image to embed (stage `texts`)."""
    rows = pd.concat([split_requests(split)[0][["prompt_row", "text"]] for split in SPLITS]).drop_duplicates("prompt_row")
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    TEXTS_FILE.write_text(json.dumps({"prompt_row": rows.prompt_row.astype(int).tolist(), "text": rows.text.tolist()}))
    print(f"wrote {len(rows):,} prompts to {TEXTS_FILE.relative_to(ROOT)}")


def embed_texts(directory: Path) -> None:
    """Inside the service image (stage `embed`): embed the texts with the encoder the service serves."""
    import platform

    import onnxruntime

    encoder, threads = os.environ.get("ENCODER", "int8"), int(os.environ.get("PREDICT_THREADS", 2))
    texts = json.loads((directory / TEXTS_FILE.name).read_text())
    p = Predictor(encoder=encoder, threads=threads)
    t0 = time.perf_counter()
    emb = np.stack([p.embed(t) for t in texts["text"]])
    meta = {"encoder": encoder, "threads": threads, "machine": platform.machine(), "system": platform.system(),
            "onnxruntime": onnxruntime.__version__, "prompts": len(emb), "seconds": time.perf_counter() - t0}
    np.savez(directory / SERVED_FILE.name, prompt_row=np.asarray(texts["prompt_row"]), embedding=emb,
             meta=json.dumps(meta))
    print(f"embedded {len(emb):,} prompts in {meta['seconds']:.0f} s ({meta})")


def served_embeddings() -> tuple[dict[int, np.ndarray], dict]:
    z = np.load(SERVED_FILE)
    return dict(zip(z["prompt_row"].tolist(), z["embedding"])), json.loads(str(z["meta"]))


def check(served: bool = False) -> dict:
    """The coverage check. With `served`, also scores the embeddings the service image computed (stage `served`),
    and the decision is taken on those: they are what /predict uses."""
    predictors = {"fp32": Predictor(encoder="fp32"), "int8": Predictor(encoder="int8"),
                  "int8_per_tensor": per_tensor_predictor()}
    given, meta = served_embeddings() if served else (None, None)
    variants = [*predictors, *(["int8_served"] if served else [])]
    result = {"rule": {"split": "validate", "max_coverage_drop": MAX_COVERAGE_DROP, "max_ratio_rise": MAX_RATIO_RISE,
                       "shift": predictors["fp32"].shift, "shift_retuned": False},
              "variants": {"fp32": "Day-11 encoder, on this machine", "int8": "per-channel int8, on this machine",
                           "int8_per_tensor": "per-tensor int8, on this machine (rejected)",
                           "int8_served": "per-channel int8 embeddings computed inside the service image"},
              "served_embeddings": meta}
    for split in SPLITS:
        samples, pos, y = split_requests(split)
        out, embs, seconds = {}, {}, {}
        for name in variants:
            t0 = time.perf_counter()
            p, emb = (predictors["int8"], given) if name == "int8_served" else (predictors[name], None)
            log_pred, bound, embs[name] = predict_samples(p, samples, emb)
            seconds[name] = time.perf_counter() - t0
            out[name] = {"coverage": coverage(y, bound[pos]), "ratio": reservation_ratio(y, bound[pos]),
                         "log_pred": log_pred, "bound": bound}
        result[split] = {"prompts": int(samples.prompt_row.nunique()), "requests": len(samples), "completions": len(y),
                         "fp32": {"coverage": out["fp32"]["coverage"], "ratio": out["fp32"]["ratio"],
                                  "seconds": seconds["fp32"]}}
        for name in variants[1:]:
            result[split][name] = {"coverage": out[name]["coverage"], "ratio": out[name]["ratio"],
                                   "seconds": seconds[name], **drift(out[name], out["fp32"], embs[name], embs["fp32"])}
        print(split, json.dumps({k: {m: v[m] for m in ("coverage", "ratio", "coverage_change_points", "ratio_change_pct")
                                     if m in v} for k, v in result[split].items() if isinstance(v, dict)}), flush=True)
    result["accepted"] = accepted(result["validate"])
    result["per_tensor_accepted"] = accepted(result["validate"], "int8_per_tensor")
    result["served_accepted"] = accepted(result["validate"], "int8_served") if served else None
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "int8_check.json").write_text(json.dumps(result, indent=2) + "\n")
    word = {True: "accepted", False: "REJECTED", None: "not checked"}
    print(f"int8 per channel here: {word[result['accepted']]}; per tensor: {word[result['per_tensor_accepted']]}; "
          f"in the service image: {word[result['served_accepted']]}. Wrote {(FIG_DIR / 'int8_check.json').relative_to(ROOT)}")
    return result


STAGES = ("quantize", "texts", "embed", "served")


def main(argv: list[str]) -> None:
    """quantize (default): write the int8 encoder and check it here. texts / embed DIR / served: the check on
    the service image's own embeddings (`make int8-served`; embed runs inside the image)."""
    stage = argv[0] if argv else "quantize"
    if stage == "quantize":
        quantize()
        check()
    elif stage == "texts":
        prompt_texts()
    elif stage == "embed":
        embed_texts(Path(argv[1]) if len(argv) > 1 else CHECK_DIR)
    elif stage == "served":
        check(served=True)
    else:
        raise SystemExit(f"stage must be one of {STAGES}")


if __name__ == "__main__":
    main(sys.argv[1:])
