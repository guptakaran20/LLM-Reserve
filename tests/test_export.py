import json

import numpy as np
import onnxruntime as ort
import pytest
import torch
import xgboost as xgb
from sklearn.decomposition import PCA

from llm_reserve import export, serving
from llm_reserve.features import FEATURES_DIR, load_features
from llm_reserve.mlp import QuantileMLP, predict_mlp
from llm_reserve.models import XGB_PARAMS, predict_xgb, shifted_bounds

FIG_DIR = export.FIG_DIR
HAS_BUNDLE = (serving.BUNDLE_DIR / serving.SPEC).exists() and (FEATURES_DIR / "features_test.parquet").exists()


def session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_pca_project_matches_sklearn_and_ignores_batch_shape():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(300, 48)).astype(np.float32)
    pca = PCA(8, random_state=0).fit(emb[:200])
    mean, comps = pca.mean_.astype(np.float32), pca.components_.astype(np.float32)
    batch = serving.pca_project(emb, mean, comps)
    one_by_one = np.concatenate([serving.pca_project(e[None, :], mean, comps) for e in emb])
    np.testing.assert_array_equal(batch, one_by_one)  # what float32 BLAS does not guarantee (findings 36)
    np.testing.assert_allclose(batch, pca.transform(emb), atol=1e-5)


def test_bounds_equal_the_day5_shifted_bounds():
    log_pred = np.array([-3.0, 0.0, 2.5, 6.9137, 9.2], dtype=np.float32)
    np.testing.assert_array_equal(serving.bounds(log_pred, 0.0335), shifted_bounds(log_pred)(0.0335))
    assert serving.bounds(np.array([-10.0]), 0.0)[0] == 1.0  # never below one token


def test_xgb_export_serves_the_best_iteration_and_leaves_the_booster_alone(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6)).astype(np.float32)
    y = (X[:, 0] * 2 + rng.normal(size=600)).astype(np.float32)
    names = ["log_chars", "a", "b", "c", "minilm128_pc0", "minilm128_pc1"]
    dtr, dva = xgb.DMatrix(X[:400], y[:400], feature_names=names), xgb.DMatrix(X[400:], y[400:], feature_names=names)
    params = {**XGB_PARAMS, "min_child_weight": 5, "nthread": 1, "eta": 0.3}
    booster = xgb.train(params, dtr, 200, evals=[(dva, "va")], early_stopping_rounds=5, verbose_eval=False)
    assert booster.best_iteration + 1 < booster.num_boosted_rounds()  # the slice matters
    (tmp_path / "x.onnx").write_bytes(export.xgb_to_onnx(booster, 6).SerializeToString())
    got = session(tmp_path / "x.onnx").run(None, {"x": X})[0].reshape(-1)
    np.testing.assert_allclose(got, predict_xgb(booster, X, names), atol=1e-5)
    assert booster.feature_names == names and booster.best_iteration + 1 < booster.num_boosted_rounds()


def test_mlp_export_matches_torch_at_any_batch_size(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    mean, scale = rng.normal(size=10).astype(np.float32), rng.uniform(0.5, 2, 10).astype(np.float32)
    model = QuantileMLP(mean, scale, hidden=(16, 8), dropout=0.2, input_dropout=0.2).eval()
    export.export_mlp_graph(model, 10, tmp_path / "mlp.onnx")
    s = session(tmp_path / "mlp.onnx")
    for n in (1, 7):
        x = rng.normal(size=(n, 10)).astype(np.float32)
        np.testing.assert_allclose(s.run(None, {"x": x})[0].reshape(-1), predict_mlp(model, x), atol=1e-6)


def test_encoder_export_mean_pools_over_the_mask_and_normalizes(tmp_path):
    from transformers import BertConfig, BertModel

    torch.manual_seed(0)
    bert = BertModel(BertConfig(vocab_size=50, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                                intermediate_size=64, max_position_embeddings=64)).eval()
    enc = export.mean_pooled_encoder(bert)
    export.export_encoder(enc, tmp_path / "enc.onnx")
    s = session(tmp_path / "enc.onnx")
    ids = np.array([[2, 7, 9, 11, 3], [2, 5, 3, 0, 0]], dtype=np.int64)
    mask = (ids != 0).astype(np.int64)
    out = s.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    with torch.no_grad():
        ref = enc(torch.as_tensor(ids), torch.as_tensor(mask)).numpy()
    np.testing.assert_allclose(out, ref, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    alone = s.run(None, {"input_ids": ids[1:, :3], "attention_mask": mask[1:, :3]})[0]
    np.testing.assert_allclose(alone[0], out[1], atol=1e-6)  # padding does not change the embedding


@pytest.mark.skipif(not HAS_BUNDLE, reason="run `make onnx`")
def test_parity_rows_are_1000_test_prompts_with_every_model_once():
    rows = export.parity_rows()
    assert rows.prompt_row.nunique() == export.N_PROMPTS
    assert not rows.duplicated(["prompt_row", "model"]).any()
    _, _, meta = load_features("test", "original")
    assert (meta.prompt_row.to_numpy()[rows.row] == rows.prompt_row).all()
    assert (meta.model.astype(str).to_numpy()[rows.row] == rows.model).all()
    assert rows.equals(export.parity_rows())  # fixed seed


@pytest.mark.skipif(not HAS_BUNDLE, reason="run `make onnx`")
def test_predictor_builds_the_day7_design_row_from_raw_text():
    p = serving.Predictor()
    rows = export.parity_rows().head(40)
    A, cols = export.design_rows(rows)
    assert cols == p.columns
    for _, g in rows.groupby("prompt_row"):
        X = p.design_rows(g.text.iat[0], g.model.tolist(), g.input_size.tolist())
        np.testing.assert_array_equal(X[:, :42], A[g.index, :42])  # hand-built, input size, one-hot: bit-identical
        np.testing.assert_allclose(X[:, 42:], A[g.index, 42:], atol=1e-5)  # PCs: per-request encoding
    with pytest.raises(ValueError, match="not trained on"):
        p.predict("hello", "openai--gpt-5", 10)
    pred = p.predict(rows.text.iat[0], rows.model.iat[0], int(rows.input_size.iat[0]))
    assert pred.bound == int(serving.bounds(np.array([pred.log_pred]), p.shift)[0]) >= 1


@pytest.mark.skipif(not HAS_BUNDLE, reason="run `make onnx`")
def test_predictor_refuses_a_bundle_whose_files_changed(tmp_path):
    import shutil

    shutil.copytree(serving.BUNDLE_DIR, tmp_path / "b")
    spec = json.loads((tmp_path / "b" / serving.SPEC).read_text())
    spec["sha256"]["mlp.onnx"] = "0" * 64
    (tmp_path / "b" / serving.SPEC).write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="mlp.onnx"):
        serving.Predictor(tmp_path / "b")


@pytest.mark.skipif(not (FIG_DIR / "onnx_parity.json").exists(), reason="run `make onnx`")
def test_day11_gate_onnx_and_python_agree_within_1e4_on_1000_prompts():
    """The Day-11 gate: every ONNX stage vs its Python model on identical inputs, and text -> bound end to end."""
    r = json.loads((FIG_DIR / "onnx_parity.json").read_text())
    assert r["passed"] and r["prompts"] == 1000 and r["split"] == "test" and r["tol"] == 1e-4
    assert r["gated"]["tokens_identical"]
    assert all(v <= 1e-4 for k, v in r["gated"].items() if k != "tokens_identical")
    assert r["imported_in_serving_process"] == []  # the service needs neither torch nor xgboost
    # Reported, not gated: drift from the Day-7 predictions stays a rounding matter for the bounds.
    assert r["bound_mismatch_max_tokens_vs_day7"] <= 1
    eff = r["coverage_and_ratio_on_parity_rows"]
    assert abs(eff["served"]["coverage"] - eff["day7"]["coverage"]) <= 1e-3
    assert abs(eff["served"]["ratio"] / eff["day7"]["ratio"] - 1) <= 1e-3
