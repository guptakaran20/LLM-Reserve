import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA

from llm_reserve import predict
from llm_reserve.features import FEATURES_DIR, load_features
from llm_reserve.mlp import QuantileMLP, predict_mlp


def test_pca_block_reproduces_sklearn_transform_exactly(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(60, 16)).astype(np.float32)
    pca = PCA(4, random_state=0).fit(emb[:40])
    np.savez(tmp_path / "pca.npz", mean=pca.mean_, components=pca.components_)
    monkeypatch.setattr(predict, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(predict, "load_embeddings", lambda n: emb)
    block = predict.pca_block("pca.npz", 128)
    assert block.columns == ["minilm128_pc0", "minilm128_pc1", "minilm128_pc2", "minilm128_pc3"]
    np.testing.assert_array_equal(block.matrix, pca.transform(emb).astype(np.float32))


def test_load_mlp_rebuilds_the_saved_network(tmp_path, monkeypatch):
    torch.manual_seed(0)
    mean, scale = np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([0.5, 1.0, 2.0], dtype=np.float32)
    model = QuantileMLP(mean, scale, hidden=(8, 4), dropout=0.2, input_dropout=0.2).eval()
    cfg = {"embedding": "pca", "hidden": (8, 4), "dropout": 0.2, "input_dropout": 0.2}
    torch.save({"state_dict": model.state_dict(), "columns": ["a", "b", "c"], "config": cfg}, tmp_path / "mlp.pt")
    monkeypatch.setattr(predict, "MODELS_DIR", tmp_path)
    loaded, columns = predict.load_mlp("mlp.pt")
    x = np.random.default_rng(1).normal(size=(5, 3)).astype(np.float32)
    assert columns == ["a", "b", "c"] and not loaded.training
    np.testing.assert_array_equal(predict_mlp(loaded, x), predict_mlp(model, x))


@pytest.mark.skipif(not (predict.PRED_DIR / "mlp_test.npz").exists()
                    or not (FEATURES_DIR / "features_test.parquet").exists(), reason="run `make eval`")
def test_saved_predictions_are_row_aligned_with_the_features():
    for split in predict.SPLITS:
        n = len(load_features(split, "original")[1])
        for name in (*predict.FAMILIES["xgb"], *predict.FAMILIES["mlp"]):
            pred = predict.load(name, split)
            assert pred.shape == (n,) and np.isfinite(pred).all()
