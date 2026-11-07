# The demo page: run it, and host it for free

`app/streamlit_app.py` is a Streamlit page over the real model: type a prompt and see the bound, the reservation each
policy would hold, a budget demo, and the simulator results. It runs the Day-11 ONNX bundle (int8 encoder) **in the
page's own process**, so it needs no Redis, no dataset and no GPU. Measured: 0.1 s to load, about 3 ms a prediction, about
300 MB of memory for the model.

## Run it locally

```bash
make app        # builds .venv-app from app/requirements.txt (streamlit, onnxruntime, tokenizers, numpy, pandas), then
                # runs `streamlit run app/streamlit_app.py` on http://localhost:8501
make app-test   # the demo's tests, including the page itself headless, in that same small venv
```

`.venv-app` is deliberately not the project's `.venv`: it holds only what a host installs, so a missing dependency shows up
here and not after deploying. The page's logic is in `src/llm_reserve/demo.py`; the main test suite covers it too
(`make test`), and skips only the page test, which needs streamlit.

`app/` is self-contained and checked in: `app/bundle/` (63 MB: the int8 encoder, the two regressors, tokenizer, PCA,
`serving.json` with sha256 hashes, which the predictor verifies at load), and `app/data/` (the per-model bar, the simulator
grid, the headline numbers). After retraining or re-running the simulator, `make demo-assets` rebuilds it from
`data/models/onnx/` and `docs/figures/`.

## Free hosting

### Streamlit Community Cloud (recommended)

Free, and made for this: it builds from a GitHub repository and rebuilds on every push.

1. Push the branch with `app/` to GitHub (the 63 MB `app/bundle/minilm128_int8.onnx` is under GitHub's 100 MB file limit,
   though above its 50 MB warning; Git LFS files are not fetched by the host, so do not move it to LFS).
2. At [share.streamlit.io](https://share.streamlit.io): sign in with GitHub, **Create app**, pick the repository and branch,
   and set the main file to `app/streamlit_app.py`. In *Advanced settings* choose Python 3.12.
3. Deploy. The first build installs `app/requirements.txt` (about a minute).

Why it works without changes:

- **Dependencies:** Community Cloud looks for a dependency file in the entrypoint's directory first and only then at the
  repository root, so `app/requirements.txt` wins over the root `requirements.txt` (which pins torch and xgboost and would
  not fit). Keep only one dependency file in `app/`.
- **Paths:** the host starts the app from the repository root; the page finds `app/bundle` and `src/` from its own file's
  location, and adds `src/` to `sys.path` itself, so there is no `pip install -e .`.
- **Config:** only a `.streamlit/config.toml` at the repository root is read, which is where this one is.
- **Memory:** about 300 MB for the model plus Streamlit itself; the documented free limits (checked November 2026, and
  subject to change without notice) are roughly 690 MB to 2.7 GB and up to 2 cores.

Free apps go to sleep after a period without visitors and wake on the next visit (a few seconds, plus model load).
Whether a private repository can be deployed on the free tier is documented as yes, but check the current terms before
relying on it; a public repository is the safe path, and the project is meant to be public anyway.

### Hugging Face Spaces (fallback)

A Space with the Streamlit SDK also runs this app: push the repository (or just `app/` and `src/`) to it, with `app/streamlit_app.py` as the app file. I would not rely on it: forum reports from mid-2026 say the free
CPU tier's quota changed for new accounts, so check Hugging Face's current pricing page first. Large files go through its
Git LFS, which, unlike Community Cloud, it does support.

### What is not hosted

The hosted page is a demonstration of the model and the simulator results. The real gateway component, the FastAPI service
with its Redis Lua scripts, needs a Redis and a long-running container, which no free web host gives for free; it runs with
`make serve`, and the numbers for it are in the README and findings 42–48.

## Checked before this was written

- A fresh virtualenv with only `app/requirements.txt` (no torch, xgboost, scikit-learn or Redis) runs the page and passes
  `tests/test_demo.py`, including the page rendered headless and every control.
- The trimmed bundle predicts exactly what the full int8 bundle does (same bounds for every model), and a bundle with a
  wrong hash is refused.
- The simulator tab reproduces the README's 1.476× (the learned bound with the 1.5× cap, 8 requests/s) from the same JSON.
- Not checked: an actual deployment, which needs your GitHub and Streamlit accounts.
