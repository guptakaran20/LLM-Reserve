# llm-reserve

Predict a tight 95th-percentile bound on an LLM request's output tokens, so a spend gateway can reserve what the request will likely cost instead of its full `max_tokens`.

Work in progress. See [docs/PRD.md](docs/PRD.md) for the plan and the data check, and [docs/findings.md](docs/findings.md) for results so far.

## Run

Requires Python 3.12.

```bash
make setup     # .venv from the pinned requirements.txt
make data      # download CASTILLO (~600 MB, cached in data/raw) and build data/processed
make explore   # Day-2 plots and stats from the train split -> docs/figures/
make baselines # Day-3 baseline table on validation at 95% coverage -> docs/figures/
make features  # Day-4 feature matrix + MiniLM embeddings (128 and 256 tokens) -> data/features/
make test
```

Data: [CASTILLO](https://huggingface.co/datasets/danfperam/castillo) (CC BY 4.0), Perez-Ramirez, Kostic and Boman, [arXiv 2505.16881](https://arxiv.org/abs/2505.16881).
