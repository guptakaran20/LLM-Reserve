# PRD: llm-reserve, learned output-token reservation for LLM budgets

28 Oct 2026 · Karan Gupta · **Revision 2** (after the day-0 data check, same day)

> **What changed in rev 2.** Before any code, I loaded every CASTILLO split and checked the PRD's assumptions against the data. Four PRD assumptions were wrong or untested. Changes are tagged **(rev 2)** throughout. The evidence is in [Data check](#data-check-28-oct-2026), and the full list is in the [changelog](#changelog).

## Overview and problem

llm-reserve predicts a tight upper bound on how many output tokens an LLM request will use. A spend gateway can then reserve what the request will likely cost instead of its worst case, and admit more requests under the same budget.

**The problem.** Spend-limit gateways (including XCEED's AI Studio) reserve each request's worst-case cost before sending it: input tokens plus the full `max_tokens` of output. After the response, they settle to the real usage. Real output lengths are heavy-tailed, so most of each reservation is never used. Under a tight budget window, parallel requests get rejected while the budget is not actually spent.

**The idea.** Replace "assume `max_tokens`" with a learned 95th-percentile prediction of output length for each request. Prove it on real data with a coverage vs. savings trade-off, then show the system-level effect in a budget simulator.

**Version 1 in 14 days:**

- Quantile models trained on the public CASTILLO dataset (13 open-source LLMs, 10 completions per prompt).
- An evaluation that compares methods fairly, at the same coverage, **with an oracle ceiling so the result is read against what is achievable (rev 2)**.
- A discrete-event simulator of a budget gateway under concurrent load.
- A FastAPI service with ONNX inference and atomic Redis reserve-and-settle.

## Goals, non-goals and success metrics

Version 1 succeeds if a trained model reserves at least 20% fewer tokens than a fixed per-model 95th percentile at the same 95% coverage, and the simulator turns that into more admitted requests without real overshoot.

### Goals

1. Beat the strongest simple baseline (a fixed percentile per model) at matched coverage.
2. Show the system-level gain: more requests admitted under the same budget.
3. Ship a reproducible repo and a working service someone else can run.
4. Show real ML skill: models you train yourself, evaluated honestly.

### Non-goals for version 1

- Data from closed API models like Claude or Gemini (planned for January).
- Fine-tuning any LLM.
- Streaming top-up reservations, multi-tenant auth, or a web UI.

### Success metrics

Targets to test, not promises. Report the real numbers either way.

| Metric | Target | How it's measured |
|---|---|---|
| Test coverage | 95% ± 1% | Share of test completions with actual length ≤ predicted bound; quantile tuned on validation |
| Reserved tokens vs. fixed per-model 95th percentile | At least 20% fewer | Sum of predicted bounds ÷ sum of baseline bounds, test set, same coverage |
| **Share of the oracle gap closed (rev 2)** | **Report it (20% fewer ≈ 50% of the gap)** | (baseline ratio − model ratio) ÷ (baseline ratio − oracle ratio). On validation the baseline is 2.25 and the oracle is about 1.31 |
| Reserved tokens vs. `max_tokens` = 4,096 | Report the ratio | Same ratio. **The 4,096 policy covers 99.96% of completions, not 100% (rev 2)** |
| Requests admitted in the simulator | At least 1.5× the `max_tokens` policy | Same budget, arrival rate and trace |
| **Requests admitted vs. fixed per-model p95 policy (rev 2)** | **Report it: this is the honest comparison** | Same budget, arrival rate and trace. Beating `max_tokens` by 1.5× is nearly guaranteed, because its reservations are 4× larger than the fixed p95's |
| Budget overshoot | At most 1% of the window budget | Largest overshoot across simulator runs |
| Predictor latency | p99 at most 15 ms on CPU | Benchmark of `/predict`, embedding included. **MiniLM input is capped at 128 tokens: at the default 256, the embedding alone measured 13.4 ms p99 (rev 2)** |
| Reproducibility | One command per stage | `make data`, `make train`, `make eval`, `make sim`, `docker compose up` |

## Users and use cases

The users are developers who put hard spending limits in front of LLM calls and hit false rejections because worst-case reservations are too large.

| User | Situation | What llm-reserve gives them |
|---|---|---|
| App developer with per-user credits (like AI Studio) | Each user has a small credit balance; one call reserves a large worst case, so users get blocked before their credits run out | Reservations close to real cost, so users can make more calls with the same credits |
| Team running a shared LLM gateway | Hourly or per-minute budget windows; parallel agents get 429s while the budget isn't actually spent | More concurrent requests admitted under the same cap, with bounded overshoot |
| Agent framework author | Agents fire many calls at once; reserving `max_tokens` for each serializes them | A drop-in predictor plus reserve-and-settle middleware |

**Core use case:** a request arrives, the predictor returns a 95th-percentile output-length bound in under 15 ms, the gateway reserves input tokens plus `min(bound, max_tokens)` in Redis, forwards the request, then settles to actual usage.

## Requirements

Eight functional requirements cover everything from data to docs. Each has an acceptance check you can run, so "done" is never a judgement call.

### Functional

| ID | Requirement | Accepted when |
|---|---|---|
| FR1 | Data pipeline: load CASTILLO, one row per completion, keep the official splits **but move the 17 prompts that appear in more than one split into a single split (rev 2)**, flag degenerate samples, cache to parquet | Row counts per split are logged; prompt overlap between all split pairs is 0, **asserted in a test (rev 2)** |
| FR2 | Features from the prompt only: length, lines, code present, cue words, length hints, target model, MiniLM embedding computed once per unique prompt. **Not `temp`, `top_p`, `top_k`: they are fixed per model, so the model one-hot already carries them (rev 2)** | Unit tests pass; an assert proves `longest_response`, `shortest_response`, `dataset` and the other "never" columns are not features |
| FR3 | Baselines: `max_tokens`, fixed per-model 95th percentile, per-model × prompt-length-bucket 95th percentile. **Plus two reference rows that are not deployable: the oracle (each prompt's own completions) and per-model × source dataset (rev 2)** | Baseline table on validation |
| FR4 | Models: linear quantile regression, XGBoost (`reg:quantileerror`, log target), PyTorch MLP with pinball loss; quantile tuned on validation to 95% coverage | Each model has a validation row in the results table |
| FR5 | Evaluation: coverage, reservation ratio, **oracle gap closed (rev 2)**, coverage vs. reservation curve (quantiles 0.80–0.99), per-model breakdown, the 100 worst under-predictions grouped by cause | Test set used once; plots saved; error categories counted |
| FR6 | Simulator: discrete-event loop, pluggable policies, token budget per window, random arrivals, settle on completion, two misprediction strategies (cap at bound, overdraft buffer). **Actual lengths come from the `original` config (degenerate outputs included), clipped at the request's `max_tokens` (rev 2)** | Policy comparison table: admitted, wrongful rejections, overshoot, truncation rate |
| FR7 | Service: FastAPI `POST /predict`, `/reserve`, `/settle`; ONNX Runtime inference; atomic Redis Lua reserve; docker-compose with API and Redis | An end-to-end curl flow works from a fresh clone |
| FR8 | Docs: README with problem, method, results, plots and run steps; short blog post | A stranger can reproduce the headline numbers |

### Non-functional

- **Latency:** `/predict` p99 at most 15 ms on CPU including the embedding; `/reserve` p99 at most 5 ms.
- **Correctness:** 1,000 parallel reserves never push spend past the cap.
- **Reproducibility:** fixed random seeds, pinned `requirements.txt`, **Python 3.12 pinned locally, in CI and in the Dockerfile (rev 2; the laptop's system Python is 3.15, which ML wheels may not support yet)**, one make target per stage.
- **Quality:** pytest for features, metrics, **split leakage (rev 2)** and the Lua script; GitHub Actions runs the tests on every push.

## Architecture

Training runs once offline. At request time the gateway gets a predicted bound, reserves it atomically in Redis, forwards the call and settles to actual usage.

1. The client sends a prompt and its `max_tokens`.
2. The gateway asks the predictor for a 95th-percentile output bound.
3. Redis runs one Lua script: reserve input tokens plus `min(bound, max_tokens)` if it fits the window; otherwise the client gets a 429.
4. The gateway forwards the request to the LLM provider.
5. When the response returns, Redis settles the reservation to the reported token usage.

## Data check (28 Oct 2026)

**(rev 2)** All numbers come from the Hugging Face parquet export of `danfperam/castillo` (sanitized and original, all three splits) and the CASTILLO paper ([arXiv 2505.16881](https://arxiv.org/abs/2505.16881)).

### What the data actually is

| Fact | Value | Consequence |
|---|---|---|
| Split names | `train`, `validate`, `test` (and a `full` split that is all three combined) | Load `validate`, not `validation`; never load `full` |
| Rows (sanitized) | train 97,825 · validate 27,927 · test 14,120 | One row per (prompt, model) |
| Completions (sanitized) | train 976,769 · validate 278,821 · test 141,004 | About 1.4M exploded rows: fits in memory as float32 |
| Unique prompts | train 7,999 · validate 2,278 · test 1,136 (11,395 total) | **The effective sample size is ~8k prompts, not ~1M rows.** The MLP needs regularization; embeddings are cheap (11.4k × 384) |
| Split method (paper) | 70/20/10 per source dataset, stratified for Dolly and Apps | Validation is twice the size of test |
| Generation cap (paper) | **15,000 output tokens**, input capped at 2,500 | `max_tokens` = 4,096 is a gateway default, not the data's cap |
| Decoding (paper and data) | Each model's default `temp`, `top_p`, `top_k`, fixed per model (gemma-3-4b has two configs) | Not useful as features: the model one-hot already carries them |
| Degenerate rule (paper) | Any completion hit 15,000, or std > 2 × mean with max ≥ 8,500 | `sanitized` drops degenerate *completions*, not rows: 954 train rows have fewer than 10 |
| `category` | 82% of prompts are `undefined` (only Dolly is labeled) | Group error analysis by `dataset` instead |
| Models with fewer prompts | gemma-3-12b (6,424 train rows) and Llama-3.2-1B (6,284) vs. ~7,738 for the rest | Note it in the per-model breakdown |

### Leakage: the day-1 check fails

| Split pair | Prompts in both |
|---|---|
| train / validate | 12 |
| train / test | 4 |
| validate / test | 3 |

That is **17 distinct prompts**: the pair counts add to 19 because one prompt is in all three splits. One prompt has two gemma-3-4b samples in train. They are not duplicates: they come from two different decoding configs (two `sample_id`s), so both are kept, and `sample_id` is the row key. `prompt_id` and `prompt_text` map one-to-one, and the overlap is the same after lower-casing and normalizing whitespace.

**Fix:** keep the official splits, and move each shared prompt to its earliest split in the order train → validate → test (removing its rows from the later ones). A test asserts zero overlap. This drops 220 rows from validate and test and keeps test strictly unseen.

### Output length: sanitized vs. original

| Config / split | p50 | p95 | p99 | > 4,096 | > 8,192 | At the 15,000 cap |
|---|---|---|---|---|---|---|
| sanitized / validate | 381 | 977 | 1,557 | 0.040% | 0.008% | 0 |
| original / validate | 382 | 988 | 1,624 | 0.201% | 0.169% | 444 completions |
| sanitized / test | 372 | 949 | 1,586 | 0.069% | 0.024% | 0 |
| original / test | 372 | 958 | 1,644 | 0.215% | 0.170% | 205 completions |

The two configs barely differ at p95. They differ in the far tail, which is where gateway overshoot comes from. So coverage is scored on both, as planned, and **the simulator draws actual lengths from `original`**.

Per-model p95 ranges from 757 (Qwen2.5-32B) to 1,792 (gemma-3-12b). Within one prompt, the 10 completions have a median coefficient of variation of 0.15, and the longest is 1.94× the median at p90 and 5.6× at p99. That spread is noise that no prompt-only predictor can remove.

### Is the 20% target reachable? Validation ceiling

Each method's quantile (or scale) is tuned on validation to exactly 95% coverage, sanitized config, 278,821 completions, mean length 421.

| Method | Deployable | Coverage | Reservation ratio | vs. per-model p95 |
|---|---|---|---|---|
| `max_tokens` = 4,096 | yes | 99.96% | 9.74 | 333% more |
| Global p95 (one number for all) | yes | 95.00% | 2.32 | 3.3% more |
| **Fixed per-model p95 (the bar)** | yes | 95.00% | **2.25** | — |
| Per-model × input-length quintile | yes | 95.00% | 2.14 | 4.6% fewer |
| Per-model × source dataset | no (dataset unknown at serving time) | 95.00% | 2.00 | 11.1% fewer |
| Oracle: own mean × 1.37 | no | 95.00% | 1.37 | 39.0% fewer |
| Oracle: own mean + 1.69 × own std | no | 95.00% | 1.34 | 40.6% fewer |
| Oracle: 1.11 × own mean + own std | no | 95.00% | **1.31** | **41.6% fewer** |

**Reading it.** The oracle knows each prompt's true length distribution, and is still about 1.31× actual usage because of the within-prompt spread. So the most any prompt-only predictor can save over the fixed per-model p95 is about **40%**. The 20% target means closing about half that gap.

That is ambitious but plausible. Coarse signals are weak: input length alone gets 4.6%, and even knowing the source dataset only gets 11%. **The model has to learn what each prompt asks for**, which is what the embedding and the length-hint features are for. If the model lands between 11% and 20%, that is still a real result, and the gap-closed metric makes it readable.

These oracle numbers are in-sample (they use validation completions to bound the same completions), so the true ceiling is slightly lower.

### Embedding latency

MiniLM (`all-MiniLM-L6-v2`) truncates input at 256 word pieces by default. 30.7% of prompts have more than 256 input tokens and 15.4% have more than 512. Apps prompts have a median of 512, so code prompts get truncated most. Length is still a hand-built feature, so the model doesn't lose it.

This is a single-prompt `encode()` benchmark on 400 test prompts: PyTorch eager, CPU, Apple M5 laptop. It measures the embedding only, with no HTTP or regressor.

| MiniLM max input | Threads | p50 | p95 | p99 |
|---|---|---|---|---|
| 128 tokens | 1 | 5.6 ms | 6.9 ms | 7.9 ms |
| 256 tokens (default) | 1 | 5.6 ms | 12.3 ms | **13.4 ms** |
| 128 tokens | 4 | 5.2 ms | 6.1 ms | 7.3 ms |
| 256 tokens (default) | 4 | 5.2 ms | 9.8 ms | 10.8 ms |

**Decision:** with the default 256-token input, the embedding alone uses 11–13 ms of the 15 ms p99 budget. That is on a fast laptop, before FastAPI, the regressor and Docker overhead, so CI or a Linux container will be slower. **Default to 128 tokens** (p99 about 8 ms). On Day 4/5, also build 256-token embeddings and compare them on validation, so the accuracy cost of truncation is measured, not assumed. The 15 ms target stays; ONNX on Day 11 may buy more headroom, but don't count on it.

## Data and evaluation method

Train on CASTILLO's sanitized config, score each completion per prompt separately, and compare every method at the same 95% coverage on a test set used once.

**Dataset.** CASTILLO records 10 completions for each prompt and model pair, across 13 open-source LLMs and 7 instruction datasets, under CC BY 4.0. It has three configs: `sanitized` (degenerate completions removed, the default), `original` (everything) and `degenerate` (only the broken ones). Each has `train`, `validate` and `test` splits.

```python
from datasets import load_dataset
ds = load_dataset("danfperam/castillo", name="sanitized")
# splits: "train", "validate", "test"  (not "validation"; ignore "full")
```

| Field | How it is used |
|---|---|
| `prompt_text` | Hand-built features and the MiniLM embedding |
| `input_size` | Feature (prompt length in tokens) |
| `model` | Feature, one-hot |
| `temp`, `top_p`, `top_k` | **Not features (rev 2):** fixed per model in this data. Revisit in v2, when real requests set them |
| `output_sizes` | The label: exploded into one row per completion (up to 10; fewer in sanitized when completions were degenerate) |
| `prompt_id` | Group key for splits and leakage checks |
| `dataset` | Error analysis only, never a feature |
| `category` | Error analysis only, never a feature; mostly `undefined`, so of little use |
| `longest_response`, `shortest_response`, `output_mean`, `output_std`, `output_percentiles`, `gen_time`, `sample_id` | Never features: they contain the answer (`sample_id` hashes generation parameters) |

### Leakage rules

1. **(rev 2)** The day-1 check has already found 17 prompts shared across splits. FR1 moves each into one split; a test asserts zero overlap between all split pairs.
2. A unit test fails the build if any "never" column reaches the feature matrix.
3. The test split is scored once, after every choice is frozen on validation.
4. **(rev 2)** Anything fitted from data, including length-bucket edges, scalers and per-model percentiles, is fitted on train only.

### Metrics

Here y is the actual output length and ŷ the predicted bound.

```
coverage          = (1/N) · Σ 1[ y_i ≤ ŷ_i ]
reservation ratio = Σ ŷ_i / Σ y_i
oracle gap closed = (ratio_baseline − ratio_model) / (ratio_baseline − ratio_oracle)      (rev 2)
```

**Matched-coverage protocol.** For every method, tune its quantile (or scale factor) on validation until coverage hits 95%, freeze it, then score test. A lower reservation ratio at the same coverage is the win. Also sweep quantiles 0.80 to 0.99 to draw the coverage vs. reservation curve.

**Oracle (rev 2).** For each (prompt, model), the bound is k × own mean + own std, with k tuned to 95% coverage on the same split. Report it on validation and test as the ceiling. It is not a competitor.

### Breakdowns and checks

- Results per model, per source dataset and per prompt-length bucket.
- Test on both sanitized and original. They agree at p95; the difference is in the > 4,096 tail.
- Error analysis: the 100 worst under-predictions by (y − ŷ) ÷ ŷ, grouped by cause and counted.
- Stretch: leave one source dataset out of training and test on it, to check generalization to new prompt styles.

### `max_tokens` handling (rev 2)

CASTILLO was generated with a 15,000-token cap, so some outputs exceed 4,096 (0.04% sanitized, 0.2% original on validation).

- **Offline evaluation** scores raw lengths. The `max_tokens` = 4,096 row reports its real coverage (99.96%), not an assumed 100%.
- **The simulator and service** treat each request as carrying `max_tokens` = 4,096. The provider stops there, so actual cost is min(y, 4,096) and every policy reserves at most `max_tokens`. The simulator reports how many requests were clipped.

## 14-day build plan

Week 1 builds and freezes the model; week 2 turns it into a simulated, served and shipped system. The plan assumes about 5 focused hours a day, with the daily DSA routine kept separate.

Each gate is a hard checkpoint: don't start the next phase until its "Done when" is true.

### Day 1: Set up and load the data

- [x] Create the llm-reserve repo: folder layout, **Python 3.12 venv (rev 2)**, pinned `requirements.txt`, Makefile, pytest, and a GitHub Actions workflow that runs the tests.
- [x] Load sanitized and original (the `degenerate` config is optional); print the schema and row counts per split.
- [x] **(rev 2)** Apply the overlap fix from the data check (17 shared prompts) and add the zero-overlap test.
- [x] Explode `output_sizes` to one row per completion; save parquet per split behind `make data`.
- [x] ~~Check whether `temp`, `top_p` and `top_k` vary.~~ Done in the data check: fixed per model.

**Done when:** `make data` rebuilds the parquet files from scratch, logs row counts, and the overlap test passes.

**Done, 30 Oct 2026.** From a fresh venv and an empty `data/`, `make data` takes 3 minutes (download) plus 2 seconds (processing), and 10 of 10 tests pass. Two implementation notes:

- **`sanitized` is stored as a view of `original`.** For every sample, the sanitized completion lengths are a sub-multiset of the original ones; 3 samples are fully degenerate and appear only in `original`. So each split is one table, `completions_{split}.parquet`, with an `is_degenerate` flag, and `load_completions(split, "sanitized")` filters on it. The pipeline asserts that the counts match the sanitized config exactly.
- **The data is pinned:** the Hugging Face parquet export at revision `92db54f9`. Only safe columns are read; asking `read_raw` for a label-derived column raises an error.

| Split | Prompts | Samples | Completions (original) | Completions (sanitized) | Moved out by the overlap fix |
|---|---|---|---|---|---|
| train | 7,999 | 97,827 | 978,270 | 976,769 | — |
| validate | 2,266 | 27,780 | 277,800 | 277,351 | 12 prompts, 147 samples |
| test | 1,130 | 14,048 | 140,480 | 140,276 | 6 prompts, 73 samples |

### Day 2: Explore the data

- [ ] Histogram of output length per model, log x-axis.
- [ ] Scatter of output length against `input_size`.
- [ ] Within-prompt spread: the distribution of std ÷ mean across each prompt's 10 completions.
- [ ] Share of degenerate samples per model (original vs. sanitized).
- [ ] Write 5 observations in `docs/findings.md` (start from the data check above).

**Done when:** 4 plots are saved and 5 observations are written.

### Day 3: Metrics and baselines

- [ ] `evaluate.py`: coverage, reservation ratio, **oracle gap closed (rev 2)**, matched-coverage tuning on validation, and the 0.80–0.99 curve sweep.
- [ ] Unit tests for the metrics using small hand-computed examples.
- [ ] Baselines: `max_tokens` = 4,096, fixed per-model 95th percentile, per-model × prompt-length-bucket 95th percentile (**quintile edges fitted on train, rev 2**).
- [ ] **(rev 2)** Reference rows: the oracle and per-model × source dataset. They should reproduce the data-check table (2.25 / 2.14 / 2.00 / about 1.31).

**Done when:** the baseline table on validation at 95% coverage exists. This is the bar every model must beat. **Gate: baselines locked.**

### Day 4: Features

- [ ] Hand-built features from `prompt_text`: characters, lines, code fences, question marks, cue words (explain, list, write, summarize), length hints ("in 100 words", "one sentence", "list 10").
- [ ] MiniLM embeddings for unique prompts only (about 11.4k), cached to `.npy`, joined back by `prompt_id`. **Use a 128-token input (from the latency check), also build a 256-token version to compare on validation, and use the same length in training and serving (rev 2).**
- [ ] Leakage unit test on the feature matrix.
- [ ] One hour on quantile regression and pinball loss.

**Done when:** `make features` builds the matrix for all splits and the leakage test passes.

### Day 5: Linear and XGBoost models

- [ ] Linear quantile regression on the hand-built features (sklearn `QuantileRegressor`, on a sample if it is slow).
- [ ] XGBoost with `reg:quantileerror` on a `log1p` target, early stopping on validation.
- [ ] Tune each model's quantile on validation to 95% coverage and compare reservation ratios with the baselines.
- [ ] Feature importance plot.

**Done when:** both models have validation rows and you know whether XGBoost beats the fixed 95th percentile.

### Day 6: Neural network

- [ ] PyTorch MLP: embedding plus hand-built features, two hidden layers (256 and 64 units, ReLU), one output (log length). **Add dropout and weight decay: train has only ~8k unique prompts (rev 2).**
- [ ] Write the training loop yourself with pinball loss and Adam; early stopping on validation.
- [ ] Tune the quantile and add the validation row.

**Done when:** the MLP has a validation row and you have picked the winning model.

### Day 7: Test once, then error analysis

- [ ] Freeze every choice; score test on both sanitized and original.
- [ ] Coverage vs. reservation curve for all methods, **with the oracle as a reference line (rev 2)**, plus the per-model breakdown.
- [ ] Group the 100 worst under-predictions by cause and count them.
- [ ] Buffer: finish anything left over from days 1–6.

**Done when:** the results table, curve and error categories are in `docs/findings.md`. **Gate: model frozen.**

### Day 8: Simulator core

- [ ] Event loop with `heapq` for arrival and completion events.
- [ ] Request generator: sample test prompts, draw each actual length from its completions **in the `original` config, clipped at `max_tokens` = 4,096 (rev 2)**, Poisson arrivals.
- [ ] Token budget per window (start at 200,000 tokens per minute): admit if the reservation fits, else 429; settle on completion, with duration = output tokens ÷ tokens per second.
- [ ] Run the `max_tokens` policy end to end.

**Done when:** one policy prints admitted, rejected and overshoot counts.

### Day 9: Policies and sweeps

- [ ] Add the fixed-percentile and model policies, using precomputed predictions.
- [ ] Add the wrongful-rejection metric: requests whose actual cost would have fit.
- [ ] Sweep budget sizes and arrival rates, 3 random seeds each.
- [ ] **(rev 2)** Report admitted requests vs. both `max_tokens` and the fixed per-model p95 policy.

**Done when:** a comparison table and a plot of admitted requests vs. arrival rate exist.

### Day 10: Misprediction strategies

- [ ] Strategy A: cap `max_tokens` at bound × margin; measure the truncation rate.
- [ ] Strategy B: a shared overdraft buffer sized from expected misses; measure overshoot.
- [ ] Choose the default and write the decision with its numbers. **Put A's truncation rate and B's overshoot in the same table: A has zero overshoot by construction (rev 2).**

**Done when:** the trade-off plot and decision paragraph are written. **Gate: simulator results final.**

### Day 11: Export to ONNX

- [ ] Export the regressor (`onnxmltools` for XGBoost, `torch.onnx.export` for the MLP).
- [ ] Embedding model: try an ONNX export; if it fights you, keep it in PyTorch on CPU.
- [ ] Parity test: ONNX and Python predictions agree within 1e-4 on 1,000 prompts.

**Done when:** the parity test passes.

### Day 12: The service

- [ ] FastAPI `POST /predict`, `/reserve` and `/settle`.
- [ ] Redis Lua script for atomic reserve; settle with `INCRBY` by (actual − reserved).
- [ ] `docker-compose.yml` with the API and Redis. **Install Docker Desktop or OrbStack before this day; neither is on the laptop yet (rev 2).**

**Done when:** predict → reserve → settle works with curl after `docker compose up`.

### Day 13: Tests and benchmarks

- [ ] Concurrency test: 1,000 parallel reserves never push spend past the cap.
- [ ] Latency benchmark with a small asyncio client: p50 and p99 for `/predict` and `/reserve`.
- [ ] Fix anything over target, or document the gap.

**Done when:** the non-functional targets are met or their gaps are written down.

### Day 14: Ship

- [ ] README: problem, method, results table, curve, simulator plot, how to run.
- [ ] Blog post draft, 800 to 1,200 words.
- [ ] Fresh-clone test: `make data train eval sim`, then `docker compose up`.
- [ ] Tag v1.0 and fill the resume bullet with your real numbers.

**Done when:** the repo is public and reproducible from a clean clone. **Gate: shipped.**

## Risks and mitigations

The biggest risk is that the model doesn't beat a fixed percentile; an honest negative result with a strong simulator still ships.

| Risk | Mitigation |
|---|---|
| Model doesn't beat the fixed per-model 95th percentile | Try one model per LLM and more length-hint features. If it still loses, publish the honest comparison and explain why; the simulator and service still stand. **The oracle-gap metric makes a partial win (for example 11–20%) readable (rev 2)** |
| **The 20% target is near the ceiling (rev 2)** | Not the case: the oracle ceiling is about 40%, so 20% is half the gap. Report gap closed either way |
| **Overfitting to ~8k training prompts (rev 2)** | Dropout, weight decay, early stopping on validation; compare XGBoost on hand-built features alone vs. with the embedding |
| CASTILLO covers open-source models only | Say so in the README; January adds data from a closed API model |
| Embedding pushes `/predict` past 15 ms | Cap MiniLM input length (see the latency check); else serve the hand-built-feature XGBoost instead and report the accuracy vs. latency trade-off |
| Prompts leak across splits | **Confirmed: 17 shared prompts (rev 2).** Fixed in FR1, with a zero-overlap test |
| Laptop runs out of memory | **Low risk (rev 2):** about 1.4M completion rows × ~30 float32 features is about 170 MB, and there are only ~11.4k unique prompts to embed. Read parquet with only the needed columns: the response-text columns are most of the file size |
| ONNX export fails for a model | Serve the Python model behind FastAPI and log the export as a follow-up |
| College or XCEED work eats days | Use the cut list below; day 7 is a built-in buffer |

**Cut list, in order, if you fall behind:**

1. The leave-one-dataset-out stretch check.
2. The linear quantile model (keep XGBoost and the MLP).
3. ONNX for the embedding model (keep it in PyTorch on CPU).
4. The blog post (the README stays).

Never cut the leakage checks, the matched-coverage evaluation, **the oracle row (rev 2)**, the simulator or the README: they are what make the project credible.

## Deliverables and definition of done

Version 1 is done when a stranger can clone the repo, reproduce the headline numbers, and run the service with one command each.

- [ ] Public GitHub repo `llm-reserve`, tagged v1.0, with tests passing in GitHub Actions.
- [ ] `docs/findings.md`: exploration plots, results table **with the oracle row (rev 2)**, coverage vs. reservation curve, error-analysis counts.
- [ ] Simulator results: policy comparison table, admitted-requests plot, misprediction trade-off and the chosen default.
- [ ] Service: `/predict`, `/reserve`, `/settle` running via `docker compose up`, with latency and concurrency results.
- [ ] README that opens with the headline numbers and the one-line problem statement.
- [ ] Blog post draft.

**Resume bullet** (fill the blanks with real numbers on day 14; never round up):

> Built llm-reserve, a learned output-length predictor (XGBoost and PyTorch, quantile loss) that cut reserved LLM budget by __% vs. a fixed-percentile baseline at 95% coverage (__% of the oracle ceiling), admitting __× more concurrent requests in a budget simulator; served via FastAPI, ONNX Runtime and atomic Redis reservations at __ ms p99.

## After version 1 (January)

Version 2 makes the project matter to people using Claude or Gemini APIs, then puts it in front of real users.

1. **Closed-API data:** run about 2,000 CASTILLO prompts through one cheap API model, 3 times each, and test whether the predictor transfers or needs a per-model head. **Vary temperature in this run, so sampling parameters can become real features (rev 2).**
2. **Streaming top-ups:** reserve more budget in chunks as tokens stream, and stop generation when the budget can't cover the next chunk.
3. **Online adaptation:** update the quantile from each app's own traffic, so the predictor fits its workload.
4. **Gateway integration:** a LiteLLM integration, then share results in the gateway issues where people report false rejections from worst-case reservations.

## Changelog

**Rev 2, 28 Oct 2026, after the day-0 data check**

1. **Leakage is real:** 17 prompts appear in more than one official split. FR1 now fixes this and a test asserts zero overlap.
2. **Oracle ceiling added:** the best prompt-only predictor can save about 40% vs. the fixed per-model p95, so the 20% target means closing half the gap. New metric: oracle gap closed. The oracle row is now on the never-cut list.
3. **`max_tokens` clarified:** CASTILLO's real cap is 15,000. The 4,096 baseline covers 99.96%, not 100%. The simulator clips actual cost at `max_tokens` and draws from `original`, where the tail is.
4. **Sampling parameters dropped as features:** they are fixed per model, so the model one-hot already carries them.
5. **Simulator comparison:** added admitted requests vs. the fixed per-model p95 policy. The 1.5× vs. `max_tokens` target is nearly guaranteed.
6. **Smaller fixes:** split name is `validate`; `category` is 82% undefined; effective sample size is ~8k prompts (MLP regularization); bucket edges fitted on train; MiniLM input capped at 128 tokens (13.4 ms p99 at the default 256); Python 3.12; install Docker before day 12.
