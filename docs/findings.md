# Findings

Living results log for llm-reserve. Each section says which split it uses. The test split is not explored; it is scored once, on Day 7.

## Day 2: Exploring the data (30 Oct 2026)

**Scope.** Train split only: 7,999 prompts, 97,827 (prompt, model) samples, 976,769 sanitized completions (978,270 original). Validate is kept for tuning, so it isn't explored here. Rebuild everything below with `make explore`; the numbers come from [`figures/explore_stats.json`](figures/explore_stats.json).

Overall output length (sanitized): p50 369, p90 776, **p95 1,005**, p99 1,649, p99.9 2,652 tokens.

### 1. The prompt decides length, not the model

How much of the variance in log(1 + output tokens) each grouping explains, if you knew its exact mean length (in-sample, so an upper bound):

| Grouping | Known at serving time? | Log-variance explained |
|---|---|---|
| Model | yes | **0.6%** |
| Source dataset | no | 13.0% |
| Model × source dataset | no | 23.7% |
| Prompt | via its text | **70.6%** |
| Prompt × model | via its text and the model | **93.4%** |

The fixed per-model p95 (the Day-3 bar) uses only the first row. Knowing the source dataset, which we can't at serving time, still leaves most of the signal unused. The signal is in what each prompt asks for, which matches the data check: per-model × dataset only saves 11%, and the oracle saves about 40%.

**So:** the prompt-text features (Day 4) are the whole project. A model that mostly learns "which LLM" or "which kind of task" will lose to the per-model p95 by very little or tie it.

### 2. Models disagree about which prompts are long

![Output length per model](figures/length_by_model.png)

Per-model p95 runs from 757 (Qwen2.5-32B) to 1,792 (gemma-3-12b). Eleven of the 13 models look alike: one main peak just below the p95, and a p95 between 757 and 977. The two gemma models are different: the **lowest medians (262 and 276) but the highest p95s (1,792 and 1,368)**, with a second peak around 1,500 tokens.

That second peak comes from the prompt type. gemma-3-12b is terse on code and verbose on chat; Qwen2.5-32B does the opposite:

| Source dataset | gemma-3-12b median | gemma-3-12b share > 1,000 | Qwen2.5-32B median | Qwen2.5-32B share > 1,000 |
|---|---|---|---|---|
| DS1000 (code) | 46 | 0% | 317 | 0% |
| Apps (code) | 196 | 1% | 576 | 1% |
| Alpaca (instructions) | 478 | 36% | 201 | 0% |
| ShareGPT (chat) | 1,369 | 67% | 455 | 4% |

Across all 13 models, the median correlation of per-prompt mean log length between two models is 0.81, so most prompts rank similarly. But going from "prompt" to "prompt × model" in the table above adds about 23 points of explained variance.

**So:** the model must interact with the prompt features, not just shift or scale them. XGBoost gets this from splits on the model one-hot. For the MLP (Day 6), feed the model embedding into the hidden layers, not only as an output offset. Gemma will dominate the per-model p95 cost, so it is also where a prompt-aware model can save the most.

### 3. Prompt length is a weak signal, and not a monotone one

![Output length vs. prompt length](figures/length_vs_input.png)

Spearman ρ between `input_size` and output length is only **0.16**. The p95 per input-length bin is U-shaped:

| Prompt tokens (bin median) | 13–35 | 41–81 | 102–305 | 377–654 | 1,193 |
|---|---|---|---|---|---|
| Output p95 | 1,026–1,543 | 813–947 | 685–781 | 816–993 | 1,466 |

Very short prompts are mostly open-ended chat and instructions (ShareGPT median input 56 tokens, output p95 1,541). Mid-length prompts are mostly self-contained code tasks (Mbpp, DS1000: output p95 593–623). The longest prompts get longer answers again (Apps, the code dataset with the longest prompts at a median of 449 input tokens, has an output median of 486).

**So:** this is why the per-model × input-length quintile baseline only saves 4.6%. Use log `input_size` and let trees split it; a linear term in raw length will fit badly. Short prompts need the text features most, because their length tells you least.

### 4. The same prompt to the same model varies a lot, and by how much varies too

![Within-prompt spread](figures/within_prompt_spread.png)

For each (prompt, model) pair, std ÷ mean across its completions: median **0.16**, p90 0.41, p99 0.98 (97,822 samples with at least 2 completions). The longest completion is 1.26× the median at p50, 1.94× at p90 and 5.6× at p99. About 7% of the log variance sits inside a (prompt, model) pair, and no prompt-only predictor can remove it. It is why even the oracle reserves 1.31× actual usage.

The spread itself varies a lot between prompts: a tenth of samples have a CV under 0.06 and a tenth have one above 0.41.

**So:** the 95% bound can't be "predicted mean × a constant". It needs the spread too, which is why the models are quantile models (pinball loss) and not mean regressors with a fixed multiplier. On Day 7, check whether under-predictions are concentrated in high-spread prompts.

### 5. Degenerate runs are a small-model problem; gemma's long outputs are real

![Degenerate output per model](figures/degenerate_by_model.png)

In the original config, **5.65% of Llama-3.2-1B samples have at least one degenerate completion** (0.95% of its completions), then Llama-3.1-8B (2.68%), Llama-3.2-3B (1.93%) and Phi-4-mini (1.47%). The other nine models are at 0.54% or less. gemma-3-12b and Ministral-8B have none.

So gemma's heavy tail (finding 2) is not a decoding failure that sanitized hides: it is real, verbose output, and it is in the training data the model learns from.

Original vs. sanitized on train: 0.181% vs. 0.028% of completions are over 4,096 tokens, and 1,482 original completions hit the 15,000 cap. Only 2 samples are degenerate in all 10 completions (both Qwen, all at 15,001 tokens).

**So:** the simulator draws actual lengths from `original` (as planned), and its clipped-request and overshoot counts should be broken down per model, since four small models produce most of the runaway generations. A model trained on sanitized data will not predict these; that's expected and it's what the misprediction strategies on Day 10 are for.

## Day 3: Baselines locked (30 Oct 2026)

**Scope.** Validation split, sanitized config: 2,266 prompts, 277,351 completions, mean length 421. Every percentile and the input-length quintile edges are fitted on train; only the knob θ (which percentile, or the oracle's k) is tuned on validation, to the smallest value that reaches 95% coverage. Bounds are rounded up to whole tokens. Rebuild with `make baselines`; the numbers come from [`figures/baselines_validate.json`](figures/baselines_validate.json).

| Method | Deployable | Knob θ | Coverage (sanitized) | Coverage (original) | Reservation ratio | vs. the bar | Oracle gap closed |
|---|---|---|---|---|---|---|---|
| `max_tokens` = 4,096 | yes | — | 99.96% | 99.80% | 9.723 | +333.0% | — |
| Global percentile | yes | 0.9463 | 95.01% | 94.86% | 2.322 | +3.4% | −8.1% |
| **Per-model percentile (the bar)** | yes | 0.9539 | 95.00% | 94.85% | **2.246** | — | 0% |
| Per-model × input-length quintile | yes | 0.9529 | 95.00% | 94.85% | 2.142 | −4.6% | 11.1% |
| Per-model × source dataset | no | 0.9535 | 95.00% | 94.85% | 1.993 | −11.3% | 27.0% |
| Oracle: k × own mean + own std | no | 1.1093 | 95.00% | 94.94% | **1.309** | −41.7% | 100% |

"Coverage (original)" applies the same frozen θ to the original config, degenerate completions included (the oracle recomputes its mean and std there).

**The bar is 2.246.** A model hits the 20% target at a reservation ratio of **1.797** or less at 95% coverage on test, which is closing 48% of the gap to the oracle. These numbers reproduce the PRD data check (2.25 / 2.14 / 2.00 / 1.31) on the overlap-fixed validation split, so the baselines are locked.

### 6. The bar is only the strongest simple baseline near 95%

![Coverage vs. reservation](figures/coverage_curve.png)

Below about 93% coverage, **one global percentile reserves less than the per-model percentile** (at 90%: 1.84 vs. 1.90). A per-model percentile gives every model the same coverage, which forces it to cover gemma's long second peak (finding 2) at the same rate as everyone else. A global percentile lets gemma's coverage drop and spends the savings on the other eleven models. Above 93%, the per-model percentile wins because the global one has to cover gemma's tail with everyone's bound.

**So:** equal coverage per model is not the cheapest way to reach 95% overall. A learned model is free to cover some models less, and that has to be visible: on Day 7, report coverage per model (and per source dataset) next to the overall number, so a win is not bought by quietly under-covering gemma.

### 7. Input length adds little, the dataset adds more, and both are small next to the prompt

Per-model × input-length quintile saves 4.6%. Knowing the source dataset, which the gateway can't, saves 11.3%. The oracle saves 41.7%. The gap between the bar and the oracle widens as coverage rises: 0.38 at 80% coverage (1.55 vs. 1.17), 0.94 at 95% and 1.68 at 99% (3.27 vs. 1.59). The tail is where a prompt-aware bound pays most. The sanitized-tuned θ loses 0.15 points of coverage on the original config for every percentile baseline, which is the degenerate tail that the simulator (Day 8) will have to absorb.
