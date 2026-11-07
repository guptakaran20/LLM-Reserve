# Reserving what an LLM call will cost, not what it could cost

*Draft, 5 November 2026. Code and every number: [llm-reserve](../README.md); the day-by-day log is [findings.md](findings.md).*

Put a spend limit in front of an LLM API and you hit an awkward fact: you have to decide whether a request fits in the budget before you know what it costs. Input tokens are known up front; output tokens aren't. So most gateways reserve the worst case, `max_tokens`, and refund the difference when the provider reports real usage.

That is safe and wasteful. In the simulator I built for this project, a gateway reserving `max_tokens` = 4,096 turned away one request in six at a load its budget could absorb, while a third of the budget went unspent. 85% of those rejections were wrongful: the request would have fit. A typical answer is a few hundred tokens; the reservation is ten times that.

The fix is to reserve less, if a model can say how much less per request.

## The setup

I used CASTILLO, a public dataset of 11,395 prompts from seven benchmarks (Alpaca, Dolly, ShareGPT, Apps, Mbpp, DS1000, BigCodeBench), each sent ten times to 13 open models from 1B to 70B parameters.

The goal: from the prompt, the target model and the input size, predict a bound the real output stays under 95% of the time, tighter than the honest baseline. That baseline isn't `max_tokens`, which any percentile beats. It is a fixed 95th percentile per model, which I called the bar.

Two choices kept the comparison honest. **Matched coverage:** a predictor that covers 93% of requests always reserves less than one that covers 96%, so every method gets one knob, tuned on validation to the smallest value that reaches 95%. **An oracle:** a bound that knows each prompt's true mean and spread reserves 1.309× what is used, against the bar's 2.246× on validation. A prompt-only predictor can be at most about 40% better than the bar, so a 20% target is half the gap, and "closes 50% of the gap" says more than "saves 20%".

Checking the data first also found 17 prompts in more than one official split. They move to their earliest split, and tests assert zero overlap and that none of the dataset's own length statistics can become a feature.

## The model

The features are what a gateway has before it sends a request: 29 hand-built features of the prompt (size, code fences, length hints like "in 500 words"), the target model, the input size, and a MiniLM embedding of the first 128 tokens, compressed to 32 dimensions. Length hints are precise but rare: only about 1% of prompts ask for a word count. Most of the signal is in the embedding; a linear probe went from 14.9% of variance explained to 37.5% with it.

Two quantile regressors learn the 95th percentile of log length with the pinball loss: XGBoost and a small PyTorch MLP. Both overfit at first. Each prompt appears 130 times, so the network memorized prompts within a few epochs until I added dropout on its input, and the trees overfit the raw embedding until it was compressed. Neither model alone hit the target on validation. The average of the two did.

## Test, scored once

The test split was read once, after every choice was frozen. At 95% coverage the averaged model reserves **1.753× what is used, against the bar's 2.203×: 20.4% less, half the gap to the oracle** (−21.8% at the knobs frozen on validation). Every target model lands between 95.0% and 96.5% coverage. That clears the target by 0.4 points, and I'd rather report a thin margin than tune it wider.

The more interesting number is a stress test. CASTILLO's prompts come from seven benchmarks with recognizable templates, and the embedding can name a prompt's source 77% of the time. So I held out each benchmark in turn, retrained without it, and scored it. **On a benchmark it never saw, the saving falls to 9.3%, and coverage to 93.9% on average and 77.4% on ShareGPT**, the real-user chat set and the one most like production traffic. Removing the same amount of random data instead keeps the full 20%. About half the gain is template recognition. A deployment has to re-tune the quantile on its own traffic before trusting the 95%.

## What a tighter bound buys a gateway

A smaller reservation is an offline number. To see what it does to a gateway, I wrote a discrete-event simulator: a token budget per minute, Poisson arrivals, a decode speed, each request billed to the window that admitted it.

**Below saturation, nothing changes.** If the budget fits all demand under a per-model percentile, a tighter bound admits nothing extra. Once the budget binds, the learned bound admits 1–10% more than the bar, mostly by admitting cheaper requests: short Alpaca questions get in at 73% under overload, long coding answers at 63%. That is a choice the operator should know about.

**Every bound below `max_tokens` overshoots once the budget binds**, by up to 10.85% of a window: a 95% bound misses 5% of the time, and in a full window every miss is money past the cap. An overdraft buffer sized on validation missed its 1% target on test (2.56%). A cap worked: send each call with `max_tokens` at 1.5× the bound and reserve exactly that. Overshoot is zero by construction, and 0.62% of real answers get cut short, visible to the client as `finish_reason = length`.

**Decode speed matters as much as the budget.** With the cap, the learned bound admits 1.476× as many requests as `max_tokens` when demand is 153% of the budget, but at 100 tokens a second never more than 1.22×, because fast decoding returns oversized reservations before they cost much.

## Serving it

FastAPI serves three calls: `/predict` for the bound, `/reserve` to hold budget, `/settle` with real usage. The model runs on ONNX Runtime alone, and reserve and settle are Redis Lua scripts, so admission is atomic across workers and a retried settle can't refund twice.

The benchmarks found two surprises. The encoder ran at 12.5 ms p99 on macOS and 33 ms in Docker on the same laptop, because Docker's Linux VM can't use the chip's matrix units; a per-channel int8 encoder, checked against a coverage rule written before the run, brought `/predict` to 12.8 ms. And 1,000 simultaneous reserves returned 511 server errors, because redis-py's default pool raises instead of waiting. With a blocking pool the window filled to 199,977 of 200,000 tokens and never past it.

## Next

CASTILLO has open models only, so the next step is running prompts through a closed API model to see whether the predictor transfers. The cap should give way to streaming top-ups, where a long answer asks for more budget instead of being cut. And since calibration doesn't transfer to new kinds of prompts, the quantile should adapt online to each deployment's traffic.

The lesson was about measurement more than modelling. An oracle that showed how much room there was, comparisons at matched coverage, a test split read once, and a stress test that halved the headline did more for this project than any model choice.
