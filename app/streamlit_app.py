"""llm-reserve demo: type a prompt, see how many output tokens a gateway should reserve for it.

Run locally:   streamlit run app/streamlit_app.py
Deploy:        see docs/deploy.md (Streamlit Community Cloud or a Hugging Face Space; both are free).

The page runs the real model (the Day-11 ONNX bundle with the int8 encoder) in this process. There is no Redis and no
API behind it: the budget demo and the simulator tab are computed from the bundle and from the simulator results
that `make sweep` and `make mispredict` wrote.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # the app folder deploys without `pip install -e .`

from llm_reserve.demo import (  # noqa: E402
    EXAMPLES, MARGIN, MAX_TOKENS, POLICY_LABELS, Demo, estimate_input_tokens, reserved_per_policy, short_name,
)

st.set_page_config(page_title="llm-reserve", page_icon="🧮", layout="wide")


@st.cache_resource(show_spinner="Loading the model (63 MB, one time)…")
def load_demo() -> Demo:
    return Demo.load(ROOT / "app", threads=2)


@st.cache_data(show_spinner=False)
def compare(text: str, input_tokens: int, max_tokens: int):
    return load_demo().compare(text, input_tokens, max_tokens)


@st.cache_data(show_spinner=False)
def queue(prompts: tuple[str, ...], model: str, max_tokens: int):
    return load_demo().queue(list(prompts), model, max_tokens)


demo = load_demo()
H = demo.headline

st.title("llm-reserve")
st.markdown(
    "**A spend gateway has to hold budget for a request before it knows how long the answer will be, so it holds the "
    "worst case, `max_tokens`, and turns away requests it could have served. This predicts a tight 95% bound on the "
    "answer's length from the prompt alone, so it can hold what the request will likely cost.**"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fewer tokens reserved vs. p95", f"{-H['vs_bar_matched']:.1%}",
          help="Fewer reserved tokens than a fixed per-model 95th percentile. Test split, scored once, both methods re-tuned to exactly 95% coverage. "
               f"{-H['vs_bar']:.1%} at the knobs frozen on validation.")
c2.metric("Oracle gap closed", f"{H['gap_closed']:.0%}",
          help="The oracle knows each prompt's true length distribution; it is the ceiling, not a method.")
c3.metric("Admitted vs. max_tokens", f"{demo.headline_ratio():.3f}×",
          help="Simulator at 8 requests/s, 200k tokens/min, 50 tokens/s, with the 1.5× cap: demand is 153% of budget.")
c4.metric("Coverage (test)", f"{H['winner']['coverage']:.1%}", help="Share of completions inside their bound.")

tab_try, tab_budget, tab_sim, tab_about = st.tabs(["Try a prompt", "Budget demo", "Simulator", "Results and limits"])

# --------------------------------------------------------------------------------------------------------------
with tab_try:
    left, right = st.columns([3, 2])
    with left:
        pick = st.selectbox("Start from an example", ["(my own)"] + list(EXAMPLES), index=2, key="pick")
        default = EXAMPLES.get(pick, "")
        prompt = st.text_area("Prompt", value=default, height=140, key=f"prompt-{pick}",
                              help="Only the text is used. Nothing leaves this page.")
    with right:
        model_key = st.selectbox("Target model", demo.models, index=demo.models.index("meta-llama--Llama-3.1-8B-Instruct"),
                                 format_func=short_name)
        max_tokens = st.number_input("Client's max_tokens", 64, 15000, MAX_TOKENS, step=256,
                                     help="What the gateway would hold without a prediction.")
        n_in = st.number_input("Input tokens", 1, 100_000, max(1, estimate_input_tokens(prompt)),
                               key=f"n_in-{hash(prompt)}",
                               help="Estimated at 4 characters a token here; a real gateway knows the exact count.")

    if not prompt.strip():
        st.info("Type a prompt above.")
    else:
        table = compare(prompt, int(n_in), int(max_tokens))
        row = table[table.key == model_key].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted 95% bound", f"{row.bound:,} tokens", help="95 of 100 answers should be at most this long.")
        m2.metric("Learned reserve", f"{row.reserve_learned:,}", help=f"min(⌈{MARGIN} × bound⌉, max_tokens), what /reserve holds "
                                                                       "and sends as max_tokens.")
        m3.metric("Fixed per-model p95", f"{row.reserve_bar:,}", delta=f"{-row.saved_vs_bar:.0%} with the learned bound",
                  delta_color="inverse", help="The same 1.5× cap on the per-model 95th percentile of training answers.")
        m4.metric("Plain max_tokens", f"{row.reserve_max:,}", delta=f"{-row.saved_vs_max:.0%} with the learned bound",
                  delta_color="inverse")

        st.caption("Reservation per target model for this prompt (tokens held before the call). Lower means more "
                   "requests fit in the same budget.")
        chart = table.set_index("model")[["reserve_learned", "reserve_bar"]].rename(
            columns={"reserve_learned": "Learned bound", "reserve_bar": "Fixed per-model p95"})
        st.bar_chart(chart, height=320)
        with st.expander("Numbers for every model"):
            st.dataframe(
                table[["model", "bound", "bar", "reserve_learned", "reserve_bar", "reserve_max"]].rename(columns={
                    "model": "Model", "bound": "Learned 95% bound", "bar": "Per-model p95",
                    "reserve_learned": "Learned reserve", "reserve_bar": "p95 reserve", "reserve_max": "max_tokens"}),
                hide_index=True, use_container_width=True)
        st.caption(
            "This is a 95% bound, so about 1 answer in 20 is longer. The gateway caps the call at the reserve, so "
            "it can never overspend; in the Day-10 test 0.62% of real answers were cut short by that cap."
        )

# --------------------------------------------------------------------------------------------------------------
with tab_budget:
    st.markdown(
        "Requests arrive in a row (12 different ones, repeated) at one gateway with a token budget for the minute. Each policy "
        "holds a reservation per request and admits it only if it still fits. **Same budget, same requests**; the only "
        "difference is how much each policy holds back."
    )
    b1, b2, b3 = st.columns(3)
    budget = b1.slider("Budget for the window (tokens)", 5_000, 60_000, 30_000, step=1_000)
    model_b = b2.selectbox("Target model ", demo.models, index=demo.models.index("meta-llama--Llama-3.1-8B-Instruct"),
                           format_func=short_name, key="model_b")
    rounds = b3.slider("Repeat the 12 requests", 1, 6, 3, help="More repeats, a longer queue and a fuller window.")
    q = queue(tuple(EXAMPLES.values()), model_b, MAX_TOKENS)
    q = pd.concat([q] * rounds, ignore_index=True)
    summary = reserved_per_policy(q, budget)
    cols = st.columns(3)
    for col, (_, r) in zip(cols, summary.iterrows(), strict=True):
        col.metric(r.policy, f"{r.admitted} of {r['of']} admitted", delta=f"{r.share:.0%} of the budget held",
                   delta_color="off")
    st.bar_chart(summary.set_index("policy")[["admitted"]], height=240)
    order = [p for p in POLICY_LABELS]
    detail = q.assign(prompt=q.prompt.str.slice(0, 70)).rename(
        columns={"max_tokens": POLICY_LABELS["max_tokens"], "bar": POLICY_LABELS["bar"], "learned": POLICY_LABELS["learned"]})
    with st.expander("Each request's reservation"):
        st.dataframe(detail[["prompt", *[POLICY_LABELS[p] for p in order]]], hide_index=True, use_container_width=True)
    st.caption(
        "Reservations only: the demo does not know how long each real answer will be, so nothing is refunded. In the "
        "service, `/settle` returns the unused part, which is why a tight reservation matters most when many "
        "requests are in flight at once. That is what the Simulator tab measures."
    )

# --------------------------------------------------------------------------------------------------------------
with tab_sim:
    st.markdown(
        "A discrete-event simulation of a gateway with a token budget per 60-second window, replaying the held-out "
        "test split (13 models, 1,130 prompts), 3 seeds per setting. Each request is billed to the window that "
        "admitted it, and rejected requests are dropped."
    )
    scen = demo.scenarios()
    s1, s2, s3 = st.columns([2, 2, 2])
    name = s1.selectbox("Budget and decode speed", list(scen), index=1)
    metric = s2.selectbox("Show", ["Admitted requests, as a multiple of max_tokens", "Admitted requests",
                                   "Largest overshoot of the budget (% of a window)", "Wrongful rejections"])
    capped = s3.toggle("With the 1.5× cap (the default)", value=True,
                       help="The cap sends each call with max_tokens = 1.5 × the bound, so no request can cost "
                            "more than it reserved: zero overshoot by construction.")
    sweep = demo.sweep(**scen[name])
    keep = {"max_tokens": "max_tokens", "bar_capped" if capped else "bar": "Fixed per-model p95",
            "winner_capped" if capped else "winner": "Learned bound"}
    sweep = sweep[sweep.policy.isin(keep)].assign(policy=lambda d: d.policy.map(keep))
    col = {"Admitted requests, as a multiple of max_tokens": "vs_max_tokens", "Admitted requests": "admitted",
           "Largest overshoot of the budget (% of a window)": "overshoot_max_share",
           "Wrongful rejections": "wrongful_rejections"}[metric]
    sweep = sweep.assign(value=sweep[col] * (100 if col == "overshoot_max_share" else 1))
    wide = sweep.pivot(index="rate", columns="policy", values="value")
    wide.index.name = "Arrival rate (requests per second)"
    st.line_chart(wide, height=380)
    st.caption(
        "Below saturation every policy but max_tokens admits everything, so the lines meet at 1. The learned bound "
        "pulls ahead only once the budget binds. At 100 tokens/s the gain over max_tokens stays small because fast "
        "decoding returns oversized reservations before they cost much. Without the cap, every bound below "
        "max_tokens overshoots once the budget binds, by up to 10.9% of a window."
    )
    with st.expander("Table"):
        st.dataframe(wide.reset_index(), hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------------------------------------------
with tab_about:
    st.subheader("Held-out test results")
    st.markdown(
        f"On the test split ({H['prompts']:,} prompts, {H['completions']:,} completions), scored once with every choice "
        f"frozen on validation:\n\n"
        f"| | Coverage | Reserved ÷ used |\n|---|---|---|\n"
        f"| Fixed per-model p95 (the bar) | {H['bar']['coverage']:.2%} | {H['bar']['ratio']:.3f} |\n"
        f"| **Learned bound (average of an MLP and XGBoost)** | **{H['winner']['coverage']:.2%}** | **{H['winner']['ratio']:.3f}** |\n"
        f"| Oracle (not deployable) | 95.00% | {H['oracle_ratio']:.3f} |\n\n"
        f"That is **{-H['vs_bar']:.1%} fewer reserved tokens than the bar** ({-H['vs_bar_matched']:.1%} when both are "
        f"re-tuned to exactly 95%), closing {H['gap_closed']:.0%} of the gap to the oracle, and "
        f"{1 - H['vs_max_tokens']:.0%} fewer than `max_tokens` = 4,096."
    )
    st.subheader("What it does not do")
    st.warning(
        f"**On kinds of prompt it never saw, the saving falls to about {-H['unseen_dataset']['matched_vs_bar']:.0%}** "
        f"and coverage to {H['unseen_dataset']['coverage']:.1%} (77% on real chat from ShareGPT), because about half "
        "of the gain is recognizing CASTILLO's seven benchmark templates. Re-tune the quantile on your own traffic "
        "before trusting the 95%."
    )
    st.markdown(
        "- **Open models only.** The data (CASTILLO) covers 13 open models, with fixed sampling per model.\n"
        "- **Runaways are invisible to a prompt-only bound.** The cap handles their cost; streaming top-ups would "
        "avoid truncating them.\n"
        "- **Latency.** At one client the service answers `/predict` in 12.8 ms p99 and `/reserve` in 0.45 ms p99 "
        "(Docker, M5 laptop); under concurrent load both queue.\n"
        "- **This page** runs the int8 ONNX model in-process, with no Redis. The service in the repo is the real "
        "gateway component."
    )
    st.markdown("Code, method and every number: [github.com/guptakaran20/LLM-Reserve](https://github.com/guptakaran20/LLM-Reserve)"
                " · data: [CASTILLO](https://huggingface.co/datasets/danfperam/castillo) (CC BY 4.0)")
