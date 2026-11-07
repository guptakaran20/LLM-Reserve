"""llm-reserve demo: type a prompt, see how many output tokens a gateway should reserve for it.

Run locally:   streamlit run app/streamlit_app.py
Deploy:        see docs/deploy.md (Streamlit Community Cloud or a Hugging Face Space; both are free).

The page runs the real model (the Day-11 ONNX bundle with the int8 encoder) in this process. There is no Redis and no
API behind it: the budget demo and the simulator tab are computed from the bundle and from the simulator results
that `make sweep` and `make mispredict` wrote.
"""

from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # the app folder deploys without `pip install -e .`

from llm_reserve.demo import (  # noqa: E402
    EXAMPLES, MARGIN, MAX_TOKENS, POLICY_LABELS, Demo, estimate_input_tokens, reserved_per_policy, short_name,
)

st.set_page_config(page_title="llm-reserve", layout="wide")


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

GREY, AMBER, BLUE = "#9aa4b2", "#e0a030", "#2b6cd6"
st.markdown(
    """
<style>
.block-container {padding-top: 2.4rem; max-width: 1150px;}
.eyebrow {font-size:.78rem; letter-spacing:.14em; text-transform:uppercase; color:#6b7280; margin-bottom:.2rem;}
.title {font-size:2.6rem; font-weight:700; letter-spacing:-.02em; line-height:1.1; margin:0 0 .6rem 0;}
.lede {font-size:1.08rem; line-height:1.6; max-width: 46rem; margin:0 0 1.4rem 0; opacity:.85;}
.lede code {background: rgba(128,128,128,.16); padding:0 .3rem; border-radius:3px;}
.rule {border:0; border-top:1px solid rgba(128,128,128,.3); margin:0 0 1.2rem 0;}
[data-testid="stMetric"] {background:transparent; border:0; border-left:3px solid #2b6cd6; border-radius:0;
       padding:.1rem 0 .1rem .8rem;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums; font-weight:650;}
.ruler {margin: .5rem 0 1rem 0;}
.ruler .lab {display:flex; justify-content:space-between; font-size:.9rem; margin-bottom:.25rem;}
.ruler .track {background: rgba(128,128,128,.16); height: 10px;}
.ruler .fill {height: 100%;}
.chips {display:flex; flex-wrap:wrap; gap:3px; margin:.25rem 0 .9rem 0;}
.chip {width:14px; height:14px; border-radius:2px;}
.chip.ok {background:#2b6cd6;} .chip.no {background:rgba(128,128,128,.25);}
.chiprow {font-size:.9rem; margin-top:.6rem;}
.live {font-size:.9rem; color:#6b7280; margin:.4rem 0 .8rem 0;}
.live b {color:#2b6cd6;}
</style>
<div class="eyebrow">Output-length prediction for LLM spend gateways</div>
<div class="title">llm-reserve</div>
<div class="lede">A gateway has to hold budget for a request before it knows how long the answer will be, so it holds the
worst case, <code>max_tokens</code>, and turns away requests it could have served. This predicts a tight 95% bound on the
answer's length from the prompt alone, so it can hold what the request will likely cost.</div>
<hr class="rule">

""",
    unsafe_allow_html=True,
)


def ruler(label: str, tokens: int, full: int, color: str) -> str:
    pct = min(100, 100 * tokens / full)
    return (f'<div class="ruler"><div class="lab"><span>{html.escape(label)}</span><b>{tokens:,} tokens</b></div>'
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%;background:{color}"></div></div></div>')


def chips(admitted: list[bool]) -> str:
    return '<div class="chips">' + "".join(f'<div class="chip {"ok" if a else "no"}"></div>' for a in admitted) + "</div>"


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
    st.session_state.setdefault("prompt_text", "")
    st.session_state.setdefault("committed", "")

    def _load_example():
        ex = EXAMPLES.get(st.session_state["pick"], "")
        st.session_state["prompt_text"] = ex
        st.session_state["committed"] = ex

    left, right = st.columns([3, 2])
    with left:
        st.selectbox("Or start from an example", ["(my own)"] + list(EXAMPLES), key="pick", on_change=_load_example)
        with st.form("prompt-form", border=False):
            st.text_area("Your prompt", key="prompt_text", height=140,
                         placeholder="Type any prompt here, then press Compute…",
                         help="Only the text is used, and it is scored live by the model on this page. Nothing leaves it.")
            go = st.form_submit_button("Compute reservation", type="primary")
        if go:
            st.session_state["committed"] = st.session_state["prompt_text"]
    prompt = st.session_state["committed"]
    with right:
        model_key = st.selectbox("Target model", demo.models, index=demo.models.index("meta-llama--Llama-3.1-8B-Instruct"),
                                 format_func=short_name)
        max_tokens = st.number_input("Client's max_tokens", 64, 15000, MAX_TOKENS, step=256,
                                     help="What the gateway would hold without a prediction.")
        n_in = st.number_input("Input tokens", 1, 100_000, max(1, estimate_input_tokens(prompt)),
                               key=f"n_in-{hash(prompt)}",
                               help="Estimated at 4 characters a token here; a real gateway knows the exact count.")

    if not prompt.strip():
        st.info("Type a prompt above and press **Compute reservation**, or pick an example. The model runs live on "
                "whatever you write.")
    else:
        t0 = time.perf_counter()
        table = demo.compare(prompt, int(n_in), int(max_tokens))
        ms = (time.perf_counter() - t0) * 1000
        row = table[table.key == model_key].iloc[0]
        st.markdown(f'<div class="live"><b>Live:</b> scored by the int8 ONNX model in {ms:.0f} ms for all {len(table)} '
                    "models. Nothing is precomputed for your text.</div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted 95% bound", f"{row.bound:,} tokens", help="95 of 100 answers should be at most this long.")
        m2.metric("Learned reserve", f"{row.reserve_learned:,}", help=f"min(⌈{MARGIN} × bound⌉, max_tokens), what /reserve holds "
                                                                       "and sends as max_tokens.")
        m3.metric("Fixed per-model p95", f"{row.reserve_bar:,}", delta=f"{-row.saved_vs_bar:.0%} with the learned bound",
                  delta_color="inverse", help="The same 1.5× cap on the per-model 95th percentile of training answers.")
        m4.metric("Plain max_tokens", f"{row.reserve_max:,}", delta=f"{-row.saved_vs_max:.0%} with the learned bound",
                  delta_color="inverse")

        with st.expander("Fixed per-model p95 vs. learned bound: what is the difference?"):
            st.markdown(
                "- **Fixed per-model p95** looks only at the target model. Over its training answers, 95% were at most N "
                "tokens, so every request to that model gets the same N, whether it asks for a one-word answer or an essay.\n"
                "- **Learned bound** reads *your prompt* (its size, code, length hints like \"in 500 words\", and a "
                "meaning embedding) plus the model, and predicts a 95% bound for that request. Short questions get a small "
                "reserve, long-form asks a large one.\n\n"
                "Both aim at the same 95% coverage. The learned one wastes less because it varies per prompt: "
                f"{-H['vs_bar_matched']:.1%} fewer reserved tokens on held-out test data."
            )
        st.markdown("**How much budget each policy holds for this one request**")
        st.markdown(ruler("Reserve max_tokens", int(row.reserve_max), int(row.reserve_max), GREY)
                    + ruler("Fixed per-model p95", int(row.reserve_bar), int(row.reserve_max), AMBER)
                    + ruler("Learned bound (llm-reserve)", int(row.reserve_learned), int(row.reserve_max), BLUE),
                    unsafe_allow_html=True)

        st.caption("Same prompt sent to each target model (tokens held before the call). Lower means more "
                   "requests fit in the same budget.")
        chart = table.set_index("model")[["reserve_learned", "reserve_bar"]].rename(
            columns={"reserve_learned": "Learned bound", "reserve_bar": "Fixed per-model p95"})
        st.bar_chart(chart, height=320, color=[BLUE, AMBER])
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
        "Requests arrive in a row at one gateway with a token budget for the minute. Each policy holds a reservation "
        "per request and admits it only if it still fits. **Same budget, same requests**; the only difference is how "
        "much each policy holds back. **Edit the queue below** (one prompt per line) to try your own."
    )
    queue_text = st.text_area("Request queue, one prompt per line (up to 60)", value="\n".join(EXAMPLES.values()),
                              height=130, key="queue_text")
    b1, b2, b3 = st.columns(3)
    budget = b1.slider("Budget for the window (tokens)", 5_000, 60_000, 30_000, step=1_000)
    model_b = b2.selectbox("Target model ", demo.models, index=demo.models.index("meta-llama--Llama-3.1-8B-Instruct"),
                           format_func=short_name, key="model_b")
    rounds = b3.slider("Repeat the queue", 1, 6, 3, help="More repeats, a longer queue and a fuller window.")
    prompts = [ln.strip() for ln in queue_text.splitlines() if ln.strip()][:60]
    if not prompts:
        st.info("Add at least one prompt to the queue.")
    else:
        q = demo.queue(prompts, model_b, MAX_TOKENS)
        q = pd.concat([q] * rounds, ignore_index=True)
        summary = reserved_per_policy(q, budget)
        cols = st.columns(3)
        for col, (_, r) in zip(cols, summary.iterrows(), strict=True):
            col.metric(r.policy, f"{r.admitted} of {r['of']}", delta=f"{r.share:.0%} of the budget held",
                       delta_color="off")
        from llm_reserve.demo import fill_budget
        st.markdown("**Requests admitted out of the queue (each square is a request: green admitted, grey turned away)**")
        for p, color in zip(POLICY_LABELS, (GREY, AMBER, BLUE), strict=True):
            f = fill_budget(q[p].tolist(), budget)
            st.markdown(f'<div class="chiprow"><span style="color:{color}">●</span> {POLICY_LABELS[p]}'
                        f' — {f["count"]} of {len(q)}</div>' + chips(f["admitted"]), unsafe_allow_html=True)
        st.bar_chart(summary.set_index("policy")[["admitted"]], height=220, color=BLUE)
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
        "Unlike the other tabs, this one replays results that `make sweep` and `make mispredict` computed offline (a full simulation takes minutes). A discrete-event simulation of a gateway with a token budget per 60-second window, replaying the held-out "
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
    st.line_chart(wide, height=380, color=[AMBER, BLUE, GREY])
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
