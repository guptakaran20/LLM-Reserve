import numpy as np
import pandas as pd
import pytest

from llm_reserve import simulate as sim
from llm_reserve.data import OUT_DIR


def trace(arrival, inputs, outputs, models=None, datasets=None) -> sim.Trace:
    n = len(arrival)
    return sim.Trace(
        arrival=np.asarray(arrival, dtype=float), row=np.arange(n),
        input_tokens=np.asarray(inputs, dtype=np.int64), output_tokens=np.asarray(outputs, dtype=np.int64),
        model=np.asarray(models or ["m"] * n), dataset=np.asarray(datasets or ["d"] * n),
    )


CFG = sim.Config(budget=100, window=60.0, duration=120.0, tokens_per_second=1.0, max_tokens=80)


def test_admits_while_the_reservation_fits_and_429s_after():
    t = trace([0, 1, 2], [10, 10, 10], [5, 5, 5])
    res = sim.simulate(t, [40, 40, 40], CFG)  # reserves 50 each; each runs 5 s, so none settles in time
    assert res.admitted.tolist() == [True, True, False]
    s = sim.summarize(t, res, CFG)
    assert (s["admitted"], s["rejected"], s["overshoot_windows"]) == (2, 1, 0)


def test_settling_gives_the_unused_reservation_back_within_the_window():
    # A reserves 10 + 80 and uses 10 + 20, finishing at t = 20. B needs 60 more: no at t = 19, yes at t = 20.
    early = sim.simulate(trace([0, 19], [10, 10], [20, 5]), [80, 50], CFG)
    on_time = sim.simulate(trace([0, 20], [10, 10], [20, 5]), [80, 50], CFG)
    assert early.admitted.tolist() == [True, False]
    assert on_time.admitted.tolist() == [True, True]  # the completion at t = 20 is processed before the arrival
    assert on_time.spend[0] == 30 + 15


def test_a_new_window_starts_empty_and_a_late_settle_is_billed_to_its_own_window():
    # A is admitted at t = 50 and runs 30 s, into window 1. B at t = 61 still fits window 1's full budget.
    t = trace([50, 55, 61], [10, 10, 10], [30, 0, 0])
    res = sim.simulate(t, [80, 80, 80], CFG)
    assert res.admitted.tolist() == [True, False, True]
    assert res.spend.tolist() == [40, 10]  # A's real cost lands on window 0, which admitted it


def test_overshoot_is_real_spend_past_the_budget_once_everything_settles():
    t = trace([0, 1], [10, 10], [70, 70], models=["a", "b"], datasets=["x", "y"])
    res = sim.simulate(t, [30, 30], CFG)  # 40 reserved each, 80 used each
    s = sim.summarize(t, res, CFG)
    assert res.admitted.all() and res.spend[0] == 160
    assert (s["overshoot_windows"], s["overshoot_max_tokens"], s["overshoot_max_share"]) == (1, 60, 0.6)
    assert (s["under_reserved_requests"], s["under_reserved_tokens"]) == (2, 80)
    by = sim.group_table(t.dataset, t, res)
    assert by["x"]["under_reserved_tokens"] == by["y"]["under_reserved_tokens"] == 40


def test_max_tokens_caps_both_the_reservation_and_the_real_cost():
    t = trace([0], [10], [500])  # the provider stops at max_tokens = 80
    res = sim.simulate(t, [1_000], CFG)
    assert res.reserved[0] == res.actual[0] == 90
    s = sim.summarize(t, res, CFG)
    assert s["truncated"] == 1 and s["overshoot_windows"] == 0


def test_the_max_tokens_policy_never_reserves_less_than_a_request_costs():
    rng = np.random.default_rng(0)
    n = 400
    t = trace(np.sort(rng.uniform(0, 120, n)), rng.integers(1, 30, n), rng.integers(0, 300, n))
    res = sim.simulate(t, sim.max_tokens_policy(CFG.max_tokens)(t), CFG)
    s = sim.summarize(t, res, CFG)
    assert s["overshoot_windows"] == 0 and s["under_reserved_tokens"] == 0
    assert (res.spend <= CFG.budget).all() and res.spend.sum() == res.actual[res.admitted].sum()


def test_poisson_arrivals_have_the_right_rate_and_stay_in_range():
    t = sim.poisson_arrivals(np.random.default_rng(0), rate=5.0, duration=2_000.0)
    assert (np.diff(t) >= 0).all() and t[0] >= 0 and t[-1] < 2_000
    assert len(t) == pytest.approx(10_000, rel=0.03)
    assert np.diff(t).mean() == pytest.approx(0.2, rel=0.03)


def rows() -> pd.DataFrame:
    # Two samples with 3 completions each, stored out of order, plus a sample with a single completion.
    return pd.DataFrame({
        "sample_id": ["s1", "s2", "s1", "s2", "s1", "s2", "s3"],
        "input_size": [1, 2, 1, 2, 1, 2, 3],
        "output_tokens": [10, 20, 11, 21, 12, 22, 30],
        "model": ["m1", "m2", "m1", "m2", "m1", "m2", "m3"],
        "dataset": ["A", "B", "A", "B", "A", "B", "C"],
    })


def test_a_trace_draws_whole_samples_uniformly_then_one_of_their_completions():
    r = rows()
    t = sim.make_trace(r, sim.Config(rate=30.0, duration=600.0, seed=3))
    picked = r.iloc[t.row]
    assert (picked.input_size.to_numpy() == t.input_tokens).all()
    assert (picked.output_tokens.to_numpy() == t.output_tokens).all()
    assert (picked.dataset.to_numpy() == t.dataset).all()
    share = picked.sample_id.value_counts(normalize=True)
    assert share.to_dict() == pytest.approx({"s1": 1 / 3, "s2": 1 / 3, "s3": 1 / 3}, abs=0.02)  # per sample, not row
    s1 = picked[picked.sample_id == "s1"].output_tokens.value_counts(normalize=True)
    assert s1.to_dict() == pytest.approx({10: 1 / 3, 11: 1 / 3, 12: 1 / 3}, abs=0.03)


def test_a_trace_is_fixed_by_its_seed():
    a, b, c = (sim.make_trace(rows(), sim.Config(duration=300.0, seed=s)) for s in (1, 1, 2))
    assert np.array_equal(a.arrival, b.arrival) and np.array_equal(a.row, b.row)
    assert not np.array_equal(a.row[:50], c.row[:50])


@pytest.mark.skipif(not (OUT_DIR / "completions_test.parquet").exists(), reason="run `make data`")
def test_the_simulator_reads_test_from_the_original_config_degenerate_included():
    r = sim.load_rows("test")
    assert len(r) == 140_480 and r.is_degenerate.sum() == 204 and r.dataset.notna().all()
    cfg = sim.Config(duration=600.0)
    t = sim.make_trace(r, cfg)
    res = sim.simulate(t, sim.max_tokens_policy()(t), cfg)
    s = sim.summarize(t, res, cfg)
    assert s["overshoot_windows"] == 0 and 0 < s["admitted"] <= s["requests"]


def test_a_wrongful_rejection_is_a_429_whose_real_cost_would_have_fit():
    # A reserves 10 + 50 and holds it. B (cost 30) would fit in the 40 left but reserves 10 + 50: wrongful.
    # C (cost 50) would not fit in 40 either way: a rightful 429.
    t = trace([0, 1, 2], [10, 10, 10], [5, 20, 40], models=["a", "a", "b"])
    res = sim.simulate(t, [50, 50, 50], CFG)
    assert res.admitted.tolist() == [True, False, False]
    assert res.fits.tolist() == [True, True, False]
    s = sim.summarize(t, res, CFG)
    assert (s["rejected"], s["wrongful_rejections"]) == (2, 1)
    assert s["wrongful_rejection_rate"] == pytest.approx(1 / 3) and s["mean_admitted_cost"] == 15
    by = sim.group_table(t.model, t, res)
    assert (by["a"]["wrongful_rejections"], by["b"]["wrongful_rejections"]) == (1, 0)


def test_an_exact_reservation_is_never_wrongfully_rejected():
    rng = np.random.default_rng(1)
    n = 400
    t = trace(np.sort(rng.uniform(0, 120, n)), rng.integers(1, 30, n), rng.integers(0, 60, n))
    res = sim.simulate(t, t.output_tokens, CFG)  # each request reserves exactly what it will use
    s = sim.summarize(t, res, CFG)
    assert s["rejected"] > 0 and s["wrongful_rejections"] == 0 and s["overshoot_windows"] == 0


def test_a_bounds_policy_looks_up_each_requests_row_and_rounds_up():
    t = sim.Trace(arrival=np.array([0.0, 1.0, 2.0]), row=np.array([2, 0, 2]), input_tokens=np.array([1, 1, 1]),
                  output_tokens=np.array([1, 1, 1]), model=np.array(["m"] * 3), dataset=np.array(["d"] * 3))
    assert sim.bounds_policy(np.array([10.0, 20.0, 30.2]))(t).tolist() == [31, 10, 31]
    assert np.array_equal(sim.bounds_policy(np.full(3, 4096.0))(t), sim.max_tokens_policy()(t))
    with pytest.raises(ValueError):
        sim.bounds_policy(np.array([1.0, -1.0, 2.0]))


def test_a_capped_request_is_cut_at_its_reservation_and_never_overshoots():
    t = trace([0, 1], [10, 10], [50, 5], datasets=["d", "d"])
    t = sim.Trace(**{**t.__dict__, "degenerate": np.array([False, False])})
    res = sim.simulate(t, [20, 20], CFG, cap=True)  # sent with max_tokens = 20
    assert res.limit.tolist() == [20, 20]
    assert res.reserved.tolist() == [30, 30] and res.actual.tolist() == [30, 15]
    s = sim.summarize(t, res, CFG)
    assert (s["truncated"], s["cut_by_cap"], s["truncated_clean"], s["overshoot_windows"]) == (1, 1, 1, 0)
    assert s["truncated_rate"] == 0.5 and s["under_reserved_tokens"] == 0


def test_a_cap_never_lifts_max_tokens_and_degenerate_cuts_are_not_counted_as_real_answers():
    t = trace([0, 61], [10, 10], [500, 60])  # one per window, so both are admitted
    t = sim.Trace(**{**t.__dict__, "degenerate": np.array([True, False])})
    res = sim.simulate(t, [1_000, 40], CFG, cap=True)
    assert res.limit.tolist() == [80, 40]  # max_tokens = 80 still wins
    s = sim.summarize(t, res, CFG)
    assert (s["truncated"], s["cut_by_cap"]) == (2, 1)  # the runaway would have been cut at 80 anyway
    assert (s["truncated_clean"], s["cut_by_cap_clean"]) == (1, 1)
    with pytest.raises(ValueError):
        sim.simulate(t, [0, 40], CFG, cap=True)


def test_a_holdback_keeps_a_share_out_of_admission_but_overshoot_is_measured_on_the_whole_budget():
    t = trace([0, 1], [10, 10], [70, 5])
    no = sim.simulate(t, [40, 40], CFG)
    held = sim.simulate(t, [40, 40], CFG, holdback=0.15)  # admits up to 85 of 100
    assert no.admitted.tolist() == [True, True] and held.admitted.tolist() == [True, False]
    s = sim.summarize(t, held, CFG)
    assert held.spend[0] == 80 and s["overshoot_windows"] == 0  # the 30-token miss fits in the buffer
    assert sim.summarize(t, held, CFG)["wrongful_rejections"] == 1  # its real cost (15) would have fit


def test_run_applies_the_strategy_a_policy_carries():
    rows = pd.DataFrame({"sample_id": ["a", "b"], "input_size": [10, 10], "output_tokens": [300, 20],
                         "model": ["m", "m"], "dataset": ["d", "d"], "is_degenerate": [False, False]})
    cfg = sim.Config(budget=2_000, duration=60.0, rate=2.0, max_tokens=4096)
    policy = sim.with_strategy(sim.bounds_policy(np.array([50.0, 50.0])), cap=True)
    r = sim.run(rows, {"capped": policy, "plain": sim.bounds_policy(np.array([50.0, 50.0]))}, cfg, seeds=(0,))
    capped, plain = r["capped"]["seeds"][0], r["plain"]["seeds"][0]
    assert capped["under_reserved_tokens"] == 0 and capped["truncated"] > 0
    assert plain["under_reserved_tokens"] > 0 and plain["truncated"] == 0
    with pytest.raises(ValueError):
        sim.with_strategy(policy, holdback=1.0)
