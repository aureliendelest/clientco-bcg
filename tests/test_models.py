"""Models: registry, deterministic rule baselines, the estimator contract shared by every family, and calibration."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from churn.config import Config, ModelCfg, override_config
from churn.models import (
    _DEFAULT_TRIAL, ALL_SEGMENTS, BGNBD, MIN_SEGMENT_FIT, MODEL_REGISTRY, SEARCH_SPACES, RuleBaseline, apply_calibrator,
    fit_calibrator, get_model, tune,
)
from churn.snapshots import FEATURE_SETS

FEATURES = FEATURE_SETS["basic"]


def _features(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic snapshot: the longer a client has been silent compared to his rhythm, the likelier he is lost."""
    rng = np.random.default_rng(seed)
    gap_p90 = rng.uniform(3, 30, n)
    recency = rng.uniform(0, 120, n)
    ratio = recency / gap_p90
    df = pd.DataFrame({
        "recency_days": recency, "freq_180d": rng.integers(6, 60, n), "freq_90d": rng.integers(0, 30, n),
        "freq_prev_90d": rng.integers(0, 30, n), "rev_180d": rng.uniform(1e3, 1e5, n), "rev_90d": rng.uniform(0, 5e4, n),
        "rev_prev_90d": rng.uniform(0, 5e4, n), "gap_median": gap_p90 / 2, "gap_p90": gap_p90, "recency_ratio": ratio,
        "freq_trend": rng.uniform(-1, 1, n), "rev_trend": rng.uniform(-1, 1, n),
    })[FEATURES]
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-(ratio - 3)))).astype(int))
    return df, y


# ------------------------------------------------------------------------------ baselines


def test_relative_recency_ranks_by_ratio_and_is_a_probability():
    X = pd.DataFrame({"recency_ratio": [0.5, 3.0, 1.0, 10.0]})
    p = RuleBaseline().fit(X, None).predict_proba(X)
    assert p.shape == (4, 2)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert ((p >= 0) & (p <= 1)).all()
    assert list(np.argsort(-p[:, 1])) == [3, 1, 2, 0]  # most silent compared to his rhythm first


def test_relative_recency_missing_ratio_is_no_risk():
    X = pd.DataFrame({"recency_ratio": [np.nan, 2.0]})
    p = RuleBaseline().predict_proba(X)[:, 1]
    assert p[0] == 0.0 and p[1] > 0.0


def test_recency_days_rule_ignores_rhythm():
    X = pd.DataFrame({"recency_days": [10.0, 90.0, 40.0], "recency_ratio": [9.0, 1.0, 1.0]})
    p = RuleBaseline(rule="recency_days").predict_proba(X)[:, 1]
    assert list(np.argsort(-p)) == [1, 2, 0]


def test_business_rules_count_fired_rules_then_break_ties_by_recency():
    trend = np.log([1.0, 1.0, 0.3, 0.3])  # the trend features are log-ratios (0 = stable)
    X = pd.DataFrame({"recency_ratio": [0.2, 1.5, 1.5, 3.0], "freq_trend": trend, "rev_trend": trend})
    s = RuleBaseline(rule="business_rules").score(X)
    assert np.floor(s).tolist() == [0, 1, 3, 3]  # rules fired
    assert s[3] > s[2]  # same count: the more silent client first
    assert s.max() < 4  # the tie-breaker never reaches another rule


def test_business_rules_thresholds_are_parameters():
    X = pd.DataFrame({"recency_ratio": [1.2], "freq_trend": [np.log(0.8)], "rev_trend": [np.log(0.8)]})
    assert np.floor(RuleBaseline(rule="business_rules").score(X))[0] == 1
    assert np.floor(RuleBaseline(rule="business_rules", ratio_threshold=2.0, trend_threshold=0.9).score(X))[0] == 2


def test_baseline_names_the_missing_columns_and_rejects_unknown_rules():
    with pytest.raises(KeyError, match="recency_ratio"):
        RuleBaseline().predict_proba(pd.DataFrame({"recency_days": [1.0]}))
    with pytest.raises(ValueError, match="Unknown rule"):
        RuleBaseline(rule="vibes").fit(pd.DataFrame(), None)
    with pytest.raises(ValueError, match="Unknown rule"):
        RuleBaseline(rule="vibes").predict_proba(pd.DataFrame({"recency_ratio": [1.0]}))


def test_baseline_is_deterministic_and_clonable():
    X, y = _features()
    m = RuleBaseline(rule="business_rules").fit(X, y)
    assert np.array_equal(m.predict_proba(X), clone(m).fit(X, y).predict_proba(X))


# ------------------------------------------------------------------------------- BG/NBD


def _simulate_bgnbd(n: int, r: float, alpha: float, a: float, b: float, T: int, H: int, seed: int):
    """Clients simulated from the BG/NBD generative story over a window of T days, then H more days observed.
    Returns the window features (`freq_180d`, `recency_days`, with a window of T) and whether each bought in the next H."""
    rng = np.random.default_rng(seed)
    lam, p = rng.gamma(r, 1 / alpha, n), rng.beta(a, b, n)
    x, t_x, bought_next = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
    for i in range(n):
        t, alive = 0.0, True
        while True:
            t += rng.exponential(1 / lam[i])
            if t > T + H:
                break
            if t <= T:
                x[i] += 1
                t_x[i] = t
            else:
                bought_next[i] = True
                break
            if rng.uniform() < p[i]:
                alive = False
                break
        assert alive or not bought_next[i]
    return pd.DataFrame({"freq_180d": x, "recency_days": np.round(T - t_x)}), bought_next


def test_bgnbd_recovers_its_parameters_and_predicts_silence():
    X, bought_next = _simulate_bgnbd(6000, r=1.5, alpha=30.0, a=0.6, b=2.5, T=180, H=90, seed=0)
    model = BGNBD(window_days=180, horizon_days=90).fit(X)
    assert model.params_["r"] == pytest.approx(1.5, rel=0.3)
    assert model.params_["alpha"] == pytest.approx(30.0, rel=0.3)
    p = model.predict_proba(X)[:, 1]
    assert abs(p.mean() - (~bought_next).mean()) < 0.02  # calibrated by construction: the mean matches the silence rate
    assert roc_auc_score(~bought_next, p) > 0.7


def test_bgnbd_risk_grows_with_recency_and_shrinks_with_the_horizon():
    X, _ = _simulate_bgnbd(3000, r=1.5, alpha=30.0, a=0.6, b=2.5, T=180, H=0, seed=1)
    model = BGNBD(horizon_days=60).fit(X)
    quiet = pd.DataFrame({"freq_180d": [12.0] * 4, "recency_days": [2.0, 30.0, 90.0, 170.0]})
    p = model.predict_proba(quiet)[:, 1]
    assert (np.diff(p) > 0).all()
    longer = BGNBD(horizon_days=180).fit(X).predict_proba(quiet)[:, 1]
    assert (longer <= p).all() and (longer < p).any()  # silence over 180 days implies silence over 60: rarer


def test_bgnbd_names_missing_columns_and_rejects_missing_values():
    X, _ = _features()
    with pytest.raises(KeyError, match="freq_180d"):
        BGNBD().fit(X.drop(columns=["freq_180d"]))
    X.loc[X.index[0], "recency_days"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        BGNBD().fit(X)


# ---------------------------------------------------------------- the contract of every family


@pytest.mark.parametrize("kind", sorted(MODEL_REGISTRY))
def test_every_model_fits_and_predicts_probabilities(kind):
    X, y = _features()
    model = get_model(ModelCfg(type=kind), seed=0).fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (len(X), 2)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert ((p >= 0) & (p <= 1)).all()
    assert pd.Series(p[:, 1]).corr(y.astype(float)) > 0.2  # the signal is picked up (or, for the rule, ranked)


@pytest.mark.parametrize("kind", ["logreg", "hgb", "xgb"])
def test_trained_models_accept_missing_features(kind):
    X, y = _features()
    X.loc[X.index[:40], ["gap_median", "gap_p90", "recency_ratio"]] = np.nan  # short history: rhythm is NaN
    p = get_model(ModelCfg(type=kind), seed=0).fit(X, y).predict_proba(X)
    assert not np.isnan(p).any()


@pytest.mark.parametrize("kind", ["logreg", "hgb", "xgb"])
def test_same_seed_same_predictions(kind):
    X, y = _features()
    a = get_model(ModelCfg(type=kind), seed=3).fit(X, y).predict_proba(X)
    b = get_model(ModelCfg(type=kind), seed=3).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b)


def test_params_override_defaults_and_unknown_type_is_refused():
    assert get_model(ModelCfg(type="hgb", params={"max_iter": 7}), seed=0).max_iter == 7
    assert get_model(ModelCfg(type="xgb", params={"max_depth": 2}), seed=0).max_depth == 2
    assert get_model(ModelCfg(type="baseline", params={"rule": "recency_days"}), seed=0).rule == "recency_days"
    with pytest.raises(ValueError, match="Unknown model type"):
        get_model(ModelCfg(type="mlp"), seed=0)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------ calibration


def _overconfident(n: int = 6000, seed: int = 0):
    """Raw probabilities that are too high: true rate = 0.4 x raw (fast) or 0.8 x raw (medium)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    seg = pd.Series(np.where(rng.uniform(size=n) < 0.5, "fast", "medium"))
    y = (rng.uniform(size=n) < p * np.where(seg == "fast", 0.4, 0.8)).astype(float)
    return p, y, seg


def test_isotonic_calibration_fixes_overconfident_probabilities():
    p, y, seg = _overconfident()
    cal = fit_calibrator(p, y, seg, ModelCfg(calibration="isotonic", calibration_by_segment=False))
    out = apply_calibrator(cal, p, seg)
    assert abs(out.mean() - y.mean()) < 0.01 < abs(p.mean() - y.mean())
    assert ((out >= 0) & (out <= 1)).all()
    order = np.argsort(p)
    assert (np.diff(out[order]) >= -1e-12).all()  # monotone: the ranking is never reversed


def test_calibration_by_segment_has_one_calibrator_per_segment():
    p, y, seg = _overconfident()
    cal = fit_calibrator(p, y, seg, ModelCfg(calibration="isotonic", calibration_by_segment=True))
    assert set(cal.models) == {ALL_SEGMENTS, "fast", "medium"}
    out = apply_calibrator(cal, p, seg)
    for value, rate in (("fast", 0.4), ("medium", 0.8)):
        m = (seg == value).to_numpy()
        assert abs(out[m].mean() - y[m].mean()) < 0.01
    # the same raw score is worth less in the segment that leaves less
    hi = apply_calibrator(cal, np.array([0.9, 0.9]), pd.Series(["fast", "medium"]))
    assert hi[0] < hi[1]


def test_small_or_unseen_segment_falls_back_on_the_global_calibrator():
    p, y, seg = _overconfident()
    seg.iloc[: MIN_SEGMENT_FIT - 1] = "rare"
    cal = fit_calibrator(p, y, seg, ModelCfg(calibration="isotonic", calibration_by_segment=True))
    assert "rare" not in cal.models
    x = np.array([0.3, 0.7])
    for name in ("rare", "never_seen"):
        assert np.array_equal(apply_calibrator(cal, x, pd.Series([name, name])), cal.models[ALL_SEGMENTS].predict(x))


def test_calibration_none_is_the_identity():
    p, y, seg = _overconfident(500)
    cal = fit_calibrator(p, y, seg, ModelCfg(calibration="none"))
    assert np.array_equal(apply_calibrator(cal, p, seg), p)


def test_calibration_refuses_missing_labels_and_one_class():
    p, y, seg = _overconfident(500)
    cfg = ModelCfg(calibration="isotonic")
    y_missing = y.copy()
    y_missing[0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        fit_calibrator(p, y_missing, seg, cfg)
    with pytest.raises(ValueError, match="both classes"):
        fit_calibrator(p, np.zeros_like(y), seg, cfg)
    with pytest.raises(ValueError, match="same length"):
        fit_calibrator(p, y[:-1], seg, cfg)


def test_calibrated_rule_baseline_is_a_usable_probability():
    X, y = _features(3000)
    raw = RuleBaseline().fit(X, y).predict_proba(X)[:, 1]
    seg = pd.Series(np.where(X["freq_180d"].to_numpy() > 20, "fast", "medium"))
    out = apply_calibrator(fit_calibrator(raw, y.to_numpy(), seg, ModelCfg()), raw, seg)
    assert abs(out.mean() - y.mean()) < 0.01
    assert pd.Series(out).corr(pd.Series(raw), method="spearman") > 0.9


# ---- tuning


def _tuning_setup(kind: str, trials: int = 4, **params):
    """A configuration that tunes `kind` and a small training table (two rows per client, all in the perimeter)."""
    cfg = override_config(Config(), {"model.type": kind, "model.tuning_trials": trials, "model.cv_folds": 3, "seed": 0})
    if params:
        cfg = override_config(cfg, {f"model.params.{k}": v for k, v in params.items()})
    X, y = _features(600)
    train = X.assign(client_id=np.arange(len(X)) // 2, snapshot_date=pd.Timestamp("2025-03-22"), rhythm="fast", label=y.to_numpy())
    return cfg, train


@pytest.mark.parametrize("kind", ["hgb", "xgb"])
def test_tune_searches_the_space_and_returns_usable_parameters(kind):
    cfg, train = _tuning_setup(kind)
    best = tune(cfg, train)
    assert set(best) == set(SEARCH_SPACES[kind])
    for name, (_, low, high, _log) in SEARCH_SPACES[kind].items():
        assert low <= best[name] <= high
    model = get_model(ModelCfg(type=kind, params=best), 0).fit(train[FEATURES], train["label"].astype(int))
    p = model.predict_proba(train[FEATURES])[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("kind", ["hgb", "xgb"])
def test_tune_is_reproducible_and_starts_from_the_defaults(kind):
    cfg, train = _tuning_setup(kind)
    assert tune(cfg, train) == tune(cfg, train)
    cfg1, train1 = _tuning_setup(kind, trials=1)
    assert tune(cfg1, train1) == _DEFAULT_TRIAL[kind]  # the first trial is the untuned configuration


def test_tune_leaves_parameters_fixed_in_the_config_alone():
    cfg, train = _tuning_setup("hgb", max_depth=5)
    assert "max_depth" not in tune(cfg, train)


def test_tune_refuses_families_without_a_search_space_and_unlabelled_rows():
    cfg, train = _tuning_setup("logreg")
    with pytest.raises(ValueError, match="Tuning is implemented"):
        tune(cfg, train)
    cfg, train = _tuning_setup("hgb")
    with pytest.raises(ValueError, match="no label"):
        tune(cfg, train.assign(label=np.nan))
