"""Models: registry, deterministic rule baselines, the estimator contract shared by every family, and calibration."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from churn.config import ModelCfg
from churn.models import ALL_SEGMENTS, MIN_SEGMENT_FIT, MODEL_REGISTRY, RuleBaseline, apply_calibrator, fit_calibrator, get_model
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
        "freq_trend": rng.uniform(0, 2, n), "rev_trend": rng.uniform(0, 2, n),
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
    X = pd.DataFrame({
        "recency_ratio": [0.2, 1.5, 1.5, 3.0], "freq_trend": [1.0, 1.0, 0.3, 0.3], "rev_trend": [1.0, 1.0, 0.3, 0.3],
    })
    s = RuleBaseline(rule="business_rules").score(X)
    assert np.floor(s).tolist() == [0, 1, 3, 3]  # rules fired
    assert s[3] > s[2]  # same count: the more silent client first
    assert s.max() < 4  # the tie-breaker never reaches another rule


def test_business_rules_thresholds_are_parameters():
    X = pd.DataFrame({"recency_ratio": [1.2], "freq_trend": [0.8], "rev_trend": [0.8]})
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
