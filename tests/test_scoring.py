"""Scoring: value, list size, reasons (SHAP / coefficients), and the batch action list written by `score_batch`."""

import pickle
import shutil

import duckdb
import numpy as np
import pandas as pd
import pytest
import yaml

from churn import scoring
from churn.config import Config, ScoringCfg, config_to_dict, override_config, run_id
from churn.models import get_model
from churn.pipeline import run
from churn.scoring import LIST_COLUMNS, apply_min_proba, list_size, load_reasons, score_batch, top_reasons
from churn.snapshots import FEATURE_SETS

BASIC = FEATURE_SETS["basic"]
K = 40  # list size of the test runs (the synthetic dataset has a few hundred regular clients per date)
SCORE_DATE = pd.Timestamp("2026-03-22").date()  # the synthetic dataset is not the real one: no final-test guard


# ----------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def runs(pipeline_cfg, tmp_path_factory):
    """One run per model family on the synthetic dataset: model name -> (run folder, runs dir)."""
    root = tmp_path_factory.mktemp("score_runs")
    out = {}
    for kind in ("logreg", "hgb", "xgb", "baseline", "bgnbd"):
        cfg = override_config(pipeline_cfg, {"model.type": kind, "name": f"score_{kind}", "scoring.list_size_k": K})
        out[kind] = run(cfg, runs_dir=root)
    return out, root


def _toy_frame(n=600, seed=0):
    """Basic features of fake clients; the risk grows with the silence and with the decline of the frequency."""
    rng = np.random.default_rng(seed)
    gap = rng.uniform(3, 15, n)
    recency = rng.uniform(0, 120, n)
    f90 = rng.integers(2, 25, n).astype(float)
    fprev = rng.integers(2, 25, n).astype(float)
    r90 = f90 * rng.uniform(50, 500, n)
    rprev = fprev * rng.uniform(50, 500, n)
    X = pd.DataFrame({
        "recency_days": recency, "freq_180d": f90 + fprev, "freq_90d": f90, "freq_prev_90d": fprev,
        "rev_180d": r90 + rprev, "rev_90d": r90, "rev_prev_90d": rprev, "gap_median": gap, "gap_p90": gap * 2,
        "recency_ratio": recency / (gap * 2), "freq_trend": np.log1p(f90) - np.log1p(fprev),
        "rev_trend": np.log1p(r90) - np.log1p(rprev),
    })[BASIC]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(3 * (X["recency_ratio"] - 1) - 2 * X["freq_trend"])))).astype(int)
    return X, y


def _fit(kind, X, y):
    cfg = Config().model
    cfg = type(cfg)(type=kind, cv_folds=3)
    return get_model(cfg, seed=0).fit(X, y)


# --------------------------------------------------------------------------- value, size


def test_compute_value_annualizes_180_days():
    df = pd.DataFrame({"rev_180d": [100.0, 0.0]})
    assert scoring.compute_value(df, Config()).tolist() == [200.0, 0.0]


def test_unsupported_value_definition_is_an_explicit_error_not_not_implemented():
    cfg = Config(scoring=ScoringCfg(value_definition="ca_12m"))
    with pytest.raises(ValueError, match="ca_12m"):
        scoring.compute_value(pd.DataFrame({"rev_180d": [1.0]}), cfg)
    cfg = Config(scoring=ScoringCfg(value_definition="nope"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown"):
        scoring.compute_value(pd.DataFrame({"rev_180d": [1.0]}), cfg)


def test_the_module_has_no_stub_left():
    assert not hasattr(scoring, "mixed_list")
    import inspect

    assert "NotImplementedError" not in inspect.getsource(scoring)


def test_list_size_is_k_of_the_configuration():
    assert list_size(Config()) == 700
    assert list_size(Config(scoring=ScoringCfg(list_size_k=50))) == 50
    for bad in (0, -3, 12.5):
        with pytest.raises(ValueError, match="list_size_k"):
            list_size(Config(scoring=ScoringCfg(list_size_k=bad)))


def test_apply_min_proba():
    df = pd.DataFrame({"client_id": [1, 2, 3], "p_cal": [0.05, 0.2, 0.5]})
    assert len(apply_min_proba(df, Config())) == 3  # 0.0 keeps everybody
    kept = apply_min_proba(df, Config(scoring=ScoringCfg(min_proba=0.2)))
    assert kept["client_id"].tolist() == [2, 3]  # the bound itself is kept
    assert apply_min_proba(df, Config(scoring=ScoringCfg(min_proba=0.9))).empty


# ------------------------------------------------------------------------------ reasons


def test_templates_cover_the_twelve_basic_variables_and_never_say_returns():
    templates = load_reasons()
    assert set(BASIC) <= set(templates)
    assert all(isinstance(t, str) and t for t in templates.values())
    assert not any("retour" in t.lower() for t in templates.values())
    assert templates["recency_days"] == "Aucun achat depuis {value} jours"  # the team's texts are kept


def test_every_template_renders_for_a_suitable_client():
    row = pd.Series({
        "recency_days": 40.0, "recency_ratio": 2.34, "freq_trend": np.log(0.6), "rev_trend": np.log(0.5),
        "freq_90d": 3.0, "freq_180d": 12.0, "freq_prev_90d": 9.0, "rev_90d": 12345.6, "rev_180d": 30000.0,
        "rev_prev_90d": 17654.0, "gap_median": 6.0, "gap_p90": 14.6,
    })
    templates = load_reasons()
    sentences = {name: scoring._render(name, row, templates) for name in BASIC}
    assert all(sentences.values()), sentences
    assert sentences["recency_days"] == "Aucun achat depuis 40 jours"
    assert sentences["recency_ratio"] == "Silence de 2,3 fois son rythme habituel"
    assert sentences["freq_trend"] == "Fréquence d'achat en baisse de 40 % sur 90 jours"
    assert sentences["rev_trend"] == "Chiffre d'affaires en baisse de 50 % sur 90 jours"
    assert sentences["rev_90d"] == "Chiffre d'affaires de 12 346 € sur 90 jours"
    assert sentences["gap_p90"] == "Écart long entre deux achats : 15 jours"
    assert sentences["freq_prev_90d"] == "9 jours d'achat entre 180 et 90 jours avant, 3 depuis"


def test_singular_day():
    templates = load_reasons()
    row = pd.Series({"gap_median": 1.0, "recency_days": 1.0, "freq_180d": 11.0})
    assert scoring._render("gap_median", row, templates) == "Écart habituel entre deux achats : 1 jour"
    assert scoring._render("recency_days", row, templates) == "Aucun achat depuis 1 jour"
    assert scoring._render("freq_180d", row, templates) == "11 jours d'achat sur 180 jours"


def test_a_sentence_is_used_only_when_it_is_true():
    templates = load_reasons()
    growth = pd.Series({"freq_trend": 0.4, "rev_trend": 0.0, "recency_ratio": 0.5, "recency_days": np.nan})
    assert scoring._render("freq_trend", growth, templates) is None  # "baisse" of a growing frequency
    assert scoring._render("rev_trend", growth, templates) is None
    assert scoring._render("recency_ratio", growth, templates) is None  # not past his usual longest pause
    assert scoring._render("recency_days", growth, templates) is None  # missing value
    assert scoring._render("not_a_feature", growth, templates) is None


@pytest.mark.parametrize("kind", ["logreg", "hgb", "xgb"])
def test_top_reasons_keep_only_variables_that_raise_the_risk(kind):
    X, y = _toy_frame()
    model = _fit(kind, X, y)
    reasons = top_reasons(model, X, n=3)
    assert list(reasons.columns) == ["reason_1", "reason_2", "reason_3"] and reasons.index.equals(X.index)
    assert (reasons != "").any(axis=None)
    # a "baisse" is never shown for a client whose trend is not a drop
    for col in ("freq_trend", "rev_trend"):
        stable = X[col] >= 0
        label = "Fréquence d'achat en baisse" if col == "freq_trend" else "Chiffre d'affaires en baisse"
        assert not reasons[stable].apply(lambda s: s.str.contains(label)).any(axis=None)
    # empty reasons come last
    filled = (reasons != "").to_numpy()
    assert (np.diff(filled.astype(int), axis=1) <= 0).all()
    # a client far past his rhythm and in decline is explained by that
    risky = X.assign(recency_days=200.0, recency_ratio=10.0, freq_trend=np.log(0.2)).iloc[:1]
    first = top_reasons(model, risky, n=3).iloc[0]
    assert any(("Silence" in s) or ("Aucun achat" in s) or ("baisse" in s) for s in first), first.tolist()
    # a calm, recent, growing client has no "risk" sentence about silence or decline
    calm = X.assign(recency_days=0.0, recency_ratio=0.0, freq_trend=1.0, rev_trend=1.0).iloc[:1]
    sentences = " ".join(top_reasons(model, calm, n=3).iloc[0])
    assert "baisse" not in sentences and "Silence" not in sentences


def test_logreg_reasons_are_coefficient_times_standardized_value():
    X, y = _toy_frame()
    model = _fit("logreg", X, y)
    contributions = scoring._contributions(model, X)
    assert list(contributions.columns) == BASIC  # the missing-value indicators are not variables
    z = model[:-1].transform(X)[:, : len(BASIC)]
    np.testing.assert_allclose(contributions.to_numpy(), z * model[-1].coef_[0][: len(BASIC)])


def test_shap_contributions_add_up_to_the_raw_log_odds():
    X, y = _toy_frame()
    model = _fit("hgb", X, y)
    contributions = scoring._contributions(model, X.iloc[:20])
    assert contributions.shape == (20, len(BASIC))
    assert np.isfinite(contributions.to_numpy()).all()


@pytest.mark.parametrize("kind", ["baseline", "bgnbd"])
def test_no_reason_for_models_without_per_variable_effect(kind):
    X, y = _toy_frame()
    X = X.assign(freq_180d=X["freq_180d"].round(), recency_days=X["recency_days"].round())
    reasons = top_reasons(_fit(kind, X, y), X.iloc[:10], n=3)
    assert reasons.shape == (10, 3) and (reasons == "").all(axis=None)


def test_unsupported_model_is_an_explicit_error():
    X, _ = _toy_frame(n=20)
    with pytest.raises(ValueError, match="not supported"):
        top_reasons(object(), X)


# -------------------------------------------------------------------------- batch list


@pytest.mark.parametrize("kind", ["logreg", "hgb", "xgb", "baseline", "bgnbd"])
def test_score_batch_writes_the_list(runs, kind):
    folders, root = runs
    path = score_batch(folders[kind].name, SCORE_DATE, runs_dir=root)
    assert path == folders[kind] / f"action_list_{SCORE_DATE:%Y-%m-%d}.csv"
    out = pd.read_csv(path, keep_default_na=False)
    assert list(out.columns) == LIST_COLUMNS == [
        "rank", "client_id", "p_cal", "V", "ev", "segment", "reason_1", "reason_2", "reason_3"]
    assert len(out) == K
    assert out["rank"].tolist() == list(range(1, K + 1)) and out["client_id"].is_unique
    assert out["ev"].is_monotonic_decreasing
    np.testing.assert_allclose(out["ev"], out["p_cal"] * out["V"])
    assert set(out["segment"]) <= {"fast", "medium"}
    assert out[["p_cal", "V", "ev"]].notna().all(axis=None) and out["p_cal"].between(0, 1).all()
    reasons = out[["reason_1", "reason_2", "reason_3"]]
    if kind in ("baseline", "bgnbd"):
        assert (reasons == "").all(axis=None)
    else:
        assert (reasons["reason_1"] != "").any()
        assert not reasons.apply(lambda s: s.str.contains("retour", case=False)).any(axis=None)


def test_score_batch_reproduces_the_predictions_of_the_run_on_the_same_date(runs):
    """Scoring the date a run was tested on must give the run's own predictions: same photo, same model, same list."""
    folders, root = runs
    folder = folders["hgb"]
    out = pd.read_csv(score_batch(folder.name, SCORE_DATE, runs_dir=root))
    pred = pd.read_parquet(folder / "predictions.parquet")
    assert pd.Timestamp(pred["snapshot_date"].iloc[0]).date() == SCORE_DATE
    top = pred.sort_values(["ev", "client_id"], ascending=[False, True]).head(K)
    assert out["client_id"].tolist() == top["client_id"].tolist()
    merged = out.merge(pred, on="client_id", suffixes=("", "_run"))
    for col in ("p_cal", "V", "ev"):
        np.testing.assert_allclose(merged[col], merged[f"{col}_run"], rtol=1e-9)
    assert (merged["segment"] == merged["segment_run"]).all()


def test_score_batch_shorter_than_k_and_min_proba(runs, tmp_path):
    folders, root = runs
    base = folders["logreg"]
    for name, options in (
        ("short", {"scoring.list_size_k": 100_000}), ("floor", {"scoring.min_proba": 1.0}), ("empty", {"scoring.min_proba": 1.1}),
    ):
        folder = tmp_path / name
        shutil.copytree(base, folder)
        cfg = override_config(yaml_to_cfg(base), options)
        (folder / "config.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))
        out = pd.read_csv(score_batch(name, SCORE_DATE, runs_dir=tmp_path))
        full = pd.read_csv(score_batch(base.name, SCORE_DATE, runs_dir=root, output_dir=tmp_path / "ref"))
        if name == "short":
            assert K < len(out) < 100_000 and out["rank"].tolist() == list(range(1, len(out) + 1))
            assert out["client_id"].head(K).tolist() == full["client_id"].tolist()
        elif name == "floor":
            assert len(out) > 0 and (out["p_cal"] >= 1.0).all()
        else:
            assert out.empty and list(out.columns) == LIST_COLUMNS


def yaml_to_cfg(folder):
    from churn.config import load_config

    return load_config(folder / "config.yaml")


# -------------------------------------------------------------------------- no leakage


def test_score_batch_never_builds_labels(runs, monkeypatch):
    """The photo of the date is built from features only: the label builders must not even be called."""
    folders, root = runs

    def boom(*args, **kwargs):
        raise AssertionError("a label or a full snapshot was built")

    import churn.snapshots as snapshots

    monkeypatch.setattr(snapshots, "build_snapshot", boom)
    monkeypatch.setattr(snapshots, "LABELS", {k: boom for k in snapshots.LABELS})
    monkeypatch.setattr(snapshots, "_add_truth", boom)
    out = pd.read_csv(score_batch(folders["xgb"].name, SCORE_DATE, runs_dir=root))
    assert len(out) == K


def test_score_batch_ignores_everything_after_the_date(runs, tmp_path):
    """The list is identical when every transaction from the snapshot date on is removed from the data."""
    folders, root = runs
    base = folders["hgb"]
    cfg = yaml_to_cfg(base)
    sales = pd.read_parquet(cfg.dataset.shifted_parquet)
    cut = sales[pd.to_datetime(sales["date_order"]) < pd.Timestamp(SCORE_DATE)]
    assert 0 < len(cut) < len(sales)
    truncated = tmp_path / "truncated.parquet"
    duckdb.connect().register("df", cut).execute(f"COPY df TO '{truncated}' (FORMAT parquet)")

    folder = tmp_path / "runs" / "truncated"
    shutil.copytree(base, folder)
    (folder / "config.yaml").write_text(
        yaml.safe_dump(config_to_dict(override_config(cfg, {"dataset.shifted_parquet": str(truncated)})), sort_keys=False))
    a = score_batch(base.name, SCORE_DATE, runs_dir=root, output_dir=tmp_path / "a")
    b = score_batch("truncated", SCORE_DATE, runs_dir=tmp_path / "runs", output_dir=tmp_path / "b")
    assert a.read_bytes() == b.read_bytes()


def test_score_batch_refuses_the_final_test_date_on_the_real_dataset(tmp_path, monkeypatch):
    from churn.pipeline import FINAL_TEST_ENV

    monkeypatch.delenv(FINAL_TEST_ENV, raising=False)
    folder = tmp_path / "real"
    folder.mkdir()
    (folder / "config.yaml").write_text(yaml.safe_dump(config_to_dict(Config()), sort_keys=False))  # real dataset path
    with (folder / "model.pkl").open("wb") as f:
        pickle.dump({"model": None, "calibrator": None, "features": BASIC, "config": config_to_dict(Config())}, f)
    with pytest.raises(PermissionError, match=FINAL_TEST_ENV):
        score_batch("real", pd.Timestamp("2026-03-22").date(), runs_dir=tmp_path)


def test_score_batch_on_a_missing_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        score_batch("nope", SCORE_DATE, runs_dir=tmp_path)
    assert run_id  # the run id helper is the one used to name the folders
