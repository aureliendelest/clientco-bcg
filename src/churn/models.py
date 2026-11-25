"""Models: registry, rule baseline, probability calibration and hyperparameter tuning."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from churn.config import Config, ModelCfg
from churn.snapshots import FEATURE_SETS, client_group_folds, in_perimeter

# ---------------------------------------------------------------------------- registry


def build_baseline(params: dict[str, Any], seed: int) -> Any:
    """Deterministic rule baseline, the score to beat. `params["rule"]` selects it (see `RuleBaseline`);
    the default is relative recency (recency / the client's p90 gap). `seed` is unused: no randomness."""
    return RuleBaseline(**params)


def build_logreg(params: dict[str, Any], seed: int) -> Any:
    """Logistic regression on imputed, scaled features (readable coefficients).

    Median imputation with missing-value indicators (rhythm features are NaN below `min_purchases`), then
    standardisation, so the coefficients are comparable. No class weighting: probabilities are calibrated later."""
    options = {"C": 1.0, "max_iter": 1000, "random_state": seed} | params
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(**options))


def build_hgb(params: dict[str, Any], seed: int) -> Any:
    """scikit-learn HistGradientBoostingClassifier. Defaults of the notebooks that chose the label (`07`, `08`).

    Handles NaN natively. Reproducible only with one OpenMP thread (`OMP_NUM_THREADS=1`) and rows sorted by client
    (see docs/03_evidence.md)."""
    options = {"max_iter": 200, "learning_rate": 0.08, "early_stopping": False, "random_state": seed} | params
    return HistGradientBoostingClassifier(**options)


def build_xgb(params: dict[str, Any], seed: int) -> Any:
    """XGBoost classifier (tree method `hist`). One thread by default: the multithreaded run is not reproducible."""
    options = {
        "tree_method": "hist", "n_estimators": 200, "learning_rate": 0.08, "max_depth": 6,
        "n_jobs": 1, "random_state": seed, "eval_metric": "logloss",
    } | params
    return XGBClassifier(**options)


def build_bgnbd(params: dict[str, Any], seed: int) -> Any:
    """BG/NBD probabilistic model (see `BGNBD`): the statistical counterpart of the learned models. `seed` is unused:
    the maximum-likelihood fit is deterministic."""
    return BGNBD(**params)


MODEL_REGISTRY: dict[str, Callable[[dict[str, Any], int], Any]] = {
    "baseline": build_baseline,
    "bgnbd": build_bgnbd,
    "logreg": build_logreg,
    "hgb": build_hgb,
    "xgb": build_xgb,
}
# Optional, time-boxed (add to the registry only if tried): small MLP, 'tested, not retained' if no gain.


def get_model(cfg: ModelCfg, seed: int) -> Any:
    """Build the estimator selected by the configuration (`cfg.type`, `cfg.params`).

    Every estimator has `fit(X, y)` and `predict_proba(X)` (an (n, 2) array, column 1 = P(lost))."""
    try:
        build = MODEL_REGISTRY[cfg.type]
    except KeyError:
        raise ValueError(f"Unknown model type {cfg.type!r}; known: {sorted(MODEL_REGISTRY)}") from None
    return build(dict(cfg.params), seed)


# ----------------------------------------------------------------------- rule baselines

_RULES = ("relative_recency", "recency_days", "business_rules")


class RuleBaseline(ClassifierMixin, BaseEstimator):
    """Deterministic risk score from a fixed rule, with no learning (`fit` only records the classes).

    It follows the scikit-learn estimator API so that it goes through the same pipeline as the models. The score
    is monotone with risk and is mapped to [0, 1) by s / (1 + s); calibration then turns it into a probability.

    Rules (`rule`):
      - `relative_recency` (default): s = recency_days / client's p90 gap (the feature `recency_ratio`). A client
        silent for longer than his usual longest pause is at risk.
      - `recency_days`: s = days since the last purchase day. The naive rule, with no notion of the client's rhythm.
      - `business_rules`: s = number of fired rules + the relative recency squashed to [0, 1) as a tie-breaker.
        Rules: (1) `recency_ratio >= ratio_threshold`; (2) purchase frequency of the last 90 days at most
        `trend_threshold` times that of the 90 days before; (3) same on revenue. `trend_threshold` is a ratio
        (0.5 = halved); the features `freq_trend` and `rev_trend` are log-ratios, compared with its logarithm.
    A missing input (a client with too short a history has no `recency_ratio`) counts as no risk: such clients
    are outside the perimeter of the model.
    """

    def __init__(self, rule: str = "relative_recency", ratio_threshold: float = 1.0, trend_threshold: float = 0.5):
        self.rule = rule
        self.ratio_threshold = ratio_threshold
        self.trend_threshold = trend_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> RuleBaseline:
        if self.rule not in _RULES:
            raise ValueError(f"Unknown rule {self.rule!r}; known: {list(_RULES)}")
        self.classes_ = np.array([0, 1])
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Raw risk score s >= 0 (higher = more at risk)."""
        if self.rule not in _RULES:
            raise ValueError(f"Unknown rule {self.rule!r}; known: {list(_RULES)}")
        needed = {
            "relative_recency": ["recency_ratio"], "recency_days": ["recency_days"],
            "business_rules": ["recency_ratio", "freq_trend", "rev_trend"],
        }[self.rule]
        missing = [c for c in needed if c not in X.columns]
        if missing:
            raise KeyError(f"rule {self.rule!r} needs the columns {missing}")
        if self.rule == "recency_days":
            return X["recency_days"].fillna(0.0).to_numpy(dtype=float)
        ratio = X["recency_ratio"].to_numpy(dtype=float)
        relative = np.nan_to_num(ratio, nan=0.0)
        if self.rule == "relative_recency":
            return relative
        log_threshold = np.log(self.trend_threshold)
        fired = (
            (ratio >= self.ratio_threshold).astype(float)  # NaN compares False: rule not fired
            + (X["freq_trend"].to_numpy(dtype=float) <= log_threshold)
            + (X["rev_trend"].to_numpy(dtype=float) <= log_threshold)
        )
        return fired + relative / (1.0 + relative)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return an (n, 2) array, like scikit-learn classifiers. Column 1 is the monotone score s / (1 + s)."""
        s = self.score(X)
        p = s / (1.0 + s)
        return np.column_stack([1.0 - p, p])


# ------------------------------------------------------------------------------ BG/NBD

_LOG_PARAM_BOUNDS = (-9.0, 9.0)  # log of r, alpha, a, b: keeps the fit away from the degenerate corners (a, b -> infinity)


class BGNBD(ClassifierMixin, BaseEstimator):
    """Beta-Geometric / Negative-Binomial model (Fader, Hardie and Lee, 2005): the classical statistical model of
    customer attrition when departure is never observed, only silence.

    A client buys at a Poisson rate lambda ~ Gamma(r, alpha) while "alive". After each purchase he leaves for good
    with probability p ~ Beta(a, b). Four parameters (r, alpha, a, b) are fitted by maximum likelihood on the clients'
    purchase counts, with no use of the label: the model is unsupervised, `y` is ignored.

    Mapping to the snapshot features (no purchase history is needed beyond the basic feature set):
      - `x` = `freq_180d`, purchase days in the window; `T` = `window_days`, its length;
      - `t_x` = `window_days - recency_days`, the time of the last purchase from the window start.
    The window start plays the role of the "first purchase" of the original model: the client is assumed alive then
    (true for regular clients, ≥ 6 purchases in the window, but an approximation). A client with several snapshots
    contributes one row per snapshot to the fit (overlapping windows: the parameters are not independent estimates).

    Departure is a silence, so the score is the probability of **no purchase in the next `horizon_days`**:
    P0 = L(x, t_x, T + H) / L(x, t_x, T), a closed form (the likelihood of the same history extended by H days with no
    purchase, over that of the history). `horizon_days` is the label horizon (180 = label A); under label B, whose
    horizon varies by client, the score is a ranking that the calibration then maps to the label.

    Limits: one population-wide mixing distribution (fast and medium rhythms share the same Gamma prior), a constant
    purchase rate (no seasonality, no trend: the `freq_trend` signal is unused), purchase counts only (revenue is
    handled by V).
    """

    def __init__(self, window_days: int = 180, horizon_days: int = 180):
        self.window_days = window_days
        self.horizon_days = horizon_days

    @staticmethod
    def _log_likelihood(log_params: np.ndarray, x: np.ndarray, t_x: np.ndarray, T: float) -> np.ndarray:
        """Log-likelihood of each (x, t_x) history observed for T days (Fader et al., 2005, eq. 6)."""
        r, alpha, a, b = np.exp(log_params)
        head = gammaln(r + x) - gammaln(r) + r * np.log(alpha) - betaln(a, b)
        alive = head - (r + x) * np.log(alpha + T) + betaln(a, b + x)  # still alive at T
        # Left after the last purchase at t_x: only possible after at least one purchase (x > 0).
        left = head - (r + x) * np.log(alpha + t_x) + betaln(a + 1.0, b + np.maximum(x, 1.0) - 1.0)
        return np.where(x > 0, np.logaddexp(alive, left), alive)

    def _inputs(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = [c for c in ("freq_180d", "recency_days") if c not in X.columns]
        if missing:
            raise KeyError(f"BG/NBD needs the columns {missing}")
        if self.window_days <= 0 or self.horizon_days <= 0:
            raise ValueError("window_days and horizon_days must be positive")
        x = X["freq_180d"].to_numpy(dtype=float)
        t_x = self.window_days - X["recency_days"].to_numpy(dtype=float)
        if np.isnan(x).any() or np.isnan(t_x).any():
            raise ValueError("BG/NBD does not accept missing freq_180d or recency_days")
        return np.maximum(x, 0.0), np.clip(t_x, 0.0, float(self.window_days))

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> BGNBD:
        x, t_x = self._inputs(X)
        # Identical histories share a likelihood term: fit on the distinct (x, t_x) pairs, weighted by their count.
        pairs, counts = np.unique(np.column_stack([x, t_x]), axis=0, return_counts=True)
        xs, ts = pairs[:, 0], pairs[:, 1]
        T = float(self.window_days)

        def negative_log_likelihood(log_params: np.ndarray) -> float:
            return -float(np.sum(counts * self._log_likelihood(log_params, xs, ts, T)))

        start = np.log([1.0, 10.0, 1.0, 1.0])
        fit = minimize(negative_log_likelihood, start, method="L-BFGS-B", bounds=[_LOG_PARAM_BOUNDS] * 4)
        if not np.isfinite(fit.fun):
            raise RuntimeError(f"BG/NBD fit failed: {fit.message}")
        self.classes_ = np.array([0, 1])
        self.params_ = dict(zip(("r", "alpha", "a", "b"), np.exp(fit.x).astype(float)))
        self._log_params = fit.x
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return an (n, 2) array. Column 1 = P(no purchase in the next `horizon_days`), the probability of loss."""
        x, t_x = self._inputs(X)
        T = float(self.window_days)
        p = np.exp(self._log_likelihood(self._log_params, x, t_x, T + self.horizon_days)
                   - self._log_likelihood(self._log_params, x, t_x, T))
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


# ------------------------------------------------------------------------- calibration
# Required: the expected value multiplies P by V, so 0.05 must really mean 5 %.


ALL_SEGMENTS = "__all__"  # key of the calibrator fitted on every client; fallback for a segment with no calibrator of its own
MIN_SEGMENT_FIT = 200  # below this many clients (or with a single class) a segment is calibrated by the global calibrator


@dataclass
class Calibrator:
    """Fitted calibrators, one per segment (or a single one).

    `models` maps a segment value (as a string) to a fitted `IsotonicRegression`, plus `ALL_SEGMENTS` for the
    calibrator fitted on every client. It is empty when `method == "none"` (identity)."""

    method: str
    by_segment: bool
    models: dict[str, Any]


def _isotonic(p_raw: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    # Bounds in [0, 1] and clipping outside the fitted range: a raw score never seen at fit time gets the
    # probability of the nearest one. Isotonic output is a step function: clients sharing a step tie on P (the
    # ranking by P x V then falls back on V inside the step).
    return IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(p_raw, y)


def fit_calibrator(p_raw: np.ndarray, y: np.ndarray, segment: pd.Series, cfg: ModelCfg) -> Calibrator:
    """Fit on validation data only, never on the test snapshot.

    `p_raw` are out-of-fold raw probabilities (or scores), `y` the training label (0 / 1, no missing value: a client
    whose label is not observed does not belong to the calibration data). With `cfg.calibration_by_segment`, one
    isotonic regression per segment; a segment with fewer than `MIN_SEGMENT_FIT` clients or a single class uses the
    global one. `cfg.calibration == "none"` returns the identity."""
    by_segment = bool(cfg.calibration_by_segment)
    if cfg.calibration == "none":
        return Calibrator("none", by_segment, {})
    if cfg.calibration != "isotonic":
        raise ValueError(f"Unknown calibration {cfg.calibration!r}; known: ['isotonic', 'none']")
    p_raw, y = np.asarray(p_raw, dtype=float), np.asarray(y, dtype=float)
    if not (len(p_raw) == len(y) == len(segment)):
        raise ValueError(f"p_raw, y and segment must have the same length, got {len(p_raw)}, {len(y)}, {len(segment)}")
    if np.isnan(p_raw).any() or np.isnan(y).any():
        raise ValueError("p_raw and y must not contain missing values: keep only the clients whose label is observed")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("y must be 0 / 1")
    if len(np.unique(y)) < 2:
        raise ValueError("Calibration needs both classes in y")
    models: dict[str, Any] = {ALL_SEGMENTS: _isotonic(p_raw, y)}
    if by_segment:
        seg = segment.astype(str).to_numpy()
        for value in np.unique(seg):
            mask = seg == value
            if mask.sum() >= MIN_SEGMENT_FIT and len(np.unique(y[mask])) == 2:
                models[value] = _isotonic(p_raw[mask], y[mask])
    return Calibrator("isotonic", by_segment, models)


def apply_calibrator(calibrator: Calibrator, p_raw: np.ndarray, segment: pd.Series) -> np.ndarray:
    """Return calibrated probabilities `p_cal`, in the order of `p_raw`."""
    p_raw = np.asarray(p_raw, dtype=float)
    if calibrator.method == "none":
        return p_raw.copy()
    if len(p_raw) != len(segment):
        raise ValueError(f"p_raw and segment must have the same length, got {len(p_raw)} and {len(segment)}")
    if calibrator.by_segment:
        seg = segment.astype(str).to_numpy()
        out = np.empty(len(p_raw))
        for value in np.unique(seg):
            mask = seg == value
            out[mask] = calibrator.models.get(value, calibrator.models[ALL_SEGMENTS]).predict(p_raw[mask])
        return out
    return calibrator.models[ALL_SEGMENTS].predict(p_raw)


# ------------------------------------------------------------------------- importances

IMPORTANCE_SAMPLE = 20_000  # rows used for the permutation importance (the full training set would take minutes)


def feature_importances(model: Any, X: pd.DataFrame, y: pd.Series, seed: int) -> pd.DataFrame:
    """Importance of each feature, long table `feature, importance, kind`. The kind depends on the family:
    `coefficient` (logistic regression, on standardized features: signed, comparable), `gain` (XGBoost),
    `permutation_ap` (HistGradientBoosting: drop in average precision when the feature is shuffled, measured on
    a sample of the rows the model was trained on). The rule baselines and BG/NBD (four fitted parameters, no
    per-feature effect) have none (empty table)."""
    if isinstance(model, (RuleBaseline, BGNBD)):
        values: dict[str, float] = {}
        kind = "none"
    elif isinstance(model, XGBClassifier):
        values, kind = dict(zip(X.columns, model.feature_importances_.astype(float))), "gain"
    elif hasattr(model, "steps") and isinstance(model[-1], LogisticRegression):
        names = model[:-1].get_feature_names_out(list(X.columns))
        values, kind = dict(zip(names, model[-1].coef_[0].astype(float))), "coefficient"
    else:
        sample = X.sample(n=min(IMPORTANCE_SAMPLE, len(X)), random_state=seed)
        res = permutation_importance(
            model, sample, y.loc[sample.index], scoring="average_precision", n_repeats=3, random_state=seed, n_jobs=1
        )
        values, kind = dict(zip(X.columns, res.importances_mean.astype(float))), "permutation_ap"
    return pd.DataFrame({"feature": list(values), "importance": list(values.values()), "kind": kind})


# ------------------------------------------------------------------------------ tuning


# Search spaces (docs/02_modeling_decisions.md, section 10). (kind, low, high, log scale). The names are the
# constructor arguments of the estimator. They bound capacity and strengthen regularization: the defaults were
# never tuned, and a lighter model transfers better from one snapshot date to the next.
SEARCH_SPACES: dict[str, dict[str, tuple[str, float, float, bool]]] = {
    "hgb": {
        "learning_rate": ("float", 0.02, 0.2, True),
        "max_iter": ("int", 100, 600, False),
        "max_leaf_nodes": ("int", 8, 63, False),
        "max_depth": ("int", 3, 10, False),
        "min_samples_leaf": ("int", 20, 1000, True),
        "l2_regularization": ("float", 1e-3, 100.0, True),
    },
    "xgb": {
        "learning_rate": ("float", 0.02, 0.2, True),
        "n_estimators": ("int", 100, 600, False),
        "max_depth": ("int", 2, 8, False),
        "min_child_weight": ("float", 1.0, 100.0, True),
        "reg_lambda": ("float", 0.1, 100.0, True),
        "subsample": ("float", 0.6, 1.0, False),
        "colsample_bytree": ("float", 0.6, 1.0, False),
    },
}
# First trial of each study: the untuned configuration, so that the search can only match or beat the defaults
# (HGB has no depth limit by default: 10 is the nearest value of the space).
_DEFAULT_TRIAL: dict[str, dict[str, float]] = {
    "hgb": {"learning_rate": 0.08, "max_iter": 200, "max_leaf_nodes": 31, "max_depth": 10, "min_samples_leaf": 20,
            "l2_regularization": 1e-3},
    "xgb": {"learning_rate": 0.08, "n_estimators": 200, "max_depth": 6, "min_child_weight": 1.0, "reg_lambda": 1.0,
            "subsample": 1.0, "colsample_bytree": 1.0},
}


def tune(cfg: Config, train: pd.DataFrame) -> dict[str, Any]:
    """Run `cfg.model.tuning_trials` Optuna trials with grouped-by-client K-fold on the training
    snapshots. The test snapshot is never used. Returns the best hyperparameters.

    Only HistGradientBoosting and XGBoost are tuned (`SEARCH_SPACES`). A parameter fixed in `model.params` is not
    searched. The objective is the mean average precision (PR-AUC) of the validation folds, on the training label:
    the quantity the calibration then turns into probabilities, with no dependence on K or on V. The sampler is seeded
    and trials run one after the other, so the same configuration gives the same result. Trials clearly worse than
    the median are pruned after a few folds. `train` is the perimeter table of the training snapshots (rows
    outside the perimeter are dropped), every label must be observed."""
    family = cfg.model.type
    if family not in SEARCH_SPACES:
        raise ValueError(f"Tuning is implemented for {sorted(SEARCH_SPACES)}, not for {family!r}")
    df = train.loc[in_perimeter(train)].sort_values(["client_id", "snapshot_date"], kind="stable").reset_index(drop=True)
    if df.empty or df["label"].isna().any():
        raise ValueError("The training rows are empty or have no label: the label window is not observed")
    fixed = dict(cfg.model.params)
    space = {name: spec for name, spec in SEARCH_SPACES[family].items() if name not in fixed}
    if not space or cfg.model.tuning_trials < 1:
        return {}
    features = FEATURE_SETS[cfg.snapshots.feature_set]
    X, y = df[features], df["label"].astype(int)
    folds = list(client_group_folds(df, cfg.model.cv_folds, seed=cfg.seed))

    def objective(trial: optuna.Trial) -> float:
        sampled: dict[str, Any] = {}
        for name, (kind, low, high, log) in space.items():
            if kind == "int":
                sampled[name] = trial.suggest_int(name, int(low), int(high), log=log)
            else:
                sampled[name] = trial.suggest_float(name, low, high, log=log)
        model_cfg = dataclasses.replace(cfg.model, params=fixed | sampled)
        scores: list[float] = []
        for step, (fit_rows, val_rows) in enumerate(folds):
            if y.iloc[val_rows].sum() == 0:  # no positive: average precision is undefined
                continue
            model = get_model(model_cfg, cfg.seed).fit(X.iloc[fit_rows], y.iloc[fit_rows])
            scores.append(float(average_precision_score(y.iloc[val_rows], model.predict_proba(X.iloc[val_rows])[:, 1])))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )
        study.enqueue_trial({name: value for name, value in _DEFAULT_TRIAL[family].items() if name in space})
        study.optimize(objective, n_trials=cfg.model.tuning_trials, n_jobs=1)
    finally:
        optuna.logging.set_verbosity(previous)
    return dict(study.best_params)
