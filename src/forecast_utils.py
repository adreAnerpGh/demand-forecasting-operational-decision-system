"""
forecast_utils.py

Reusable utility functions for the Daily Occupancy Forecasting &
Demand Dynamics project.

This module centralizes the main logic developed in the notebooks:
- feature engineering
- train/test split with train-only seasonality
- spike labeling
- point forecasting models
- guardrail analysis
- directional modeling
- multi-horizon operational output

The other notebooks are useful for storytelling, interpretation, and results.
This file is useful for reusability and cleaner notebook structure.

Typical usage:

from forecast_utils import (
    add_features,
    compute_spike_threshold,
    apply_spike_labels,
    split_train_test_with_seasonality,
    run_linear_model,
    run_ridge_model,
    build_guardrail,
    evaluate_with_guardrail,
    analyze_spikes,
    add_direction_target,
    run_direction_workflow,
    build_operational_signal_multi,
    build_forecast_summary,
    build_guardrail_summary,
    build_direction_summary,
)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import os
import joblib

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features for multi-horizon occupancy forecasting.

    Project logic
    -------------
    Occupancy is treated as a state that evolves over time. It is not
    modeled as an isolated series, but as the result of:
    - current occupancy persistence
    - booking inflows
    - notice outflows
    - timing-adjusted demand pressure
    - calendar and seasonal effects

    Parameters
    ----------
    daily : pd.DataFrame
        Cleaned daily dataset produced in Notebook 1.

    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe with added features and target columns.
    """
    required_cols = [
    "Date", "Occupied_Units", "Occupancy",
    "Bookings", "Notices",
    "Avg_Booking_Time", "Avg_Notice_Time",
    "Has_Bookings", "Has_Notices",
    "units_available",
    ]
    _validate_columns(daily, required_cols, "add_features")

    daily = daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"])

    # --------------------------------------------------------
    # Capacity input
    # --------------------------------------------------------
    # units_available is now expected to be provided by Notebook 1.
    # It may vary over time, so it must not be reconstructed here.
    if (daily["units_available"].isna()).any():
        raise ValueError("units_available contains missing values.")
    if (daily["units_available"] <= 0).any():
        raise ValueError("units_available must be strictly positive.")

    # --------------------------------------------------------
    # Flow variables
    # --------------------------------------------------------
    # Net_Flow is the simplest operational representation of demand balance:
    # positive  -> occupancy pressure upward
    # negative  -> occupancy pressure downward
    daily["Net_Flow"] = daily["Bookings"] - daily["Notices"]

    # Normalize flows by capacity to make them easier to compare over time.
    daily["Demand_Pressure"] = daily["Net_Flow"] / daily["units_available"]
    daily["Booking_Intensity"] = daily["Bookings"] / daily["units_available"]
    daily["Notice_Pressure"] = daily["Notices"] / daily["units_available"]

    # --------------------------------------------------------
    # Timing-adjusted flow
    # --------------------------------------------------------
    # Raw booking and notice counts do not fully capture impact timing.
    # A booking made far in advance has a different immediate implication
    # than one made close to arrival; similarly for notices.
    #
    # This is not a mechanistic law of motion, but an interpretable proxy
    # for "effective near-term pressure".
    daily["Eff_Net_Flow"] = (
        daily["Bookings"] / (1 + daily["Avg_Booking_Time"])
        - daily["Notices"] / (1 + daily["Avg_Notice_Time"])
    )

    # Lagged versions are added because booking/notice effects may not
    # appear immediately in realized occupancy.
    daily["Eff_Net_Flow_Lag_1"] = daily["Eff_Net_Flow"].shift(1)
    daily["Eff_Net_Flow_Lag_7"] = daily["Eff_Net_Flow"].shift(7)

    # --------------------------------------------------------
    # Occupancy persistence
    # --------------------------------------------------------
    # In accommodation systems, current and recent occupancy are often the
    # strongest short-term predictors. These lags capture persistence and
    # delayed occupancy structure.
    daily["Lag_1d"] = daily["Occupancy"].shift(1)
    daily["Lag_2d"] = daily["Occupancy"].shift(2)
    daily["Lag_3d"] = daily["Occupancy"].shift(3)
    daily["Lag_7d"] = daily["Occupancy"].shift(7)
    daily["Lag_14d"] = daily["Occupancy"].shift(14)
    daily["Lag_21d"] = daily["Occupancy"].shift(21)
    daily["Lag_28d"] = daily["Occupancy"].shift(28)

    # --------------------------------------------------------
    # Short-run movement
    # --------------------------------------------------------
    # Delta features capture recent movement.
    # Acceleration captures turning / curvature in the recent path.
    daily["Delta_1d"] = daily["Lag_1d"] - daily["Lag_2d"]
    daily["Delta_2d"] = daily["Lag_2d"] - daily["Lag_3d"]
    daily["Acceleration"] = daily["Lag_1d"] - 2 * daily["Lag_2d"] + daily["Lag_3d"]

    # --------------------------------------------------------
    # Calendar structure
    # --------------------------------------------------------
    # Day-of-week and day-of-year can matter even if demand is not purely
    # hotel-like, because operational patterns often still have recurring
    # weekly and annual structure.
    daily["DayOfWeek"] = daily["Date"].dt.dayofweek
    daily["DayOfYear"] = daily["Date"].dt.dayofyear

    # Sin/cos encoding avoids artificial discontinuities, such as
    # Sunday -> Monday or Dec 31 -> Jan 1.
    daily["dow_sin"] = np.sin(2 * np.pi * daily["DayOfWeek"] / 7)
    daily["dow_cos"] = np.cos(2 * np.pi * daily["DayOfWeek"] / 7)
    daily["doy_sin"] = np.sin(2 * np.pi * daily["DayOfYear"] / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * daily["DayOfYear"] / 365.25)

    # --------------------------------------------------------
    # Special business periods
    # --------------------------------------------------------
    # These are domain-informed flags for periods where occupancy behavior
    # may deviate from average seasonal patterns.
    daily["Is_Christmas_Period"] = (
        ((daily["Date"].dt.month == 12) & (daily["Date"].dt.day >= 10)) |
        ((daily["Date"].dt.month == 1) & (daily["Date"].dt.day <= 7))
    ).astype(int)

    daily["Is_Summer_Period"] = (
        daily["Date"].dt.isocalendar().week.astype(int).between(28, 36)
    ).astype(int)

    # --------------------------------------------------------
    # Multi-horizon targets
    # --------------------------------------------------------
    # These are the future occupancy levels to be forecast.
    daily["target_t1"] = daily["Occupancy"].shift(-1)
    daily["target_t7"] = daily["Occupancy"].shift(-7)
    daily["target_t14"] = daily["Occupancy"].shift(-14)
    daily["target_t21"] = daily["Occupancy"].shift(-21)
    daily["target_t28"] = daily["Occupancy"].shift(-28)

    # --------------------------------------------------------
    # Delta targets
    # --------------------------------------------------------
    # These are useful when forecasting change rather than level, which is
    # conceptually closer to the state-transition view of the business.
    daily["delta_t1"] = daily["target_t1"] - daily["Occupancy"]
    daily["delta_t7"] = daily["target_t7"] - daily["Occupancy"]
    daily["delta_t14"] = daily["target_t14"] - daily["Occupancy"]
    daily["delta_t21"] = daily["target_t21"] - daily["Occupancy"]
    daily["delta_t28"] = daily["target_t28"] - daily["Occupancy"]

    return daily


def compute_spike_threshold(train: pd.DataFrame, quantile: float = 0.80) -> float:
    """
    Compute a spike threshold using training data only.

    Why training only:
    the threshold is estimated on the train set to avoid leakage from the
    test period. This keeps the spike definition realistic.

    Parameters
    ----------
    train : pd.DataFrame
        Training dataframe with 'Occupancy' and 'target_t1'.
    quantile : float, default 0.80
        Quantile used to define a spike.

    Returns
    -------
    float
        Threshold for absolute one-step occupancy change.
    """
    _validate_columns(train, ["Occupancy", "target_t1"], "compute_spike_threshold")

    train = train.copy()

    # A spike is based on one-step change in occupancy.
    train["Change_t1"] = train["target_t1"] - train["Occupancy"]
    train["abs_change"] = train["Change_t1"].abs()

    return float(train["abs_change"].quantile(quantile))


def apply_spike_labels(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Apply spike labels using a precomputed threshold.

    Labels
    ------
    Spike      -> large absolute change
    Spike_Up   -> large upward change
    Spike_Down -> large downward change
    """
    _validate_columns(df, ["Occupancy", "target_t1"], "apply_spike_labels")

    df = df.copy()
    df["Change_t1"] = df["target_t1"] - df["Occupancy"]
    df["abs_change"] = df["Change_t1"].abs()

    df["Spike"] = (df["abs_change"] > threshold).astype(int)
    df["Spike_Up"] = (df["Change_t1"] > threshold).astype(int)
    df["Spike_Down"] = (df["Change_t1"] < -threshold).astype(int)

    return df


# ============================================================
# TRAIN / TEST SPLIT AND TRAIN-ONLY SEASONALITY
# ============================================================

def split_train_test_80_20_with_seasonality(
    daily_model: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a chronological 80/20 train/test split and create a train-only
    seasonal baseline.

    Forecasting requires preserving time order. Also, the seasonal average
    must be estimated from train only, otherwise future information would
    leak into the test set.

    Returns
    -------
    (train, test) : tuple[pd.DataFrame, pd.DataFrame]
    """
    _validate_columns(
        daily_model,
        ["DayOfYear", "Occupancy"],
        "split_train_test_chronological_with_seasonality"
    )

    if len(daily_model) < 10:
        raise ValueError("daily_model is too short for a meaningful split.")

    split_idx = int(len(daily_model) * 0.8)
    train = daily_model.iloc[:split_idx].copy()
    test = daily_model.iloc[split_idx:].copy()

    # Seasonal baseline estimated only on train.
    seasonal_by_day_train = train.groupby("DayOfYear")["Occupancy"].mean()

    train["Seasonal_Avg"] = train["DayOfYear"].map(seasonal_by_day_train)
    test["Seasonal_Avg"] = test["DayOfYear"].map(seasonal_by_day_train)

    # Fallback if some DayOfYear values are not represented in train.
    global_occ_mean = train["Occupancy"].mean()
    train["Seasonal_Avg"] = train["Seasonal_Avg"].fillna(global_occ_mean)
    test["Seasonal_Avg"] = test["Seasonal_Avg"].fillna(global_occ_mean)

    # Position relative to seasonal norm.
    train["Occ_vs_Seasonal"] = train["Occupancy"] - train["Seasonal_Avg"]
    test["Occ_vs_Seasonal"] = test["Occupancy"] - test["Seasonal_Avg"]

    return train, test


def split_train_test_pre_2019_with_seasonality(
    daily_model: pd.DataFrame,
    split_date: str = "2019-01-01",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a date-based train/test split and create a train-only seasonal baseline.

    Example
    -------
    If split_date = "2019-01-01":
    - train = all rows before 2019-01-01
    - test  = all rows on or after 2019-01-01

    Seasonal averages are estimated on train only and then applied to test.

    Parameters
    ----------
    daily_model : pd.DataFrame
        Modeling dataframe.
    split_date : str, default "2019-01-01"
        Date used to separate train and test.

    Returns
    -------
    (train, test) : tuple[pd.DataFrame, pd.DataFrame]
    """
    _validate_columns(
        daily_model,
        ["Date", "DayOfYear", "Occupancy"],
        "split_train_test_by_date_with_seasonality"
    )

    daily_model = daily_model.copy()
    daily_model["Date"] = pd.to_datetime(daily_model["Date"])
    split_date = pd.Timestamp(split_date)

    train = daily_model[daily_model["Date"] < split_date].copy()
    test = daily_model[daily_model["Date"] >= split_date].copy()

    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Date split at {split_date.date()} produced an empty train or test set."
        )

    # Seasonal baseline estimated only on train.
    seasonal_by_day_train = train.groupby("DayOfYear")["Occupancy"].mean()

    train["Seasonal_Avg"] = train["DayOfYear"].map(seasonal_by_day_train)
    test["Seasonal_Avg"] = test["DayOfYear"].map(seasonal_by_day_train)

    # Fallback if some DayOfYear values are not represented in train.
    global_occ_mean = train["Occupancy"].mean()
    train["Seasonal_Avg"] = train["Seasonal_Avg"].fillna(global_occ_mean)
    test["Seasonal_Avg"] = test["Seasonal_Avg"].fillna(global_occ_mean)

    # Position relative to seasonal norm.
    train["Occ_vs_Seasonal"] = train["Occupancy"] - train["Seasonal_Avg"]
    test["Occ_vs_Seasonal"] = test["Occupancy"] - test["Seasonal_Avg"]

    return train, test


# ============================================================
# POINT FORECASTING MODELS
# ============================================================

def run_linear_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: List[str],
    target_col: str,
    naive_col: str,
    label: str,
) -> Dict[str, Any]:
    """
    Fit one linear regression model for one forecast horizon and compare it
    against the correct naive baseline for that horizon.

    Choosing linear regression:
    it is interpretable, suitable for a relatively small dataset, and aligned
    with the additive state-transition logic of the project.
    """
    _validate_columns(train, features + [target_col], "run_linear_model(train)")
    _validate_columns(test, features + [target_col, naive_col], "run_linear_model(test)")

    X_train = train[features]
    X_test = test[features]
    y_train = train[target_col]
    y_test = test[target_col]

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    naive_pred = test[naive_col]

    mae_model = mean_absolute_error(y_test, pred)
    rmse_model = np.sqrt(mean_squared_error(y_test, pred))

    mae_naive = mean_absolute_error(y_test, naive_pred)
    rmse_naive = np.sqrt(mean_squared_error(y_test, naive_pred))

    # Keeping coefficients is useful for interpretation, not just performance.
    coeff = pd.Series(model.coef_, index=features).sort_values(key=abs, ascending=False)

    print(f"\n=== {label} MODEL ===")
    print("MAE:", round(mae_model, 5))
    print("RMSE:", round(rmse_model, 5))

    print(f"\n=== {label} NAIVE ===")
    print("MAE:", round(mae_naive, 5))
    print("RMSE:", round(rmse_naive, 5))

    print(f"\n=== IMPROVEMENT {label} ===")
    print("MAE improvement:", round(mae_naive - mae_model, 5))

    print(f"\n=== {label} COEFFICIENTS ===")
    print(coeff)

    return {
        "model": model,
        "pred": pred,
        "y_test": y_test,
        "mae_model": mae_model,
        "rmse_model": rmse_model,
        "mae_naive": mae_naive,
        "rmse_naive": rmse_naive,
        "coeff": coeff,
    }


def run_ridge_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: List[str],
    target_col: str,
    naive_col: str,
    label: str,
    alpha: float = 1.0,
) -> Tuple[Ridge, np.ndarray]:
    """
    Fit a Ridge regression benchmark.

    Ridge is useful as a robustness check when features may be correlated
    (for example Net_Flow, Demand_Pressure, Booking_Intensity, and
    Notice_Pressure).

    In this project it is best treated as an extension, not the main model.
    """
    _validate_columns(train, features + [target_col], "run_ridge_model(train)")
    _validate_columns(test, features + [target_col, naive_col], "run_ridge_model(test)")

    X_train = train[features]
    X_test = test[features]
    y_train = train[target_col]
    y_test = test[target_col]

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    naive_pred = test[naive_col]

    mae_model = mean_absolute_error(y_test, pred)
    mae_naive = mean_absolute_error(y_test, naive_pred)

    print(f"\n=== {label} RIDGE ===")
    print("MAE:", round(mae_model, 5))
    print("Improvement vs naive:", round(mae_naive - mae_model, 5))

    return model, pred


# ============================================================
# GUARDRAIL AND SPIKE ANALYSIS
# ============================================================

def build_guardrail(
    train: pd.DataFrame,
    test: pd.DataFrame,
    quantile: float = 0.80,
) -> pd.Series:
    """
    Build an instability warning flag from extreme booking or notice pressure.

    Important
    ---------
    This is not meant to be a forecasting feature. It is an operational
    warning indicator used to mark periods where forecast reliability is
    expected to be lower.
    """
    required_cols = ["Booking_Intensity", "Notice_Pressure"]
    _validate_columns(train, required_cols, "build_guardrail(train)")
    _validate_columns(test, required_cols, "build_guardrail(test)")

    b_thresh = train["Booking_Intensity"].quantile(quantile)
    n_thresh = train["Notice_Pressure"].quantile(quantile)

    # Flag test cases that exceed train-based high-pressure thresholds.
    guardrail_test = (
        (test["Booking_Intensity"] > b_thresh) |
        (test["Notice_Pressure"] > n_thresh)
    ).astype(int)

    print("\nGuardrail counts:")
    print(guardrail_test.value_counts())

    return guardrail_test


def evaluate_with_guardrail(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    guardrail: pd.Series | np.ndarray,
    label: str,
) -> Dict[str, float]:
    """
    Compare model error under stable vs unstable conditions.

    This helps test whether the guardrail is meaningful in practice:
    if MAE is systematically worse when the guardrail is ON, then the
    warning signal adds operational value.
    """
    df_eval = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred,
        "Guardrail": guardrail,
    }).dropna()

    overall_mae = mean_absolute_error(df_eval["Actual"], df_eval["Predicted"])

    safe_df = df_eval[df_eval["Guardrail"] == 0]
    risk_df = df_eval[df_eval["Guardrail"] == 1]

    mae_safe = mean_absolute_error(safe_df["Actual"], safe_df["Predicted"]) if len(safe_df) else np.nan
    mae_risk = mean_absolute_error(risk_df["Actual"], risk_df["Predicted"]) if len(risk_df) else np.nan
    diff = mae_risk - mae_safe if pd.notna(mae_safe) and pd.notna(mae_risk) else np.nan

    print(f"\n=== {label} ===")
    print("Overall MAE:", round(overall_mae, 5))
    print("MAE (Guardrail OFF - stable):", round(mae_safe, 5) if pd.notna(mae_safe) else np.nan)
    print("MAE (Guardrail ON - unstable):", round(mae_risk, 5) if pd.notna(mae_risk) else np.nan)
    print("Difference:", round(diff, 5) if pd.notna(diff) else np.nan)

    return {
        "Horizon": label,
        "Overall_MAE": overall_mae,
        "MAE_Stable": mae_safe,
        "MAE_Unstable": mae_risk,
        "Difference": diff,
    }


def analyze_spikes(daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Explore whether spikes are random or clustered in recurring periods.

    Returns
    -------
    spikes : pd.DataFrame
        Rows labeled as spikes.
    spike_pivot : pd.DataFrame
        Spike counts by week-of-year and year.
    """
    _validate_columns(daily, ["Date", "Spike", "Spike_Up", "Spike_Down"], "analyze_spikes")

    spikes = daily[daily["Spike"] == 1].copy()
    spikes["Date"] = pd.to_datetime(spikes["Date"])

    spikes["Month"] = spikes["Date"].dt.month
    spikes["WeekOfYear"] = spikes["Date"].dt.isocalendar().week.astype(int)
    spikes["Year"] = spikes["Date"].dt.year

    print("\nSpikes by month")
    print(spikes["Month"].value_counts().sort_index())

    spike_pivot = spikes.pivot_table(
        index="WeekOfYear",
        columns="Year",
        values="Spike",
        aggfunc="count",
    ).fillna(0)

    print("\nSpikes by week and year")
    print(spike_pivot)

    print("\nUpward spikes by month")
    print(spikes[spikes["Spike_Up"] == 1]["Month"].value_counts().sort_index())

    print("\nDownward spikes by month")
    print(spikes[spikes["Spike_Down"] == 1]["Month"].value_counts().sort_index())

    return spikes, spike_pivot


# ============================================================
# DIRECTIONAL MODELING
# ============================================================

def add_direction_target(
    df: pd.DataFrame,
    target_col: str,
    new_col: str,
    threshold: float = 0.005,
) -> pd.DataFrame:
    """
    Create directional labels for a given forecast horizon.

    Labels
    ------
    1   -> Up
    -1  -> Down
    0   -> Stable

    Important
    ---------
    This is not a spike-only label.
    It includes all meaningful movements and excludes only very small changes.
    """
    _validate_columns(df, [target_col, "Occupancy"], "add_direction_target")

    df = df.copy()

    # Direction is defined relative to current known occupancy.
    change = df[target_col] - df["Occupancy"]

    # Default = Stable.
    df[new_col] = 0

    # Assign Up / Down only when the move is large enough to be considered
    # operationally relevant.
    df.loc[change > threshold, new_col] = 1
    df.loc[change < -threshold, new_col] = -1

    return df


def run_direction_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: List[str],
    target_col: str,
    label: str,
) -> Tuple[LogisticRegression, np.ndarray, np.ndarray, pd.Series, pd.DataFrame]:
    """
    Train a directional classifier for one horizon.

    Stable cases are excluded from training and evaluation, so the classifier
    focuses only on meaningful Up vs Down movement.

    Choosing logistic regression:
    it is interpretable, suitable for a relatively small dataset, and useful
    for generating class probabilities that can later be converted into
    selective operational signals.
    """
    _validate_columns(train, features + [target_col], "run_direction_model(train)")
    _validate_columns(test, features + [target_col], "run_direction_model(test)")

    train_clf = train[train[target_col] != 0].copy()
    test_clf = test[test[target_col] != 0].copy()

    if len(train_clf) == 0 or len(test_clf) == 0:
        raise ValueError(f"{label}: no non-stable cases available for directional modeling.")

    X_train = train_clf[features]
    y_train = train_clf[target_col]

    X_test = test_clf[features]
    y_test = test_clf[target_col]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # These probabilities are generated using feature inputs only.
    # They do not use future actual occupancy from the test period.
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    print(f"\n=== {label} DIRECTION MODEL ===")
    print(classification_report(y_test, pred))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, pred))

    return model, pred, proba, y_test, test_clf


def add_selective_direction_output(
    test_clf: pd.DataFrame,
    pred: np.ndarray,
    proba: np.ndarray,
    target_col: str,
    class_order: np.ndarray,
    threshold: float = 0.55,
) -> pd.DataFrame:
    """
    Convert raw directional probabilities into selective signals:
    UP / DOWN / UNCERTAIN.

    In operations, it is better for the model to abstain than to force
    a weak signal into a definite action.
    """
    _validate_columns(test_clf, [target_col], "add_selective_direction_output")

    df = test_clf.copy().reset_index(drop=True)

    class_to_idx = {cls: i for i, cls in enumerate(class_order)}

    if -1 not in class_to_idx or 1 not in class_to_idx:
        raise ValueError("Directional model must contain both classes -1 and 1.")

    down_idx = class_to_idx[-1]
    up_idx = class_to_idx[1]

    df["Prob_Down"] = proba[:, down_idx]
    df["Prob_Up"] = proba[:, up_idx]
    df["Direction_Pred"] = pred

    df["Selective_Direction"] = "UNCERTAIN"
    df.loc[df["Prob_Up"] >= threshold, "Selective_Direction"] = "UP"
    df.loc[df["Prob_Down"] >= threshold, "Selective_Direction"] = "DOWN"

    df["True_Label"] = df[target_col].map({1: "UP", -1: "DOWN"})

    return df


def evaluate_selective_direction(df_selective: pd.DataFrame, label: str) -> Dict[str, Any]:
    """
    Evaluate only the confident directional predictions.

    Main metrics
    ------------
    Coverage           - how often the model was confident enough to act
    Selective_Accuracy - accuracy on that confident subset
    """
    _validate_columns(df_selective, ["Selective_Direction", "True_Label"], "evaluate_selective_direction")

    eval_df = df_selective[df_selective["Selective_Direction"] != "UNCERTAIN"].copy()

    if len(eval_df) == 0:
        print(f"\n=== {label} SELECTIVE DIRECTION EVALUATION ===")
        print("No confident directional predictions at this threshold.")
        return {
            "Horizon": label,
            "Coverage": 0.0,
            "Selective_Accuracy": np.nan,
            "Confident_Count": 0,
            "UP_Count": 0,
            "DOWN_Count": 0,
        }

    accuracy = (eval_df["Selective_Direction"] == eval_df["True_Label"]).mean()
    coverage = len(eval_df) / len(df_selective)

    print(f"\n=== {label} SELECTIVE DIRECTION EVALUATION ===")
    print("Coverage:", round(coverage, 3))
    print("Accuracy on confident predictions:", round(accuracy, 3))

    print("\nCounts:")
    print(eval_df["Selective_Direction"].value_counts())

    return {
        "Horizon": label,
        "Coverage": coverage,
        "Selective_Accuracy": accuracy,
        "Confident_Count": len(eval_df),
        "UP_Count": int((eval_df["Selective_Direction"] == "UP").sum()),
        "DOWN_Count": int((eval_df["Selective_Direction"] == "DOWN").sum()),
    }


def run_direction_workflow(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: List[str],
    target_col: str,
    label: str,
    threshold: float = 0.55,
) -> Dict[str, Any]:
    """
    Run the full directional workflow for one horizon:
    - train classifier
    - create selective UP / DOWN / UNCERTAIN output
    - evaluate confident predictions only
    """
    model, pred, proba, y_test, test_clf = run_direction_model(
        train=train,
        test=test,
        features=features,
        target_col=target_col,
        label=label,
    )

    selective = add_selective_direction_output(
        test_clf=test_clf,
        pred=pred,
        proba=proba,
        target_col=target_col,
        class_order=model.classes_,
        threshold=threshold,
    )

    summary = evaluate_selective_direction(selective, label)

    return {
        "model": model,
        "pred": pred,
        "proba": proba,
        "y_test": y_test,
        "test_clf": test_clf,
        "selective": selective,
        "summary": summary,
    }


# ============================================================
# MULTI-HORIZON OPERATIONAL OUTPUT
# ============================================================

def build_operational_signal_multi(
    test: pd.DataFrame,
    pred_t7: np.ndarray,
    pred_t14: np.ndarray,
    pred_t21: np.ndarray,
    pred_t28: np.ndarray,
    model_t7: LogisticRegression,
    model_t14: LogisticRegression,
    model_t21: LogisticRegression,
    model_t28: LogisticRegression,
    features: List[str],
    guardrail: pd.Series | np.ndarray,
    threshold: float = 0.55,
) -> pd.DataFrame:
    """
    Build a multi-horizon operational output using:
    - T+7, T+14, T+21, T+28 occupancy forecasts
    - directional models at each horizon
    - guardrail-based confidence
    - hierarchical action logic

    This function is where the project moves from prediction toward
    decision support.
    """
    _validate_columns(test, ["Date", "Occupancy"] + features, "build_operational_signal_multi")

    df = test.copy().reset_index(drop=True)

    if len(df) != len(guardrail):
        raise ValueError("Length mismatch between test dataframe and guardrail.")
    if not (len(df) == len(pred_t7) == len(pred_t14) == len(pred_t21) == len(pred_t28)):
        raise ValueError("Prediction arrays must have the same length as test.")

    # --------------------------------------------------------
    # Instability / confidence layer
    # --------------------------------------------------------
    df["Guardrail"] = np.array(guardrail)

    # --------------------------------------------------------
    # Point forecasts
    # --------------------------------------------------------
    df["Predicted_Occ_t7"] = np.array(pred_t7)
    df["Predicted_Occ_t14"] = np.array(pred_t14)
    df["Predicted_Occ_t21"] = np.array(pred_t21)
    df["Predicted_Occ_t28"] = np.array(pred_t28)

    # --------------------------------------------------------
    # Direction probabilities
    # --------------------------------------------------------
    # Important: these use current/past feature inputs only.
    # They do not use future actual occupancy from the test period.
    def extract_up_down_proba(model: LogisticRegression, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        proba = model.predict_proba(X)
        class_to_idx = {cls: i for i, cls in enumerate(model.classes_)}

        if -1 not in class_to_idx or 1 not in class_to_idx:
            raise ValueError("Directional model must contain both classes -1 and 1.")

        down_idx = class_to_idx[-1]
        up_idx = class_to_idx[1]

        return proba[:, down_idx], proba[:, up_idx]

    prob_down_t7, prob_up_t7 = extract_up_down_proba(model_t7, df[features])
    prob_down_t14, prob_up_t14 = extract_up_down_proba(model_t14, df[features])
    prob_down_t21, prob_up_t21 = extract_up_down_proba(model_t21, df[features])
    prob_down_t28, prob_up_t28 = extract_up_down_proba(model_t28, df[features])

    df["Prob_Down_t7"] = prob_down_t7
    df["Prob_Up_t7"] = prob_up_t7

    df["Prob_Down_t14"] = prob_down_t14
    df["Prob_Up_t14"] = prob_up_t14

    df["Prob_Down_t21"] = prob_down_t21
    df["Prob_Up_t21"] = prob_up_t21

    df["Prob_Down_t28"] = prob_down_t28
    df["Prob_Up_t28"] = prob_up_t28

    # --------------------------------------------------------
    # Selective directional labels
    # --------------------------------------------------------
    # If neither class is confident enough, keep the horizon UNCERTAIN.
    df["Direction_t7"] = "UNCERTAIN"
    df.loc[df["Prob_Up_t7"] >= threshold, "Direction_t7"] = "UP"
    df.loc[df["Prob_Down_t7"] >= threshold, "Direction_t7"] = "DOWN"

    df["Direction_t14"] = "UNCERTAIN"
    df.loc[df["Prob_Up_t14"] >= threshold, "Direction_t14"] = "UP"
    df.loc[df["Prob_Down_t14"] >= threshold, "Direction_t14"] = "DOWN"

    df["Direction_t21"] = "UNCERTAIN"
    df.loc[df["Prob_Up_t21"] >= threshold, "Direction_t21"] = "UP"
    df.loc[df["Prob_Down_t21"] >= threshold, "Direction_t21"] = "DOWN"

    df["Direction_t28"] = "UNCERTAIN"
    df.loc[df["Prob_Up_t28"] >= threshold, "Direction_t28"] = "UP"
    df.loc[df["Prob_Down_t28"] >= threshold, "Direction_t28"] = "DOWN"

    # --------------------------------------------------------
    # Confidence label
    # --------------------------------------------------------
    # The guardrail reduces confidence when the system is under unusually
    # high booking or notice pressure.
    df["Confidence"] = np.where(df["Guardrail"] == 1, "LOW", "HIGH")

    # --------------------------------------------------------
    # Hierarchical action logic
    # --------------------------------------------------------
    # Longer horizons are treated as more strategic, shorter horizons as
    # more tactical. Instability always overrides action confidence.
    def decide_action(row: pd.Series) -> str:
        # 1. Instability overrides everything.
        if row["Confidence"] == "LOW":
            return "Monitor (unstable)"

        # 2. Strong agreement between the longer horizons.
        if (row["Direction_t28"] == "UP") and (row["Direction_t21"] == "UP"):
            return "Strong demand increase → act aggressively"

        if (row["Direction_t28"] == "DOWN") and (row["Direction_t21"] == "DOWN"):
            return "Strong demand drop → act aggressively"

        # 3. Strategic signal from the longest horizon.
        if row["Direction_t28"] == "UP":
            return "Increase pricing (strategic)"

        if row["Direction_t28"] == "DOWN":
            return "Reduce pricing / push demand"

        # 4. Medium-term preparation.
        if row["Direction_t21"] == "UP":
            return "Prepare for demand increase"

        if row["Direction_t21"] == "DOWN":
            return "Prepare promotions"

        # 5. Tactical layer.
        if row["Direction_t14"] == "UP":
            return "Adjust pricing upward (tactical)"

        if row["Direction_t14"] == "DOWN":
            return "Adjust promotions (tactical)"

        if row["Direction_t7"] == "UP":
            return "Increase pricing / reduce availability"

        if row["Direction_t7"] == "DOWN":
            return "Promote / increase availability"

        # 6. Default if no strong directional signal is available.
        return "Maintain current settings"

    df["Action"] = df.apply(decide_action, axis=1)

    return df[[
        "Date",
        "Occupancy",
        "Predicted_Occ_t7",
        "Predicted_Occ_t14",
        "Predicted_Occ_t21",
        "Predicted_Occ_t28",
        "Prob_Down_t7",
        "Prob_Up_t7",
        "Prob_Down_t14",
        "Prob_Up_t14",
        "Prob_Down_t21",
        "Prob_Up_t21",
        "Prob_Down_t28",
        "Prob_Up_t28",
        "Direction_t7",
        "Direction_t14",
        "Direction_t21",
        "Direction_t28",
        "Guardrail",
        "Confidence",
        "Action",
    ]]

# ============================================================
# SUMMARY HELPERS
# ============================================================

def build_forecast_summary(results_by_horizon: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a compact forecast summary from run_linear_model outputs.

    Example input
    -------------
    {
        "T+1": result_t1,
        "T+7": result_t7,
        ...
    }
    """
    rows = []
    for horizon, result in results_by_horizon.items():
        rows.append({
            "Horizon": horizon,
            "Model_MAE": result["mae_model"],
            "Naive_MAE": result["mae_naive"],
            "MAE_Improvement": result["mae_naive"] - result["mae_model"],
        })
    return pd.DataFrame(rows)


def build_guardrail_summary(guardrail_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of guardrail evaluation dictionaries into a dataframe.
    """
    return pd.DataFrame(guardrail_results)


def build_direction_summary(direction_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a summary dataframe from directional workflow outputs.

    Example input
    -------------
    {
        "T+1": direction_results["T+1"],
        "T+7": direction_results["T+7"],
        ...
    }
    """
    rows = [v["summary"] for v in direction_results.values()]
    return pd.DataFrame(rows)


# ============================================================
# INTERNAL VALIDATION HELPER
# ============================================================

def _validate_columns(df: pd.DataFrame, columns: List[str], func_name: str) -> None:
    """
    Internal helper to ensure required columns exist before a function runs.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{func_name}: missing required columns: {missing}")


# ============================================================
# MODEL PERSISTENCE (SAVE / LOAD TRAINED ARTIFACTS)
# ============================================================

def save_trained_artifacts(
    save_dir: str,
    point_models: Dict[str, Any],
    direction_models: Dict[str, Any],
    metadata: Dict[str, Any]
) -> None:
    """
    Save trained point models, directional models, and metadata to disk.
    """
    os.makedirs(save_dir, exist_ok=True)

    for horizon, model in point_models.items():
        joblib.dump(model, os.path.join(save_dir, f"point_model_{horizon}.joblib"))

    for horizon, model in direction_models.items():
        joblib.dump(model, os.path.join(save_dir, f"direction_model_{horizon}.joblib"))

    joblib.dump(metadata, os.path.join(save_dir, "metadata.joblib"))

    print(f"Artifacts saved to: {save_dir}")


def load_trained_artifacts(save_dir: str) -> Dict[str, Any]:
    """
    Load trained point models, directional models, and metadata from disk.
    """
    point_models = {}
    direction_models = {}

    for horizon in ["t1", "t7", "t14", "t21", "t28"]:
        point_path = os.path.join(save_dir, f"point_model_{horizon}.joblib")
        direction_path = os.path.join(save_dir, f"direction_model_{horizon}.joblib")

        if os.path.exists(point_path):
            point_models[horizon] = joblib.load(point_path)

        if os.path.exists(direction_path):
            direction_models[horizon] = joblib.load(direction_path)

    metadata = joblib.load(os.path.join(save_dir, "metadata.joblib"))

    return {
        "point_models": point_models,
        "direction_models": direction_models,
        "metadata": metadata
    }


def apply_saved_seasonality(df: pd.DataFrame, seasonal_by_day: pd.Series) -> pd.DataFrame:
    """
    Apply a saved train-only seasonal baseline to a dataframe.
    """
    _validate_columns(df, ["DayOfYear", "Occupancy"], "apply_saved_seasonality")

    df = df.copy()

    global_occ_mean = float(seasonal_by_day.mean())

    df["Seasonal_Avg"] = df["DayOfYear"].map(seasonal_by_day)
    df["Seasonal_Avg"] = df["Seasonal_Avg"].fillna(global_occ_mean)
    df["Occ_vs_Seasonal"] = df["Occupancy"] - df["Seasonal_Avg"]

    return df


def build_guardrail_from_saved_thresholds(
    df: pd.DataFrame,
    booking_threshold: float,
    notice_threshold: float
) -> pd.Series:
    """
    Apply previously learned guardrail thresholds to new data.
    """
    _validate_columns(df, ["Booking_Intensity", "Notice_Pressure"], "build_guardrail_from_saved_thresholds")

    return (
        (df["Booking_Intensity"] > booking_threshold) |
        (df["Notice_Pressure"] > notice_threshold)
    ).astype(int)

def extract_up_down_proba(model, X):
    """
    Extract DOWN and UP probabilities from a fitted directional classifier
    using the classifier's own class ordering.
    """
    proba = model.predict_proba(X)
    class_to_idx = {cls: i for i, cls in enumerate(model.classes_)}

    if -1 not in class_to_idx or 1 not in class_to_idx:
        raise ValueError("Directional model must contain both classes -1 and 1.")

    down_idx = class_to_idx[-1]
    up_idx = class_to_idx[1]

    return proba[:, down_idx], proba[:, up_idx]


def classify_strength(prob_up, prob_down, weak_thr=0.55, strong_thr=0.70):
    """
    Convert directional probabilities into a strength band.

    Returns one of:
    - STRONG_UP
    - MODERATE_UP
    - STRONG_DOWN
    - MODERATE_DOWN
    - UNCERTAIN
    """
    if prob_up >= strong_thr:
        return "STRONG_UP"
    if prob_down >= strong_thr:
        return "STRONG_DOWN"
    if prob_up >= weak_thr:
        return "MODERATE_UP"
    if prob_down >= weak_thr:
        return "MODERATE_DOWN"
    return "UNCERTAIN"


def alignment_band(score):
    """
    Convert a numeric alignment score into a reliability band.
    """
    if pd.isna(score):
        return "UNKNOWN"
    if score >= 0.65:
        return "HIGH"
    if score >= 0.50:
        return "MEDIUM"
    return "LOW"


def calibrated_action(row: pd.Series) -> str:
    """
    Convert guardrail, alignment, and directional strength into a final
    operational recommendation.
    """
    # 1. Structural instability always overrides
    if row["Guardrail"] == 1:
        return "Monitor (unstable)"

    # 2. Low recent reliability -> stay cautious
    if row["Alignment_Band"] == "LOW":
        return "Monitor (low recent reliability)"

    # 3. Strong actions require HIGH alignment
    if row["Strength_t28"] == "STRONG_UP" and row["Strength_t21"] == "STRONG_UP":
        if row["Alignment_Band"] == "HIGH":
            return "Strong demand increase → act aggressively"

    if row["Strength_t28"] == "STRONG_DOWN" and row["Strength_t21"] == "STRONG_DOWN":
        if row["Alignment_Band"] == "HIGH":
            return "Strong demand drop → act aggressively"

    # 4. UNKNOWN alignment:
    # allow strategic actions, but do not allow aggressive actions
    if row["Alignment_Band"] == "UNKNOWN":
        if row["Strength_t28"] in ["STRONG_UP", "MODERATE_UP"]:
            return "Increase pricing (strategic)"
        if row["Strength_t28"] in ["STRONG_DOWN", "MODERATE_DOWN"]:
            return "Reduce pricing / push demand"
        if row["Strength_t21"] in ["STRONG_UP", "MODERATE_UP"]:
            return "Prepare for demand increase"
        if row["Strength_t21"] in ["STRONG_DOWN", "MODERATE_DOWN"]:
            return "Prepare promotions"
        return "Maintain current settings"

    # 5. HIGH or MEDIUM alignment:
    # allow strategic and preparatory actions
    if row["Strength_t28"] in ["STRONG_UP", "MODERATE_UP"] and row["Alignment_Band"] in ["HIGH", "MEDIUM"]:
        return "Increase pricing (strategic)"

    if row["Strength_t28"] in ["STRONG_DOWN", "MODERATE_DOWN"] and row["Alignment_Band"] in ["HIGH", "MEDIUM"]:
        return "Reduce pricing / push demand"

    if row["Strength_t21"] in ["STRONG_UP", "MODERATE_UP"] and row["Alignment_Band"] in ["HIGH", "MEDIUM"]:
        return "Prepare for demand increase"

    if row["Strength_t21"] in ["STRONG_DOWN", "MODERATE_DOWN"] and row["Alignment_Band"] in ["HIGH", "MEDIUM"]:
        return "Prepare promotions"

    # 6. Tactical actions only under HIGH alignment
    if row["Strength_t14"] in ["STRONG_UP", "MODERATE_UP"] and row["Alignment_Band"] == "HIGH":
        return "Adjust pricing upward (tactical)"

    if row["Strength_t14"] in ["STRONG_DOWN", "MODERATE_DOWN"] and row["Alignment_Band"] == "HIGH":
        return "Adjust promotions (tactical)"

    if row["Strength_t7"] in ["STRONG_UP", "MODERATE_UP"] and row["Alignment_Band"] == "HIGH":
        return "Increase pricing / reduce availability"

    if row["Strength_t7"] in ["STRONG_DOWN", "MODERATE_DOWN"] and row["Alignment_Band"] == "HIGH":
        return "Promote / increase availability"

    return "Maintain current settings"


def build_operational_signal_multi_calibrated(
    df: pd.DataFrame,
    point_models: Dict[str, Any],
    direction_models: Dict[str, Any],
    features_t1: List[str],
    features_medium: List[str],
    guardrail: pd.Series | np.ndarray,
    direction_threshold: float,
    window: int = 21,
    weak_thr: float = 0.55,
    strong_thr: float = 0.70
) -> pd.DataFrame:
    """
    Build a calibrated multi-horizon operational scoring table using:

    - saved point forecast models
    - saved directional models
    - guardrail-based instability flags
    - directional strength bands
    - a rolling alignment layer when realized outcomes are available

    Purpose
    -------
    This function is the production-style version of the operational layer.
    It is designed to score the latest available rows without retraining the
    models.

    Compared with the earlier operational output, this calibrated version
    adds two important ideas:

    1. Directional strength
       Not all UP/DOWN signals should be treated equally. Probabilities are
       split into STRONG, MODERATE, and UNCERTAIN bands.

    2. Reliability-aware action logic
       Operational actions are made more cautious by combining:
       - structural instability (guardrail)
       - directional strength
       - an alignment band when available

    Important note on alignment
    ---------------------------
    If realized future direction columns are available in df (for example
    Direction_t1 and Direction_t7 on historical rows), the function computes
    matured correctness and a rolling alignment score.

    If those realized outcomes are not available, alignment remains UNKNOWN.
    In that case, the function still produces valid forecasts, probabilities,
    strength bands, and guardrail-aware operational actions.

    Parameters
    ----------
    df : pd.DataFrame
        Scoring dataframe containing the latest rows with all required
        engineered features already present.

    point_models : dict
        Dictionary of saved point forecast models, expected keys:
        't7', 't14', 't21', 't28'. Optionally 't1'.

    direction_models : dict
        Dictionary of saved directional models, expected keys:
        't1', 't7', 't14', 't21', 't28'.

    features_t1 : list[str]
        Feature set used by the T+1 directional model.

    features_medium : list[str]
        Feature set used by medium-horizon models (T+7 to T+28).

    guardrail : pd.Series or np.ndarray
        Instability flag aligned row-by-row with df.

    direction_threshold : float
        Probability threshold used to convert directional probabilities into
        directional labels (-1, 0, 1).

    window : int, default 21
        Rolling window used for alignment-score smoothing when matured
        correctness is available.

    weak_thr : float, default 0.55
        Minimum probability required to classify a directional signal as
        MODERATE_UP or MODERATE_DOWN.

    strong_thr : float, default 0.70
        Minimum probability required to classify a directional signal as
        STRONG_UP or STRONG_DOWN.

    Returns
    -------
    pd.DataFrame
        Copy of the scoring dataframe enriched with:
        - point forecasts
        - directional probabilities
        - directional predictions
        - strength bands
        - alignment columns
        - guardrail
        - calibrated operational action
    """
    _validate_columns(df, ["Date", "Occupancy"] + features_t1 + features_medium, "build_operational_signal_multi_calibrated")

    df = df.copy().reset_index(drop=True)

    if len(df) != len(guardrail):
        raise ValueError("Length mismatch between dataframe and guardrail.")

    # --------------------------------------------------------
    # Point forecasts
    # --------------------------------------------------------
    df["Pred_t7"] = point_models["t7"].predict(df[features_medium])
    df["Pred_t14"] = point_models["t14"].predict(df[features_medium])
    df["Pred_t21"] = point_models["t21"].predict(df[features_medium])
    df["Pred_t28"] = point_models["t28"].predict(df[features_medium])

    if "t1" in point_models:
        df["Pred_t1"] = point_models["t1"].predict(df[features_t1])

    # --------------------------------------------------------
    # Direction probabilities and directional predictions
    # --------------------------------------------------------
    for h in ["t1", "t7", "t14", "t21", "t28"]:
        feats = features_t1 if h == "t1" else features_medium

        prob_down, prob_up = extract_up_down_proba(direction_models[h], df[feats])

        df[f"Prob_Down_{h}"] = prob_down
        df[f"Prob_Up_{h}"] = prob_up

        df[f"Direction_Pred_{h}"] = np.where(
            df[f"Prob_Up_{h}"] >= direction_threshold, 1,
            np.where(df[f"Prob_Down_{h}"] >= direction_threshold, -1, 0)
        )

    # --------------------------------------------------------
    # Strength bands
    # --------------------------------------------------------
    for h in ["t7", "t14", "t21", "t28"]:
        df[f"Strength_{h}"] = [
            classify_strength(u, d, weak_thr=weak_thr, strong_thr=strong_thr)
            for u, d in zip(df[f"Prob_Up_{h}"], df[f"Prob_Down_{h}"])
        ]

    # --------------------------------------------------------
    # Alignment layer
    # --------------------------------------------------------
    # If true direction columns exist, compute correctness and matured
    # correctness exactly as in Notebook 3 for the historical part of df.
    has_true_t1 = "Direction_t1" in df.columns
    has_true_t7 = "Direction_t7" in df.columns

    if has_true_t1:
        df["Correct_t1_confident_only"] = np.where(
            df["Direction_Pred_t1"] != 0,
            (df["Direction_Pred_t1"] == df["Direction_t1"]).astype(float),
            np.nan
        )
        df["Matured_T1_Correct"] = df["Correct_t1_confident_only"].shift(1)
    else:
        df["Correct_t1_confident_only"] = np.nan
        df["Matured_T1_Correct"] = np.nan

    if has_true_t7:
        df["Correct_t7_confident_only"] = np.where(
            df["Direction_Pred_t7"] != 0,
            (df["Direction_Pred_t7"] == df["Direction_t7"]).astype(float),
            np.nan
        )
        df["Matured_T7_Correct"] = df["Correct_t7_confident_only"].shift(7)
    else:
        df["Correct_t7_confident_only"] = np.nan
        df["Matured_T7_Correct"] = np.nan

    df[f"Recent_T1_Accuracy_{window}"] = df["Matured_T1_Correct"].rolling(window, min_periods=3).mean()
    df[f"Recent_T7_Accuracy_{window}"] = df["Matured_T7_Correct"].rolling(window, min_periods=3).mean()

    df[f"Alignment_Score_{window}"] = (
        0.30 * df[f"Recent_T1_Accuracy_{window}"] +
        0.70 * df[f"Recent_T7_Accuracy_{window}"]
    )

    df["Alignment_Band"] = df[f"Alignment_Score_{window}"].apply(alignment_band)

    # --------------------------------------------------------
    # Guardrail
    # --------------------------------------------------------
    df["Guardrail"] = np.array(guardrail)

    # --------------------------------------------------------
    # Final action
    # --------------------------------------------------------
    df["Calibrated_Action"] = df.apply(calibrated_action, axis=1)

    return df