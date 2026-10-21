import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

BIRTH_FEATURES = [
    "start_height_z",
    "start_weight_z",
    "Sex"
]

PREDICTION_FEATURES = [
    # baseline measurements
    "start_height_z",
    "start_weight_z",

    # early growth windows
    "height_z_0_6",
    "height_z_6_12",

    "weight_z_0_6",
    "weight_z_6_12",

    # demographics
    "Sex"
]

TARGET_COLUMNS = [
    "end_height_z",
    "end_weight_z"
]


def prepare_data(features, feature_set):
    data = features.copy()

    data["Sex"] = data["Sex"].map(
        {
            "Male": 1,
            "Female": 0
        }
    )

    X = data[feature_set].copy()
    y = data[TARGET_COLUMNS].copy()

    return X, y


def evaluate_model(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):
    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    height_true = y_test["end_height_z"]
    weight_true = y_test["end_weight_z"]

    height_pred = predictions[:, 0]
    weight_pred = predictions[:, 1]

    height_r2 = r2_score(height_true, height_pred)
    weight_r2 = r2_score(weight_true, weight_pred)

    height_mae = mean_absolute_error(height_true, height_pred)
    weight_mae = mean_absolute_error(weight_true, weight_pred)

    height_rmse = root_mean_squared_error(height_true, height_pred)
    weight_rmse = root_mean_squared_error(weight_true, weight_pred)

    print("\nHEIGHT Z-SCORE")
    print(f"R²   : {height_r2:.4f}")
    print(f"MAE  : {height_mae:.4f}")
    print(f"RMSE : {height_rmse:.4f}")

    print("\nWEIGHT Z-SCORE")
    print(f"R²   : {weight_r2:.4f}")
    print(f"MAE  : {weight_mae:.4f}")
    print(f"RMSE : {weight_rmse:.4f}")

    return {
        "model": model_name,
        "height_r2": height_r2,
        "weight_r2": weight_r2,
        "height_mae": height_mae,
        "weight_mae": weight_mae,
        "height_rmse": height_rmse,
        "weight_rmse": weight_rmse
    }


def print_linear_interpretation(linear_model):
    model = linear_model.named_steps["model"]

    coef_df = pd.DataFrame(
        {
            "feature": PREDICTION_FEATURES,
            "height_coef": model.coef_[0],
            "weight_coef": model.coef_[1]
        }
    )

    print("\n" + "=" * 60)
    print("LINEAR REGRESSION COEFFICIENTS (FULL MODEL)")
    print("=" * 60)

    print("\nHEIGHT OUTCOME")
    print(
        coef_df[["feature", "height_coef"]]
        .assign(abs_coef=lambda x: x["height_coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .to_string(index=False)
    )

    print("\nWEIGHT OUTCOME")
    print(
        coef_df[["feature", "weight_coef"]]
        .assign(abs_coef=lambda x: x["weight_coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .to_string(index=False)
    )


def run_prediction_models(features):
    results = []

    # =====================================================
    # 1. Birth-Only Baseline Model (Linear Regression)
    # =====================================================
    print("\n" + "=" * 60)
    print("RUNNING BIRTH-ONLY BASELINE MODEL")
    print("=" * 60)

    X_birth, y_birth = prepare_data(features, BIRTH_FEATURES)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_birth, y_birth, test_size=0.20, random_state=42
    )

    birth_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    results.append(
        evaluate_model(
            "Birth-Only Linear Baseline",
            birth_model,
            X_train_b,
            X_test_b,
            y_train_b,
            y_test_b
        )
    )

    # =====================================================
    # 2. Full First-Year Models
    # =====================================================
    print("\n" + "=" * 60)
    print("RUNNING FULL FIRST-YEAR MODELS")
    print("=" * 60)

    X_full, y_full = prepare_data(features, PREDICTION_FEATURES)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.20, random_state=42
    )

    # Linear Regression
    linear_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    results.append(
        evaluate_model(
            "Full Linear Regression",
            linear_model,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    print_linear_interpretation(linear_model)

    # Random Forest
    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=500, random_state=42, n_jobs=-1
        ))
    ])

    results.append(
        evaluate_model(
            "Full Random Forest",
            random_forest,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # MLP
    mlp = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=1000,
            random_state=42
        ))
    ])

    results.append(
        evaluate_model(
            "Full MLP Regressor",
            mlp,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # =====================================================
    # Model Comparison Summary
    # =====================================================
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return results_df