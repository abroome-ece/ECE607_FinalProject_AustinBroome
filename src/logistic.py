import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

LOGISTIC_FEATURES = [

    "start_height_z",
    "end_height_z",
    "height_z_change",
    "height_z_rate",

    "start_weight_z",
    "end_weight_z",
    "weight_z_change",
    "weight_z_rate"

]


# =============================================================
# logistic regression training with cross fold validation
# =============================================================

def run_logistic_regression(
    features_with_groups,
    n_splits=5,
    random_state=42
):

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)

    print(
        f"\nUsing {n_splits}-fold stratified cross-validation."
    )

    data = features_with_groups.copy()

    # ---------------------------------------------------------
    # required columns
    # ---------------------------------------------------------

    required_columns = (
        LOGISTIC_FEATURES
        + [
            "anon_id",
            "growth_group"
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + str(missing_columns)
        )


    model_data = data[
        required_columns
    ].copy()

    # ---------------------------------------------------------
    # remove row without labels
    # ---------------------------------------------------------

    model_data = model_data.dropna(
        subset=["growth_group"]
    ).copy()

    # ---------------------------------------------------------
    # features and labels
    # ---------------------------------------------------------

    X = model_data[
        LOGISTIC_FEATURES
    ].copy()

    y = model_data[
        "growth_group"
    ].astype(int).copy()

    child_ids = model_data[
        "anon_id"
    ].copy()

    print(
        f"\nChildren used: "
        f"{len(model_data):,}"
    )

    print("\nClass distribution:")

    class_distribution = (
        y.value_counts()
        .sort_index()
    )

    class_percent = (
        y.value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    for group in class_distribution.index:

        print(
            f"Group {group}: "
            f"{class_distribution[group]:,} "
            f"({class_percent[group]:.2f}%)"
        )

    # ---------------------------------------------------------
    # one child per row
    # ---------------------------------------------------------

    duplicate_children = (
        child_ids.duplicated()
        .sum()
    )

    if duplicate_children > 0:

        raise ValueError(
            "Duplicate children detected in "
            "logistic regression dataset: "
            f"{duplicate_children}"
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )


    fold_results = []

    y_true_oof = np.empty(
        len(y),
        dtype=int
    )

    y_pred_oof = np.empty(
        len(y),
        dtype=int
    )

    # ---------------------------------------------------------
    # loops for cross validation
    # ---------------------------------------------------------

    for fold, (
        train_indices,
        validation_indices
    ) in enumerate(
        cv.split(X, y),
        start=1
    ):

        print("\n" + "-" * 60)
        print(
            f"FOLD {fold}/{n_splits}"
        )
        print("-" * 60)

        # -----------------------------------------------------
        # split data between test and training 
        # -----------------------------------------------------

        X_train = X.iloc[
            train_indices
        ].copy()

        X_validation = X.iloc[
            validation_indices
        ].copy()

        y_train = y.iloc[
            train_indices
        ].copy()

        y_validation = y.iloc[
            validation_indices
        ].copy()

        train_ids = child_ids.iloc[
            train_indices
        ]

        validation_ids = child_ids.iloc[
            validation_indices
        ]

        # -----------------------------------------------------
        # check for overlap
        # -----------------------------------------------------

        overlap = set(train_ids).intersection(
            set(validation_ids)
        )

        if overlap:

            raise ValueError(
                "Child leakage detected in fold "
                f"{fold}: {len(overlap)} "
                "children appear in both sets."
            )

        print(
            f"Training children: "
            f"{len(X_train):,}"
        )

        print(
            f"Validation children: "
            f"{len(X_validation):,}"
        )

        # imputatation

        imputer = SimpleImputer(
            strategy="median"
        )

        X_train_imputed = imputer.fit_transform(
            X_train
        )

        X_validation_imputed = imputer.transform(
            X_validation
        )

        # scaling
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train_imputed
        )

        X_validation_scaled = scaler.transform(
            X_validation_imputed
        )

        # -----------------------------------------------------
        # logistic regression model
        # -----------------------------------------------------

        model = LogisticRegression(
            max_iter=2000,
            random_state=random_state
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        # -----------------------------------------------------
        # predictions
        # -----------------------------------------------------

        y_pred = model.predict(
            X_validation_scaled
        )


        y_true_oof[
            validation_indices
        ] = y_validation.to_numpy()

        y_pred_oof[
            validation_indices
        ] = y_pred

        accuracy = accuracy_score(
            y_validation,
            y_pred
        )

        precision = precision_score(
            y_validation,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            y_pred,
            average="macro",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_validation,
            y_pred,
            average="macro",
            zero_division=0
        )

        fold_results.append(
            {
                "fold": fold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "macro_f1": macro_f1
            }
        )

        print(
            f"Accuracy:  {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall:    {recall:.4f}"
        )

        print(
            f"Macro-F1:  {macro_f1:.4f}"
        )

    # =========================================================
    # cross-validation results
    # =========================================================

    results_df = pd.DataFrame(
        fold_results
    )

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ---------------------------------------------------------
    # mean and SD
    # ---------------------------------------------------------

    print("\n=== MEAN and STANDARD DEVIATION ===")

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "macro_f1"
    ]:

        mean = results_df[
            metric
        ].mean()

        std = results_df[
            metric
        ].std()

        print(
            f"{metric.capitalize():<10}: "
            f"{mean:.4f} ± {std:.4f}"
        )

    # =========================================================
    # out-of-fold  report
    # =========================================================

    print("\n" + "=" * 60)
    print("OUT-OF-FOLD CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_true_oof,
            y_pred_oof,
            zero_division=0
        )
    )

    # =========================================================
    # out of fold confusion matrix 
    # =========================================================

    print("\n" + "=" * 60)
    print("OUT-OF-FOLD CONFUSION MATRIX")
    print("=" * 60)

    cm = confusion_matrix(
        y_true_oof,
        y_pred_oof
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual 0",
            "Actual 1",
            "Actual 2"
        ],
        columns=[
            "Predicted 0",
            "Predicted 1",
            "Predicted 2"
        ]
    )

    print(
        cm_df
    )

    # =========================================================
    # out of fold accuracy
    # =========================================================

    oof_accuracy = accuracy_score(
        y_true_oof,
        y_pred_oof
    )

    oof_precision = precision_score(
        y_true_oof,
        y_pred_oof,
        average="macro",
        zero_division=0
    )

    oof_recall = recall_score(
        y_true_oof,
        y_pred_oof,
        average="macro",
        zero_division=0
    )

    oof_macro_f1 = f1_score(
        y_true_oof,
        y_pred_oof,
        average="macro",
        zero_division=0
    )

    print("\n=== OVERALL OUT-OF-FOLD METRICS ===")

    print(
        f"Accuracy:  {oof_accuracy:.4f}"
    )

    print(
        f"Precision: {oof_precision:.4f}"
    )

    print(
        f"Recall:    {oof_recall:.4f}"
    )

    print(
        f"Macro-F1:  {oof_macro_f1:.4f}"
    )

    return {
        "fold_results": results_df,

        "mean_accuracy":
            results_df["accuracy"].mean(),

        "std_accuracy":
            results_df["accuracy"].std(),

        "mean_precision":
            results_df["precision"].mean(),

        "std_precision":
            results_df["precision"].std(),

        "mean_recall":
            results_df["recall"].mean(),

        "std_recall":
            results_df["recall"].std(),

        "mean_macro_f1":
            results_df["macro_f1"].mean(),

        "std_macro_f1":
            results_df["macro_f1"].std(),

        "oof_accuracy":
            oof_accuracy,

        "oof_precision":
            oof_precision,

        "oof_recall":
            oof_recall,

        "oof_macro_f1":
            oof_macro_f1,

        "y_true_oof":
            y_true_oof,

        "y_pred_oof":
            y_pred_oof,

        "confusion_matrix":
            cm,

        "feature_names":
            LOGISTIC_FEATURES
    }