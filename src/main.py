import argparse

from data_quality import add_quality_flags
from prediction import run_prediction_models

from dataprocessing import (
    load_height_data,
    load_weight_data,
    merge_height_weight
)

from who_zscores import add_who_zscores

from features import create_growth_features

from validate_gmm import validate_gmm

from eda import (
    plot_height_by_age,
    plot_weight_by_age,
    plot_height_by_age_sex,
    plot_weight_by_age_sex,
    plot_individual_height_trajectories,
    plot_individual_weight_trajectories
)

from logistic import run_logistic_regression


# =============================================================
# consts
# =============================================================

MAX_AGE_DAYS = 24 * 30.4375

HEIGHT_BOYS = "database/reference/height_boys.xlsx"
HEIGHT_GIRLS = "database/reference/height_girls.xlsx"

WEIGHT_BOYS = "database/reference/weight_boys.xlsx"
WEIGHT_GIRLS = "database/reference/weight_girls.xlsx"


# =============================================================
# cli arguments
# =============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Infant growth analysis pipeline"
    )

    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run exploratory data analysis plots"
    )

    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run detailed data and growth diagnostics"
    )

    parser.add_argument(
        "--logistic",
        action="store_true",
        help="Run logistic regression using provisional GMM labels"
    )

    parser.add_argument(
    "--predict",
    action="store_true",
    help="Predict 24-month growth outcomes"
)

    return parser.parse_args()


# =============================================================
# loading and preparing data
# =============================================================

def load_growth_data():

    height = load_height_data()

    weight = load_weight_data()

    growth = merge_height_weight(
        height,
        weight
    )

    # ---------------------------------------------------------
    # project only cover birth to 2 year mark
    # ---------------------------------------------------------

    growth = growth[
        growth["AgeBabyDays"] <= MAX_AGE_DAYS
    ].copy()

    # ---------------------------------------------------------
    # add who z scores to growth
    # ---------------------------------------------------------

    growth = add_who_zscores(
        growth,
        HEIGHT_BOYS,
        HEIGHT_GIRLS,
        WEIGHT_BOYS,
        WEIGHT_GIRLS
    )

    growth = add_quality_flags(growth)

    print(
        f"Flagged observations: "
        f"{growth['quality_flag'].sum():,}"
    )

    return height, weight, growth


# =============================================================
# data summary
# =============================================================

def print_basic_summary(
    growth
):

    print("\n" + "=" * 60)
    print("0–24 MONTH DATA")
    print("=" * 60)

    print(
        f"Observations: {len(growth):,}"
    )

    print(
        f"Children: "
        f"{growth['anon_id'].nunique():,}"
    )

    print(
        f"Maximum age: "
        f"{growth['AgeBabyDays'].max() / 30.4375:.2f} months"
    )

    print("\n=== SEX DISTRIBUTION ===")

    print(
        growth["Sex"]
        .value_counts(dropna=False)
    )


# =============================================================
# exploratory data analysis
# =============================================================

def run_eda(growth):

    print("\n" + "=" * 60)
    print("GENERATING EDA")
    print("=" * 60)

    plot_height_by_age(
        growth
    )

    plot_weight_by_age(
        growth
    )

    plot_height_by_age_sex(
        growth
    )

    plot_weight_by_age_sex(
        growth
    )

    plot_individual_height_trajectories(
        growth
    )

    plot_individual_weight_trajectories(
        growth
    )


# =============================================================
# growth features
# =============================================================

def create_features(growth):

    print("\n" + "=" * 60)
    print("GROWTH FEATURES")
    print("=" * 60)

    features = create_growth_features(
        growth
    )

    print(
        f"Children with valid trajectories: "
        f"{len(features):,}"
    )

    print("\nFeature columns:")

    print(
        features.columns.tolist()
    )

    print("\nFeature missing values:")

    print(
        features.isna().sum()
    )

    return features


# =============================================================
# gmm provisional labels 
# =============================================================

def create_gmm_labels(features):

    print("\n" + "=" * 60)
    print("GMM PROVISIONAL LABELS")
    print("=" * 60)

    (
        features_with_groups,
        gmm,
        gmm_scaler,
        gmm_feature_columns
    ) = validate_gmm(
        features,
        n_components=3
    )

    print("\nGrowth group counts:")

    print(
        features_with_groups[
            "growth_group"
        ]
        .value_counts()
        .sort_index()
    )

    print("\nSample provisional labels:")

    print(
        features_with_groups[
            [
                "anon_id",
                "growth_group",
                "cluster_confidence"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    return (
        features_with_groups,
        gmm,
        gmm_scaler,
        gmm_feature_columns
    )


def run_diagnostics(
    height,
    weight,
    growth,
    features
):

    print("\n" + "=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)

    # ---------------------------------------------------------
    # quality of the data reporting
    # ---------------------------------------------------------

    print("\n=== DATA QUALITY ===")

    print(
        f"Height records: "
        f"{len(height):,}"
    )

    print(
        f"Weight records: "
        f"{len(weight):,}"
    )

    print(
        f"Merged records: "
        f"{len(growth):,}"
    )

    print(
        f"Children: "
        f"{growth['anon_id'].nunique():,}"
    )

    print("\nMissing values:")

    print(
        growth.isna().sum()
    )

    # ---------------------------------------------------------
    # summary of z score
    # ---------------------------------------------------------

    print("\n=== HEIGHT Z-SCORES ===")

    print(
        growth["height_z"]
        .describe()
    )

    print("\n=== WEIGHT Z-SCORES ===")

    print(
        growth["weight_z"]
        .describe()
    )

    # ---------------------------------------------------------
    # extreme z score
    # ---------------------------------------------------------

    print("\n=== EXTREME HEIGHT Z-SCORES ===")

    print("\nLowest:")

    print(
        growth[
            [
                "anon_id",
                "Sex",
                "AgeBabyDays",
                "Jit_Heightcm",
                "height_z"
            ]
        ]
        .sort_values("height_z")
        .head(10)
        .to_string(index=False)
    )

    print("\nHighest:")

    print(
        growth[
            [
                "anon_id",
                "Sex",
                "AgeBabyDays",
                "Jit_Heightcm",
                "height_z"
            ]
        ]
        .sort_values(
            "height_z",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # extreme z score
    # ---------------------------------------------------------

    print("\n=== EXTREME WEIGHT Z-SCORES ===")

    print("\nLowest:")

    print(
        growth[
            [
                "anon_id",
                "Sex",
                "AgeBabyDays",
                "jit_weightkg",
                "weight_z"
            ]
        ]
        .sort_values("weight_z")
        .head(10)
        .to_string(index=False)
    )

    print("\nHighest:")

    print(
        growth[
            [
                "anon_id",
                "Sex",
                "AgeBabyDays",
                "jit_weightkg",
                "weight_z"
            ]
        ]
        .sort_values(
            "weight_z",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # sumamary of trajectory
    # ---------------------------------------------------------

    print("\n=== TRAJECTORY SUMMARY ===")

    print(
        f"Children with valid trajectories: "
        f"{len(features):,}"
    )

    print("\nMeasurements per child:")

    print(
        features[
            "num_measurements"
        ]
        .describe()
    )

    print("\nFollow-up duration:")

    print(
        features[
            "duration_days"
        ]
        .describe()
    )


# =============================================================
# logistic regression subroutine
# =============================================================

def run_logistic(
    features_with_groups
):

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)

    results = run_logistic_regression(
        features_with_groups
    )

    return results

def main():

    args = parse_arguments()

    (
        height,
        weight,
        growth
    ) = load_growth_data()

    # ---------------------------------------------------------
    # prints growth summary 
    # ---------------------------------------------------------

    print_basic_summary(
        growth
    )

    # ---------------------------------------------------------
    # if diagnostics argument is passed via cli
    # ---------------------------------------------------------

    if args.eda:

        run_eda(
            growth
        )

    # ---------------------------------------------------------
    # growth features
    # ---------------------------------------------------------

    features = create_features(
        growth
    )

    # ---------------------------------------------------------
    # if diagnostics argument is passed via cli
    # ---------------------------------------------------------

    if args.diagnostics:

        run_diagnostics(
            height,
            weight,
            growth,
            features
        )

    # ---------------------------------------------------------
    # using the labels made by gmm as logistic regression target
    # ---------------------------------------------------------

    (
        features_with_groups,
        gmm,
        gmm_scaler,
        gmm_feature_columns
    ) = create_gmm_labels(
        features
    )

    # ---------------------------------------------------------
    # if logistic regression cli argument is used
    # ---------------------------------------------------------

    if args.logistic:

        run_logistic(
            features_with_groups
        )

    if args.predict:

        run_prediction_models(
            features
        )
# =============================================================
# start point 
# =============================================================

if __name__ == "__main__":
    main()