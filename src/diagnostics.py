import pandas as pd


def print_growth_diagnostics(growth):

    print("\n" + "=" * 60)
    print("GROWTH DATA DIAGNOSTICS")
    print("=" * 60)

    # ---------------------------------------------------------
    # basic data
    # ---------------------------------------------------------

    print("\n=== DATASET SIZE ===")

    print(f"Observations: {len(growth):,}")
    print(f"Children: {growth['anon_id'].nunique():,}")

    print(
        f"Maximum age: "
        f"{growth['AgeBabyDays'].max() / 30.4375:.2f} months"
    )

    # ---------------------------------------------------------
    # sex distribution
    # ---------------------------------------------------------

    print("\n=== SEX DISTRIBUTION ===")

    print(
        growth["Sex"]
        .value_counts(dropna=False)
    )

    # ---------------------------------------------------------
    # state missing values
    # ---------------------------------------------------------

    print("\n=== MISSING VALUES ===")

    print(
        growth.isna().sum()
    )

    # ---------------------------------------------------------
    # check duplicates
    # ---------------------------------------------------------

    print("\n=== DUPLICATE CHECK ===")

    duplicate_child_age = (
        growth
        .duplicated(
            subset=["anon_id", "AgeBabyDays"]
        )
        .sum()
    )

    print(
        f"Duplicate child-age records: "
        f"{duplicate_child_age}"
    )

    # ---------------------------------------------------------
    # measurement ranges
    # ---------------------------------------------------------

    print("\n=== MEASUREMENT RANGES ===")

    print(
        f"Height: "
        f"{growth['Jit_Heightcm'].min():.2f} - "
        f"{growth['Jit_Heightcm'].max():.2f} cm"
    )

    print(
        f"Weight: "
        f"{growth['jit_weightkg'].min():.2f} - "
        f"{growth['jit_weightkg'].max():.2f} kg"
    )


    measurements_per_child = (
        growth
        .groupby("anon_id")
        .size()
    )

    print("\n=== MEASUREMENTS PER CHILD ===")

    print(
        f"Minimum: "
        f"{measurements_per_child.min()}"
    )

    print(
        f"Maximum: "
        f"{measurements_per_child.max()}"
    )

    print(
        f"Mean: "
        f"{measurements_per_child.mean():.2f}"
    )

    print(
        f"Median: "
        f"{measurements_per_child.median():.0f}"
    )

    print("\nDistribution:")

    print(
        measurements_per_child
        .value_counts()
        .sort_index()
    )
    
    child_age_ranges = (
        growth
        .groupby("anon_id")["AgeBabyDays"]
        .agg(
            first_age="min",
            last_age="max"
        )
    )

    child_age_ranges["duration"] = (
        child_age_ranges["last_age"]
        - child_age_ranges["first_age"]
    )

    print("\n=== LONGITUDINAL AGE COVERAGE ===")

    print(
        f"Mean follow-up: "
        f"{child_age_ranges['duration'].mean():.1f} days"
    )

    print(
        f"Median follow-up: "
        f"{child_age_ranges['duration'].median():.1f} days"
    )

    print(
        f"Minimum follow-up: "
        f"{child_age_ranges['duration'].min():.0f} days"
    )

    print(
        f"Maximum follow-up: "
        f"{child_age_ranges['duration'].max():.0f} days"
    )

    for days in [90, 180, 365, 540, 730]:

        count = (
            child_age_ranges["duration"]
            >= days
        ).sum()

        print(
            f"Children with measurements spanning "
            f"at least {days} days: {count:,}"
        )


def print_quality_diagnostics(growth):

    print("\n" + "=" * 60)
    print("DATA QUALITY FLAGS")
    print("=" * 60)

    if "height_quality_flag" in growth.columns:

        print(
            "\nHeight flags:",
            growth["height_quality_flag"].sum()
        )

    if "weight_quality_flag" in growth.columns:

        print(
            "Weight flags:",
            growth["weight_quality_flag"].sum()
        )

    if "quality_flag" in growth.columns:

        print(
            "Total flagged observations:",
            growth["quality_flag"].sum()
        )

    # ---------------------------------------------------------
    # flags for height
    # ---------------------------------------------------------

    if "height_quality_flag" in growth.columns:

        height_flags = growth[
            growth["height_quality_flag"]
        ].copy()

        if len(height_flags) > 0:

            columns = [
                "anon_id",
                "Sex",
                "AgeBabyDays",
                "Jit_Heightcm"
            ]

            optional = [
                "previous_height",
                "height_change_from_previous"
            ]

            columns += [
                c for c in optional
                if c in height_flags.columns
            ]

            print("\n=== FLAGGED HEIGHT MEASUREMENTS ===")

            print(
                height_flags[columns]
                .to_string(index=False)
            )

    # ---------------------------------------------------------
    # flags for weight
    # ---------------------------------------------------------

    if "weight_quality_flag" in growth.columns:

        weight_flags = growth[
            growth["weight_quality_flag"]
        ].copy()

        if len(weight_flags) > 0:

            columns = [
                "anon_id",
                "Sex",
                "AgeBabyDays",
                "jit_weightkg"
            ]

            optional = [
                "previous_weight",
                "weight_change_from_previous"
            ]

            columns += [
                c for c in optional
                if c in weight_flags.columns
            ]

            print("\n=== FLAGGED WEIGHT MEASUREMENTS ===")

            print(
                weight_flags[columns]
                .to_string(index=False)
            )


def print_zscore_diagnostics(growth):
    """
    Print WHO z-score diagnostics and extreme observations.
    """

    print("\n" + "=" * 60)
    print("WHO Z-SCORE DIAGNOSTICS")
    print("=" * 60)

    # ---------------------------------------------------------
    # summaries
    # ---------------------------------------------------------

    print("\nHeight Z-score summary:")

    print(
        growth["height_z"]
        .describe()
    )

    print("\nWeight Z-score summary:")

    print(
        growth["weight_z"]
        .describe()
    )

    print("\nMissing Z-scores:")

    print(
        growth[
            ["height_z", "weight_z"]
        ]
        .isna()
        .sum()
    )

    # ---------------------------------------------------------
    # extreme height
    # ---------------------------------------------------------

    columns = [
        "anon_id",
        "Sex",
        "AgeBabyDays",
        "Jit_Heightcm",
        "height_z"
    ]

    print("\n=== EXTREME HEIGHT Z-SCORES ===")

    print("\nLowest:")

    print(
        growth[columns]
        .sort_values("height_z")
        .head(10)
        .to_string(index=False)
    )

    print("\nHighest:")

    print(
        growth[columns]
        .sort_values(
            "height_z",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # extreme weight
    # ---------------------------------------------------------

    columns = [
        "anon_id",
        "Sex",
        "AgeBabyDays",
        "jit_weightkg",
        "weight_z"
    ]

    print("\n=== EXTREME WEIGHT Z-SCORES ===")

    print("\nLowest:")

    print(
        growth[columns]
        .sort_values("weight_z")
        .head(10)
        .to_string(index=False)
    )

    print("\nHighest:")

    print(
        growth[columns]
        .sort_values(
            "weight_z",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )


def run_all_diagnostics(growth):

    print_growth_diagnostics(growth)

    print_quality_diagnostics(growth)

    print_zscore_diagnostics(growth)