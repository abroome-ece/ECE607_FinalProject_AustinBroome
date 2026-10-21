def print_data_summary(height, weight, growth):
    print("=" * 60)
    print("INFANT GROWTH DATA - INITIAL DATA CHECK")
    print("=" * 60)

    print("\n=== RAW DATA ===")

    print("\nHeight:")
    print(f"  Records:        {len(height):,}")
    print(f"  Unique children: {height['anon_id'].nunique():,}")
    print(
        f"  Age range:      "
        f"{height['AgeBabyDays'].min()} - "
        f"{height['AgeBabyDays'].max()} days"
    )

    print("\nWeight:")
    print(f"  Records:        {len(weight):,}")
    print(f"  Unique children: {weight['anon_id'].nunique():,}")
    print(
        f"  Age range:      "
        f"{weight['AgeBabyDays'].min()} - "
        f"{weight['AgeBabyDays'].max()} days"
    )

    print("\n=== MERGED DATA ===")

    print(f"  Matched records: {len(growth):,}")
    print(f"  Unique children: {growth['anon_id'].nunique():,}")

    print(
        f"  Age range:       "
        f"{growth['AgeBabyDays'].min()} - "
        f"{growth['AgeBabyDays'].max()} days"
    )

    print("\n=== SEX DISTRIBUTION ===")

    print(
        growth.groupby("Sex")["anon_id"]
        .nunique()
        .to_string()
    )

    print("\n=== MISSING VALUES ===")

    print(growth.isna().sum().to_string())

    print("\n=== DUPLICATE CHECK ===")

    height_duplicates = height.duplicated(
        subset=["anon_id", "AgeBabyDays"]
    ).sum()

    weight_duplicates = weight.duplicated(
        subset=["anon_id", "AgeBabyDays"]
    ).sum()

    merged_duplicates = growth.duplicated(
        subset=["anon_id", "AgeBabyDays"]
    ).sum()

    print(f"  Height duplicate records: {height_duplicates:,}")
    print(f"  Weight duplicate records: {weight_duplicates:,}")
    print(
        f"  Duplicate child-age records after merge: "
        f"{merged_duplicates:,}"
    )

    print("\n=== MEASUREMENT RANGES ===")

    print(
        f"  Height: "
        f"{growth['Jit_Heightcm'].min():.2f} - "
        f"{growth['Jit_Heightcm'].max():.2f} cm"
    )

    print(
        f"  Weight: "
        f"{growth['jit_weightkg'].min():.2f} - "
        f"{growth['jit_weightkg'].max():.2f} kg"
    )

    measurements_per_child = growth.groupby("anon_id").size()

    print("\n=== MEASUREMENTS PER CHILD ===")
    print(f"  Minimum: {measurements_per_child.min()}")
    print(f"  Maximum: {measurements_per_child.max()}")
    print(f"  Mean:    {measurements_per_child.mean():.2f}")
    print(f"  Median:  {measurements_per_child.median():.0f}")

    print("\n=== SAMPLE RECORDS ===")

    print(
        growth[
            [
                "anon_id",
                "Sex",
                "AgeBabyDays",
                "Jit_Heightcm",
                "jit_weightkg"
            ]
        ].head(10).to_string(index=False)
    )

    print("\n=== MEASUREMENTS PER CHILD DISTRIBUTION ===")

    measurement_counts = growth.groupby("anon_id").size()

    print(
        measurement_counts.value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nChildren with at least 2 measurements:",
        (measurement_counts >= 2).sum()
    )

    print(
        "Children with at least 3 measurements:",
        (measurement_counts >= 3).sum()
    )

    print(
        "Children with at least 5 measurements:",
        (measurement_counts >= 5).sum()
    )

    print(
        "Children with at least 8 measurements:",
        (measurement_counts >= 8).sum()
    )

    print("\n=== LONGITUDINAL AGE COVERAGE ===")

    child_age = growth.groupby("anon_id")["AgeBabyDays"].agg(
        first_age="min",
        last_age="max"
    )

    child_age["duration_days"] = (
        child_age["last_age"] - child_age["first_age"]
    )

    print(
        f"  Mean follow-up:   "
        f"{child_age['duration_days'].mean():.1f} days"
    )

    print(
        f"  Median follow-up: "
        f"{child_age['duration_days'].median():.1f} days"
    )

    print(
        f"  Minimum follow-up: "
        f"{child_age['duration_days'].min():.0f} days"
    )

    print(
        f"  Maximum follow-up: "
        f"{child_age['duration_days'].max():.0f} days"
    )

    print("\nChildren with measurements spanning at least:")

    for days in [90, 180, 365, 540, 730]:
        count = (child_age["duration_days"] >= days).sum()

        print(
            f"  {days:3} days: "
            f"{count:,} children"
        )

    print("\n" + "=" * 60)