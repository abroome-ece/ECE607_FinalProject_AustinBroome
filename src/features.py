import pandas as pd


def create_growth_features(growth):

    data = growth.copy()

    # =========================================================
    # remove the previously flagged data
    # =========================================================

    if "quality_flag" in data.columns:

        data = data[
            ~data["quality_flag"]
        ].copy()

    data = data.sort_values(
        ["anon_id", "AgeBabyDays"]
    )

    # =========================================================
    # these are the age windows used in the features in days
    # =========================================================

    # turn the months into days
    MONTH_6 = 6 * 30.4375
    MONTH_12 = 12 * 30.4375
    MONTH_18 = 18 * 30.4375
    MONTH_24 = 24 * 30.4375

    features = []

    # =========================================================
    # features for child
    # =========================================================

    for child_id, child_data in data.groupby(
        "anon_id"
    ):

        # -----------------------------------------------------
        # two valid measurements needed
        # -----------------------------------------------------

        if len(child_data) < 2:
            continue

        child_data = child_data.sort_values(
            "AgeBabyDays"
        )

        first = child_data.iloc[0]
        last = child_data.iloc[-1]

        duration = (
            last["AgeBabyDays"]
            - first["AgeBabyDays"]
        )

        if duration <= 0:
            continue

        # =====================================================
        # START / END Z-SCORE
        # =====================================================

        height_z_change = (
            last["height_z"]
            - first["height_z"]
        )

        height_z_rate = (
            height_z_change / duration
        )

        weight_z_change = (
            last["weight_z"]
            - first["weight_z"]
        )

        weight_z_rate = (
            weight_z_change / duration
        )

        # =====================================================
        # AGE-WINDOW FEATURES
        # =====================================================

        # -----------------------------------------------------
        # 0–6 MONTHS
        # -----------------------------------------------------

        window_0_6 = child_data[
            child_data["AgeBabyDays"] < MONTH_6
        ]

        # -----------------------------------------------------
        # 6–12 MONTHS
        # -----------------------------------------------------

        window_6_12 = child_data[
            (child_data["AgeBabyDays"] >= MONTH_6)
            &
            (child_data["AgeBabyDays"] < MONTH_12)
        ]

        # -----------------------------------------------------
        # 12–18 MONTHS
        # -----------------------------------------------------

        window_12_18 = child_data[
            (child_data["AgeBabyDays"] >= MONTH_12)
            &
            (child_data["AgeBabyDays"] < MONTH_18)
        ]

        # -----------------------------------------------------
        # 18–24 MONTHS
        # -----------------------------------------------------

        window_18_24 = child_data[
            (child_data["AgeBabyDays"] >= MONTH_18)
            &
            (child_data["AgeBabyDays"] <= MONTH_24)
        ]


        def mean_or_nan(
            dataframe,
            column
        ):

            if len(dataframe) == 0:
                return float("nan")

            return dataframe[column].mean()

        # =====================================================
        # MEAN
        # =====================================================

        height_0_6 = mean_or_nan(
            window_0_6,
            "height_z"
        )

        height_6_12 = mean_or_nan(
            window_6_12,
            "height_z"
        )

        height_12_18 = mean_or_nan(
            window_12_18,
            "height_z"
        )

        height_18_24 = mean_or_nan(
            window_18_24,
            "height_z"
        )

        weight_0_6 = mean_or_nan(
            window_0_6,
            "weight_z"
        )

        weight_6_12 = mean_or_nan(
            window_6_12,
            "weight_z"
        )

        weight_12_18 = mean_or_nan(
            window_12_18,
            "weight_z"
        )

        weight_18_24 = mean_or_nan(
            window_18_24,
            "weight_z"
        )

        features.append({

            # -------------------------------------------------
            # IDENTIFICATION
            # -------------------------------------------------

            "anon_id": child_id,

            "Sex": first["Sex"],

            # -------------------------------------------------
            # AGE / FOLLOW-UP
            # -------------------------------------------------

            "start_age": first["AgeBabyDays"],

            "end_age": last["AgeBabyDays"],

            "duration_days": duration,

            "num_measurements": len(
                child_data
            ),

            # -------------------------------------------------
            # OVERALL HEIGHT TRAJECTORY
            # -------------------------------------------------

            "start_height_z": first["height_z"],

            "end_height_z": last["height_z"],

            "height_z_change": height_z_change,

            "height_z_rate": height_z_rate,

            # -------------------------------------------------
            # OVERALL WEIGHT TRAJECTORY
            # -------------------------------------------------

            "start_weight_z": first["weight_z"],

            "end_weight_z": last["weight_z"],

            "weight_z_change": weight_z_change,

            "weight_z_rate": weight_z_rate,

            # -------------------------------------------------
            # HEIGHT BY AGE WINDOW
            # -------------------------------------------------

            "height_z_0_6": height_0_6,

            "height_z_6_12": height_6_12,

            "height_z_12_18": height_12_18,

            "height_z_18_24": height_18_24,

            # -------------------------------------------------
            # WEIGHT BY AGE WINDOW
            # -------------------------------------------------

            "weight_z_0_6": weight_0_6,

            "weight_z_6_12": weight_6_12,

            "weight_z_12_18": weight_12_18,

            "weight_z_18_24": weight_18_24
        })

    return pd.DataFrame(
        features
    )
