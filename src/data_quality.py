import pandas as pd

def add_quality_flags(
    growth,
    height_drop_threshold=5.0,
    weight_change_threshold=5.0,
    zscore_limit=5.0
):

    data = growth.sort_values(
        ["anon_id", "AgeBabyDays"]
    ).copy()

    # ---------------------------------------------------------
    # previous measurement
    # ---------------------------------------------------------

    data["previous_height"] = (
        data.groupby("anon_id")["Jit_Heightcm"]
        .shift(1)
    )

    data["previous_weight"] = (
        data.groupby("anon_id")["jit_weightkg"]
        .shift(1)
    )

    # ---------------------------------------------------------
    # change from previous measurement
    # ---------------------------------------------------------

    data["height_change_from_previous"] = (
        data["Jit_Heightcm"]
        - data["previous_height"]
    )

    data["weight_change_from_previous"] = (
        data["jit_weightkg"]
        - data["previous_weight"]
    )

    # ---------------------------------------------------------
    # longitudinal quality checks
    # ---------------------------------------------------------

    data["height_quality_flag"] = (
        data["height_change_from_previous"]
        < -height_drop_threshold
    )

    data["weight_quality_flag"] = (
        data["weight_change_from_previous"].abs()
        > weight_change_threshold
    )

    # ---------------------------------------------------------
    # biological plausibility checks
    # ---------------------------------------------------------

    data["height_z_flag"] = (
        data["height_z"].abs()
        > zscore_limit
    )

    data["weight_z_flag"] = (
        data["weight_z"].abs()
        > zscore_limit
    )

    # ---------------------------------------------------------
    # first measurements should not be flagged
    # ---------------------------------------------------------

    data["height_quality_flag"] = (
        data["height_quality_flag"]
        .fillna(False)
    )

    data["weight_quality_flag"] = (
        data["weight_quality_flag"]
        .fillna(False)
    )

    # ---------------------------------------------------------
    # combined flag
    # ---------------------------------------------------------

    data["quality_flag"] = (
        data["height_quality_flag"]
        | data["weight_quality_flag"]
        | data["height_z_flag"]
        | data["weight_z_flag"]
    )

    return data

def find_height_reversals(
    growth,
    drop_threshold=5.0
):
    """
    this checks for the height reversing
    as mentioned in the checkpoint there
    are 6 examples of this
    """

    data = growth.sort_values(
        ["anon_id", "AgeBabyDays"]
    ).copy()

    data["previous_height"] = (
        data.groupby("anon_id")["Jit_Heightcm"]
        .shift(1)
    )

    data["height_change"] = (
        data["Jit_Heightcm"]
        - data["previous_height"]
    )

    reversals = data[
        data["height_change"] < -drop_threshold
    ].copy()

    return reversals