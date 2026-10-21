import pandas as pd
import numpy as np


def load_who_reference(path):

    reference = pd.read_excel(path)

    required = ["Month", "L", "M", "S"]

    missing = [col for col in required if col not in reference.columns]

    if missing:
        raise ValueError(
            f"Missing required WHO columns: {missing}"
        )

    return reference[required].copy()


def calculate_lms_zscore(value, L, M, S):

    if pd.isna(value) or pd.isna(L) or pd.isna(M) or pd.isna(S):
        return np.nan

    if value <= 0 or M <= 0 or S <= 0:
        return np.nan

    if L == 0:
        return np.log(value / M) / S

    return ((value / M) ** L - 1) / (L * S)


def calculate_zscore(value, age_days, reference):


    if pd.isna(value) or pd.isna(age_days):
        return np.nan

    age_months = age_days / 30.4375

    min_month = reference["Month"].min()
    max_month = reference["Month"].max()

    if age_months < min_month or age_months > max_month:
        return np.nan

    lower_month = int(np.floor(age_months))
    upper_month = int(np.ceil(age_months))


    if lower_month == upper_month:

        row = reference[
            reference["Month"] == lower_month
        ].iloc[0]

        return calculate_lms_zscore(
            value,
            row["L"],
            row["M"],
            row["S"]
        )

    lower = reference[
        reference["Month"] == lower_month
    ].iloc[0]

    upper = reference[
        reference["Month"] == upper_month
    ].iloc[0]

    fraction = age_months - lower_month

    L = lower["L"] + fraction * (
        upper["L"] - lower["L"]
    )

    M = lower["M"] + fraction * (
        upper["M"] - lower["M"]
    )

    S = lower["S"] + fraction * (
        upper["S"] - lower["S"]
    )

    return calculate_lms_zscore(
        value,
        L,
        M,
        S
    )
    
def add_who_zscores(
    growth,
    height_boys_path,
    height_girls_path,
    weight_boys_path,
    weight_girls_path
):

    height_boys = load_who_reference(height_boys_path)
    height_girls = load_who_reference(height_girls_path)

    weight_boys = load_who_reference(weight_boys_path)
    weight_girls = load_who_reference(weight_girls_path)

    height_z = []
    weight_z = []

    for _, row in growth.iterrows():

        sex = row["Sex"]

        if sex == "Male":
            height_ref = height_boys
            weight_ref = weight_boys
        else:
            height_ref = height_girls
            weight_ref = weight_girls

        hz = calculate_zscore(
            row["Jit_Heightcm"],
            row["AgeBabyDays"],
            height_ref
        )

        wz = calculate_zscore(
            row["jit_weightkg"],
            row["AgeBabyDays"],
            weight_ref
        )

        height_z.append(hz)
        weight_z.append(wz)

    growth = growth.copy()

    growth["height_z"] = height_z
    growth["weight_z"] = weight_z

    return growth