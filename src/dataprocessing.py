from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "database" / "raw"

def load_height_data():
    path = RAW_DATA / "starr_source_height_data_jittered.txt"
    return pd.read_csv(path, sep="\t")

def load_weight_data():
    path = RAW_DATA / "starr_source_weight_data_jittered.txt"
    return pd.read_csv(path, sep="\t")

def merge_height_weight(height, weight):
    merged = height.merge(
        weight,
        on=["anon_id", "Sex", "AgeBabyDays"],
        how="inner"
    )
    return merged