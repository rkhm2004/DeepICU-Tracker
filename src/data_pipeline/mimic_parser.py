import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path("data/raw/mimic-iv-clinical-database-demo-2.2")
ICU_DIR = BASE_DIR / "icu"
HOSP_DIR = BASE_DIR / "hosp"
PROCESSED_DIR = Path("data/processed")

# MIMIC-IV item IDs for the three features used by the current prototype.
ITEM_IDS = {
    "heart_rate": 220045,
    "sbp": 220179,
    "wbc": 51301,
}

FEATURES = ["heart_rate", "sbp", "wbc"]
WINDOW_HOURS = 48


def _assign_labs_to_icu_stays(labs: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    """Attach each lab observation to the ICU stay containing its timestamp."""
    labs = labs.merge(
        stays[["subject_id", "stay_id", "intime", "outtime"]],
        on="subject_id",
        how="inner",
    )
    in_stay = (labs["charttime"] >= labs["intime"]) & (labs["charttime"] <= labs["outtime"])
    return labs.loc[in_stay, ["stay_id", "charttime", "itemid", "valuenum"]]


def parse_mimic_data():
    """Build a uniformly sampled, patient-aware 48-hour ICU tensor.

    The tensor rows are kept in exactly the same order as
    data/processed/mimic_index.csv so downstream CTMC estimation can never
    create a transition across two different ICU stays.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Loading ICU stays...")
    stays = pd.read_csv(
        ICU_DIR / "icustays.csv",
        usecols=["subject_id", "stay_id", "intime", "outtime"],
    )
    stays["intime"] = pd.to_datetime(stays["intime"])
    stays["outtime"] = pd.to_datetime(stays["outtime"])

    print("2. Processing vitals (chartevents)...")
    vitals = pd.read_csv(
        ICU_DIR / "chartevents.csv",
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
    )
    vitals = vitals[vitals["itemid"].isin([ITEM_IDS["heart_rate"], ITEM_IDS["sbp"]])].copy()
    vitals["charttime"] = pd.to_datetime(vitals["charttime"])

    print("3. Processing WBC labs (labevents)...")
    labs = pd.read_csv(
        HOSP_DIR / "labevents.csv",
        usecols=["subject_id", "itemid", "charttime", "valuenum"],
    )
    labs = labs[labs["itemid"] == ITEM_IDS["wbc"]].copy()
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    labs = _assign_labs_to_icu_stays(labs, stays)

    all_events = pd.concat([vitals, labs], ignore_index=True)
    all_events = all_events.merge(
        stays[["stay_id", "intime", "outtime"]], on="stay_id", how="inner"
    )

    all_events["hour"] = (
        (all_events["charttime"] - all_events["intime"]).dt.total_seconds() // 3600
    ).astype(int)
    all_events = all_events[
        (all_events["hour"] >= 0) & (all_events["hour"] < WINDOW_HOURS)
    ]

    print("4. Aggregating observations into hourly windows...")
    pivot = (
        all_events.pivot_table(
            index=["stay_id", "hour"],
            columns="itemid",
            values="valuenum",
            aggfunc="mean",
        )
        .rename(
            columns={
                ITEM_IDS["heart_rate"]: "heart_rate",
                ITEM_IDS["sbp"]: "sbp",
                ITEM_IDS["wbc"]: "wbc",
            }
        )
        .reset_index()
    )

    # Keep only stays with at least one usable observation. Then construct a
    # complete 0..47 hour grid so a one-hour transition really means one hour.
    usable_stays = pivot["stay_id"].drop_duplicates()
    full_index = pd.MultiIndex.from_product(
        [usable_stays.tolist(), range(WINDOW_HOURS)],
        names=["stay_id", "hour"],
    ).to_frame(index=False)
    pivot = full_index.merge(pivot, on=["stay_id", "hour"], how="left")
    pivot = pivot.sort_values(["stay_id", "hour"]).reset_index(drop=True)

    print("5. Imputing missing values and scaling...")
    pivot[FEATURES] = pivot.groupby("stay_id", group_keys=False)[FEATURES].ffill().bfill()
    pivot[FEATURES] = pivot[FEATURES].fillna(pivot[FEATURES].median(numeric_only=True))
    if pivot[FEATURES].isna().any().any():
        raise ValueError("Missing feature values remain after imputation.")

    # Save the raw, patient-aware index before scaling. This is the contract
    # between data preprocessing, VAE inference, and CTMC trajectory fitting.
    index_df = pivot[["stay_id", "hour"] + FEATURES].copy()
    index_df.to_csv(PROCESSED_DIR / "mimic_index.csv", index=False)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(index_df[FEATURES].values)
    tensor_data = torch.tensor(scaled_features, dtype=torch.float32)
    torch.save(tensor_data, PROCESSED_DIR / "mimic_tensor.pt")

    preprocessing = {
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "window_hours": WINDOW_HOURS,
    }
    with open(PROCESSED_DIR / "preprocessing.json", "w", encoding="utf-8") as f:
        json.dump(preprocessing, f, indent=2)

    print(
        f"Success! Extracted {len(index_df):,} hourly records across "
        f"{index_df['stay_id'].nunique():,} ICU stays."
    )
    print(f"Saved tensor: {PROCESSED_DIR / 'mimic_tensor.pt'}")
    print(f"Saved trajectory index: {PROCESSED_DIR / 'mimic_index.csv'}")

    return tensor_data


if __name__ == "__main__":
    parse_mimic_data()
