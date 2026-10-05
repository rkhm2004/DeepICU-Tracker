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

ITEM_IDS = {
    "heart_rate": 220045,
    "sbp": 220179,
    "wbc": 51301,
}

FEATURES = ["heart_rate", "sbp", "wbc"]
WINDOW_HOURS = 48
RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def assign_stay_splits(stay_ids, seed=RANDOM_SEED):
    """Assign complete ICU stays to train/validation/test without patient-row leakage."""
    unique_stays = np.array(sorted(pd.Series(stay_ids).dropna().unique()))
    if len(unique_stays) < 3:
        raise ValueError("At least 3 ICU stays are required for train/validation/test splits.")

    rng = np.random.default_rng(seed)
    shuffled = unique_stays.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(np.floor(n * TRAIN_FRACTION)))
    n_val = max(1, int(np.floor(n * VAL_FRACTION)))
    n_test = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        n_train -= 1

    split = {}
    split.update({stay_id: "train" for stay_id in shuffled[:n_train]})
    split.update(
        {stay_id: "validation" for stay_id in shuffled[n_train:n_train + n_val]}
    )
    split.update(
        {stay_id: "test" for stay_id in shuffled[n_train + n_val:]}
    )
    return split


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
    """Build patient-aware 48-hour trajectories with train-only preprocessing statistics.

    ICU stays, rather than individual hourly rows, are randomly assigned to
    train/validation/test. Missing-value medians and StandardScaler statistics
    are fitted on training stays only, then applied to validation/test stays.
    This prevents information from validation/test stays leaking into training.
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

    usable_stays = pivot["stay_id"].drop_duplicates()
    full_index = pd.MultiIndex.from_product(
        [usable_stays.tolist(), range(WINDOW_HOURS)],
        names=["stay_id", "hour"],
    ).to_frame(index=False)
    pivot = full_index.merge(pivot, on=["stay_id", "hour"], how="left")
    pivot = pivot.sort_values(["stay_id", "hour"]).reset_index(drop=True)

    split_map = assign_stay_splits(pivot["stay_id"])
    pivot["split"] = pivot["stay_id"].map(split_map)

    if pivot["split"].isna().any():
        raise RuntimeError("Some ICU stays were not assigned to a data split.")

    print("5. Imputing missing values using training-only statistics...")
    pivot[FEATURES] = pivot.groupby("stay_id")[FEATURES].transform(
        lambda group: group.ffill().bfill()
    )

    # Fit fallback medians on training stays only.
    train_mask = pivot["split"] == "train"
    train_medians = pivot.loc[train_mask, FEATURES].median()
    pivot[FEATURES] = pivot[FEATURES].fillna(train_medians)

    if pivot[FEATURES].isna().any().any():
        raise ValueError("Missing feature values remain after train-only imputation.")

    index_df = pivot[["stay_id", "hour", "split"] + FEATURES].copy()

    print("6. Scaling features using training stays only...")
    scaler = StandardScaler()
    scaler.fit(index_df.loc[index_df["split"] == "train", FEATURES].values)
    scaled_features = scaler.transform(index_df[FEATURES].values)

    tensor_data = torch.tensor(scaled_features, dtype=torch.float32)
    torch.save(tensor_data, PROCESSED_DIR / "mimic_tensor.pt")
    index_df.to_csv(PROCESSED_DIR / "mimic_index.csv", index=False)

    split_summary = (
        index_df.groupby("split")["stay_id"]
        .agg(["nunique", "count"])
        .rename(columns={"nunique": "icu_stays", "count": "hourly_records"})
        .reindex(["train", "validation", "test"])
    )
    split_summary.to_csv(PROCESSED_DIR / "split_summary.csv")

    preprocessing = {
        "features": FEATURES,
        "window_hours": WINDOW_HOURS,
        "random_seed": RANDOM_SEED,
        "split_fractions": {
            "train": TRAIN_FRACTION,
            "validation": VAL_FRACTION,
            "test": TEST_FRACTION,
        },
        "imputation_median": train_medians.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "fit_on_split": "train",
    }
    with open(PROCESSED_DIR / "preprocessing.json", "w", encoding="utf-8") as f:
        json.dump(preprocessing, f, indent=2)

    print(
        f"Success! Extracted {len(index_df):,} hourly records across "
        f"{index_df['stay_id'].nunique():,} ICU stays."
    )
    print("\n--- Stay-level split ---")
    print(split_summary.to_string())
    print(f"\nSaved tensor: {PROCESSED_DIR / 'mimic_tensor.pt'}")
    print(f"Saved trajectory index: {PROCESSED_DIR / 'mimic_index.csv'}")
    print(f"Saved preprocessing metadata: {PROCESSED_DIR / 'preprocessing.json'}")

    return tensor_data


if __name__ == "__main__":
    parse_mimic_data()
