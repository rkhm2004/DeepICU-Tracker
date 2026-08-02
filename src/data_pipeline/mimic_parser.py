import pandas as pd
import numpy as np
from pathlib import Path
import torch
from sklearn.preprocessing import StandardScaler

# Define the exact paths based on your check.py output
BASE_DIR = Path("data/raw/mimic-iv-clinical-database-demo-2.2")
ICU_DIR = BASE_DIR / "icu"
HOSP_DIR = BASE_DIR / "hosp"

# MIMIC-IV Item IDs for our 3 core features
ITEM_IDS = {
    'heart_rate': 220045,
    'sbp': 220179, # Non Invasive Blood Pressure systolic
    'wbc': 51301   # White Blood Cell count
}

def parse_mimic_data():
    print("1. Loading ICU stays...")
    stays = pd.read_csv(ICU_DIR / "icustays.csv", usecols=['subject_id', 'stay_id', 'intime', 'outtime'])
    stays['intime'] = pd.to_datetime(stays['intime'])
    
    print("2. Processing Vitals (chartevents)...")
    # In the full dataset this is chunked, but demo fits in memory
    vitals = pd.read_csv(ICU_DIR / "chartevents.csv", usecols=['stay_id', 'itemid', 'charttime', 'valuenum'])
    vitals = vitals[vitals['itemid'].isin([ITEM_IDS['heart_rate'], ITEM_IDS['sbp']])]
    vitals['charttime'] = pd.to_datetime(vitals['charttime'])
    
    print("3. Processing Labs (labevents)...")
    labs = pd.read_csv(HOSP_DIR / "labevents.csv", usecols=['subject_id', 'itemid', 'charttime', 'valuenum'])
    labs = labs[labs['itemid'] == ITEM_IDS['wbc']]
    labs['charttime'] = pd.to_datetime(labs['charttime'])
    
    # Merge labs with stay_id based on subject_id and time (simplified for demo)
    labs = pd.merge(labs, stays[['subject_id', 'stay_id']], on='subject_id', how='inner')
    
    # Combine vitals and labs
    all_events = pd.concat([vitals, labs], ignore_index=True)
    
    print("4. Aligning into 1-hour time windows...")
    # Merge with admission time to calculate relative hours
    all_events = pd.merge(all_events, stays[['stay_id', 'intime']], on='stay_id')
    all_events['hour'] = ((all_events['charttime'] - all_events['intime']).dt.total_seconds() // 3600).astype(int)
    
    # Filter out data before admission or past 48 hours to keep tensors uniform
    all_events = all_events[(all_events['hour'] >= 0) & (all_events['hour'] < 48)]
    
    # Pivot so each feature is a column
    pivot_df = all_events.pivot_table(index=['stay_id', 'hour'], columns='itemid', values='valuenum', aggfunc='mean')
    pivot_df.columns = ['wbc', 'heart_rate', 'sbp']
    pivot_df = pivot_df.reset_index()
    
    print("5. Imputing missing values and scaling...")
    # Forward fill missing hours for each patient, then fill remaining with medians
    pivot_df = pivot_df.groupby('stay_id').ffill().bfill()
    pivot_df.fillna(pivot_df.median(), inplace=True)
    
    features = pivot_df[['heart_rate', 'sbp', 'wbc']].values
    
    # Scale and save tensor
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    tensor_data = torch.FloatTensor(scaled_features)
    torch.save(tensor_data, 'data/processed/mimic_tensor.pt')
    
    print(f"\nSuccess! Extracted {len(pivot_df)} hourly records across {pivot_df['stay_id'].nunique()} ICU stays.")
    print("Saved production tensor to: data/processed/mimic_tensor.pt")
    
    return tensor_data

if __name__ == "__main__":
    parse_mimic_data()