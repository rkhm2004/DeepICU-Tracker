import pandas as pd
import os

def generate_dataset_summary():
    csv_path = "results/inference_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run inference first.")
        return
        
    df = pd.read_csv(csv_path)
    
    print("=========================================")
    print("      DATASET SUMMARY FOR SLIDE          ")
    print("=========================================\n")
    
    print(f"Total Patient Records (Hourly Snapshots): {len(df):,}")
    print("Features Extracted: 3 (Heart Rate, SBP, WBC)")
    
    print("\n--- Feature Statistics ---")
    stats = df[['heart_rate', 'sbp', 'wbc']].describe().loc[['mean', 'std', 'min', 'max']]
    print(stats.round(2))
    
    print("\n--- Risk State Distribution ---")
    state_counts = df['state_name'].value_counts()
    for state, count in state_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{state}: {count:,} ({percentage:.1f}%)")

if __name__ == "__main__":
    generate_dataset_summary()