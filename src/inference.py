import torch
import numpy as np
from scipy.linalg import inv
import sys
import os
import csv

# Temporarily add src/vae to path so we can import the model class
sys.path.append(os.path.abspath("src/vae"))
from model import ICU_VAE

def load_expected_times():
    """Calculates the exact hours to critical using the stored Q matrix."""
    Q = np.load("data/processed/q_matrix.npy")
    T = Q[:3, :3]
    T_inv = inv(T)
    ones_vector = np.ones((3, 1))
    return -np.dot(T_inv, ones_vector)

def main():
    print("=========================================")
    print("   DEEP-ICU EARLY WARNING SYSTEM LIVE    ")
    print("=========================================\n")
    
    # 1. Setup & Load Model
    states_map = ["Low Risk", "Medium Risk", "High Risk", "Critical"]
    expected_times = load_expected_times()
    cutoffs = [-0.41759565, 0.00931145, 0.51120774]
    
    print("Loading VAE Model...")
    model = ICU_VAE(input_dim=3, hidden_dim=16, latent_dim=1)
    model.load_state_dict(torch.load("src/vae/vae_checkpoint.pt"))
    model.eval()

    # 2. Load Real MIMIC-IV Data
    print("Loading real MIMIC-IV patient data...")
    if not os.path.exists("data/processed/mimic_tensor.pt"):
        print("Error: mimic_tensor.pt not found. Run mimic_parser.py first.")
        return
        
    real_data_tensor = torch.load("data/processed/mimic_tensor.pt")
    total_records = len(real_data_tensor)
    
    # 3. Setup Results Directory and CSV File
    os.makedirs("results", exist_ok=True)
    csv_filename = "results/inference_results.csv"
    
    print(f"Total records found: {total_records}")
    print(f"Starting batch inference... Saving results to {csv_filename}")
    print("-" * 40)

    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the CSV Header
        writer.writerow(["record_id", "heart_rate", "sbp", "wbc", 
                         "risk_score_mu", "markov_state", "state_name", "hours_to_critical"])
        
        # 4. Process ALL records to generate a massive dataset for plotting
        for i, patient_tensor in enumerate(real_data_tensor):
            with torch.no_grad():
                # Add batch dimension: shape becomes [1, 3]
                _, mu, _ = model(patient_tensor.unsqueeze(0))
                risk_score = mu.item()
                
            raw_bin = int(np.digitize(risk_score, cutoffs))
            state = 3 - raw_bin 
            
            # Display logic for human-readable vitals
            display_hr = (patient_tensor[0].item() * 15.0) + 85.0
            display_sbp = (patient_tensor[1].item() * 20.0) + 110.0
            display_wbc = (patient_tensor[2].item() * 4.0) + 12.0
            
            # Expected time logic
            if state == 3:
                countdown = 0.0
            else:
                countdown = expected_times[state][0]
                
            # Save the row to the CSV
            writer.writerow([
                i + 1,
                round(display_hr, 2),
                round(display_sbp, 2),
                round(display_wbc, 2),
                round(risk_score, 4),
                state,
                states_map[state],
                round(countdown, 2)
            ])
            
            # Print only the first 3 records to the terminal to confirm it's running
            if i < 3:
                print(f"Analyzed Record #{i+1} -> Score: {risk_score:.4f} | State: {states_map[state]}")

    print("-" * 40)
    print(f"SUCCESS! Inference complete.")
    print(f"Saved all {total_records} data points to: {csv_filename}")

if __name__ == "__main__":
    main()