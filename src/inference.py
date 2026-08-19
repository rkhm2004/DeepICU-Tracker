import torch
import numpy as np
from scipy.linalg import inv
import sys
import os

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
    
    # 1. Setup & Load Model (Done once for efficiency)
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
    
    # Let's slice the first 5 hourly records for a live test
    sample_records = real_data_tensor[:5] 
    
    for i, patient_tensor in enumerate(sample_records):
        print(f"\nAnalyzing Real Record #{i+1}...")
        
        # Reverse the approximation for display purposes only 
        # (Assuming dataset means ~85 HR, 110 SBP, 12 WBC for printing)
        display_hr = (patient_tensor[0].item() * 15.0) + 85.0
        display_sbp = (patient_tensor[1].item() * 20.0) + 110.0
        display_wbc = (patient_tensor[2].item() * 4.0) + 12.0
        
        print(f"Vitals (Approx) -> Heart Rate: {display_hr:.1f} | SBP: {display_sbp:.1f} | WBC: {display_wbc:.1f}")
        
        # 3. Predict Status
        with torch.no_grad():
            # Add batch dimension: shape becomes [1, 3]
            _, mu, _ = model(patient_tensor.unsqueeze(0))
            risk_score = mu.item()
            
        raw_bin = int(np.digitize(risk_score, cutoffs))
        state = 3 - raw_bin 
        
        print(f"Deep Learning Risk Score: {risk_score:.4f}")
        print(f"Current Markov State: {states_map[state]}")
        
        if state == 3:
            print("ALERT: Patient is already in CRITICAL condition!")
        else:
            countdown = expected_times[state][0]
            print(f"Prognosis: Expected time to Critical is {countdown:.2f} hours")
        print("-" * 40)

if __name__ == "__main__":
    main()