import torch
import numpy as np
from scipy.linalg import inv
import sys
import os

# Temporarily add src/vae to path so we can import your model class
sys.path.append(os.path.abspath("src/vae"))
from model import ICU_VAE

def load_expected_times():
    """Calculates the exact hours to critical using the stored Q matrix."""
    Q = np.load("data/processed/q_matrix.npy")
    T = Q[:3, :3]
    T_inv = inv(T)
    ones_vector = np.ones((3, 1))
    return -np.dot(T_inv, ones_vector)

def predict_patient_status(hr, sbp, wbc):
    """Passes vitals through the VAE to get the discrete risk state."""
    # Approximate scaling based on general ICU means to match your tensor
    scaled_hr = (hr - 85.0) / 15.0
    scaled_sbp = (sbp - 110.0) / 20.0
    scaled_wbc = (wbc - 12.0) / 4.0
    x = torch.FloatTensor([[scaled_hr, scaled_sbp, scaled_wbc]])
    
    # Load the trained Neural Network
    model = ICU_VAE(input_dim=3, hidden_dim=16, latent_dim=1)
    model.load_state_dict(torch.load("src/vae/vae_checkpoint.pt"))
    model.eval()
    
    with torch.no_grad():
        _, mu, _ = model(x)
        risk_score = mu.item()
        
    # Cutoffs generated from your training loop
    cutoffs = [-0.41759565, 0.00931145, 0.51120774]
    state = int(np.digitize(risk_score, cutoffs))
    
    return state, risk_score

def main():
    print("=========================================")
    print("   DEEP-ICU EARLY WARNING SYSTEM LIVE    ")
    print("=========================================\n")
    
    # Test Case 1: A stable patient
    patients = [
        {"name": "Patient A (Stable)", "hr": 75, "sbp": 120, "wbc": 8.5},
        {"name": "Patient B (Deteriorating)", "hr": 115, "sbp": 88, "wbc": 18.2}
    ]
    
    states_map = ["Low Risk", "Medium Risk", "High Risk", "Critical"]
    expected_times = load_expected_times()
    
    for p in patients:
        print(f"Analyzing {p['name']}...")
        print(f"Vitals -> Heart Rate: {p['hr']} | SBP: {p['sbp']} | WBC: {p['wbc']}")
        
        state, score = predict_patient_status(p['hr'], p['sbp'], p['wbc'])
        
        print(f"Deep Learning Risk Score: {score:.4f}")
        print(f"Current Markov State: {states_map[state]}")
        
        if state == 3:
            print("ALERT: Patient is already in CRITICAL condition!")
        else:
            countdown = expected_times[state][0]
            print(f"Prognosis: Expected time to Critical is {countdown:.2f} hours")
        print("-" * 40)

if __name__ == "__main__":
    main()