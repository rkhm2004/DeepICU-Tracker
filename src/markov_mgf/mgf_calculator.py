
import numpy as np
from scipy.linalg import inv

def calculate_expected_time():
    print("1. Loading Q Matrix...")
    Q = np.load("data/processed/q_matrix.npy")
    
    print("2. Extracting Transient Matrix (T)...")
    # State 3 (Critical) is the absorbing state. 
    # T is the 3x3 sub-matrix of transient states: 0 (Low), 1 (Medium), 2 (High)
    T = Q[:3, :3]
    print(np.round(T, 4))
    
    print("\n3. Calculating Expected Time to Critical...")
    # Using the Phase-Type distribution expectation formula: E[X] = -T^(-1) * 1
    T_inv = inv(T)
    ones_vector = np.ones((3, 1))
    
    expected_times = -np.dot(T_inv, ones_vector)
    
    states = ["Low Risk (State 0)", "Medium Risk (State 1)", "High Risk (State 2)"]
    print("\n--- Clinical Prognosis ---")
    for i, state in enumerate(states):
        print(f"If patient is in {state} -> Expected time to Critical: {expected_times[i][0]:.2f} hours")

if __name__ == "__main__":
    calculate_expected_time()