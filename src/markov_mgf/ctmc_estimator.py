import torch
import numpy as np
from scipy.linalg import logm

def estimate_ctmc():
    print("1. Loading discretized risk states...")
    states = torch.load("data/processed/latent_states.pt").numpy()
    
    num_states = 4
    transition_counts = np.zeros((num_states, num_states))
    
    print("2. Counting state transitions...")
    # Count transitions from state(t) to state(t+1)
    for t in range(len(states) - 1):
        current_state = states[t]
        next_state = states[t + 1]
        transition_counts[current_state, next_state] += 1
        
    # Add a tiny epsilon to prevent division by zero or log of zero errors
    transition_counts += 1e-6 
    
    print("3. Calculating Probability Matrix (P)...")
    # Normalize rows so they sum to 1
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    P = transition_counts / row_sums
    
    print("4. Calculating Transition Rate Matrix (Q)...")
    # Q = ln(P) / delta_t (where delta_t = 1 hour)
    Q = logm(P)
    
    # Clean up any small imaginary parts caused by numerical approximation
    Q = np.real(Q)
    
    # Mathematical constraint: CTMC Q-matrix rows MUST sum exactly to 0
    # We enforce this by adjusting the diagonal
    for i in range(num_states):
        Q[i, i] -= np.sum(Q[i, :])
        
    print("\n--- Estimated Q Matrix ---")
    print(np.round(Q, 4))
    
    # Save the Q matrix for Phase 3 (MGF/Phase-Type)
    np.save("data/processed/q_matrix.npy", Q)
    print("\n5. Saved Q matrix to data/processed/q_matrix.npy")

if __name__ == "__main__":
    estimate_ctmc()