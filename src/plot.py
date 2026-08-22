import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_three_plots():
    # 1. Load the inference data
    csv_path = "results/inference_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid")
    
    # Standardize colors for the Markov States
    state_colors = {
        "Low Risk": "#2ca02c",   # Green
        "Medium Risk": "#ff7f0e", # Orange
        "High Risk": "#d62728",   # Red
        "Critical": "#9467bd"     # Purple
    }
    state_order = ["Low Risk", "Medium Risk", "High Risk", "Critical"]
    os.makedirs("results", exist_ok=True)

    # ==========================================
    # GRAPH 1: The VAE Compression & Cutoffs
    # ==========================================
    print("Generating Graph 1...")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="risk_score_mu", hue="state_name", 
                 palette=state_colors, hue_order=state_order, 
                 kde=True, bins=60, alpha=0.6)
                 
    # Draw the mathematical cutoffs
    cutoffs = [-0.4175, 0.0093, 0.5112]
    for cutoff in cutoffs:
        plt.axvline(x=cutoff, color='black', linestyle='--', linewidth=2)
        
    plt.title("Graph 1: VAE Latent Risk Score (μ) Discretized into CTMC States", fontsize=14)
    plt.xlabel("VAE Latent Risk Score (μ)", fontsize=12)
    plt.ylabel("Patient Count", fontsize=12)
    plt.savefig("results/graph_1_vae_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # GRAPH 2: The Non-Linear Clinical Scatter
    # ==========================================
    print("Generating Graph 2...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="heart_rate", y="sbp", hue="state_name", 
                    palette=state_colors, hue_order=state_order, 
                    alpha=0.7, edgecolor=None)
                    
    plt.title("Graph 2: How Vitals Map to Markov States (Non-Linearity)", fontsize=14)
    plt.xlabel("Heart Rate (BPM)", fontsize=12)
    plt.ylabel("Systolic Blood Pressure (mmHg)", fontsize=12)
    # Move legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("results/graph_2_clinical_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # GRAPH 3: Phase-Type Countdown Trajectory
    # ==========================================
    print("Generating Graph 3...")
    plt.figure(figsize=(8, 6))
    
    # Filter out the critical state since time-to-critical is 0
    df_transient = df[df["state_name"] != "Critical"]
    
    sns.barplot(data=df_transient, x="state_name", y="hours_to_critical", 
                palette=state_colors, order=["Low Risk", "Medium Risk", "High Risk"],
                errorbar=None) # Disabling error bars for a clean look at the exact mean
                
    plt.title("Graph 3: Phase-Type Expected Time to Critical Collapse", fontsize=14)
    plt.xlabel("Current Transient State", fontsize=12)
    plt.ylabel("Expected Countdown (Hours)", fontsize=12)
    
    # Add numerical labels on top of the bars
    expected_values = [15.55, 17.85, 7.52] # Based on the matrix math we discussed
    for i, val in enumerate(expected_values):
        plt.text(i, val + 0.2, f"{val} hrs", ha='center', fontsize=11, fontweight='bold')
        
    plt.savefig("results/graph_3_mgf_countdown.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Successfully generated all 3 graphs in the 'results/' directory!")

if __name__ == "__main__":
    generate_three_plots()