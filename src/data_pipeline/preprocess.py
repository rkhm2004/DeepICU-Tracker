import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df = df.sort_values(by=['patient_id', 'hour'])
    
    features = df[['heart_rate', 'bp_systolic', 'wbc']].values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    tensor_data = torch.FloatTensor(scaled_features)
    torch.save(tensor_data, 'data/processed/vitals_tensor.pt')
    print("Data preprocessed and saved to data/processed/vitals_tensor.pt")
    
    return tensor_data

if __name__ == "__main__":
    load_and_preprocess('data/raw/mock_vitals.csv')