import pandas as pd
import numpy as np

def generate_mock_icu_data(num_patients=100, hours_per_patient=48):
    data = []
    for pid in range(1, num_patients + 1):
        for hour in range(hours_per_patient):
            hr = int(np.random.normal(85, 10))
            bp_sys = int(np.random.normal(110, 15))
            wbc = np.random.normal(12.0, 2.0)
            data.append([pid, hour, hr, bp_sys, wbc])
            
    df = pd.DataFrame(data, columns=['patient_id', 'hour', 'heart_rate', 'bp_systolic', 'wbc'])
    df.to_csv('data/raw/mock_vitals.csv', index=False)
    print("Mock data generated at data/raw/mock_vitals.csv")

if __name__ == "__main__":
    generate_mock_icu_data()