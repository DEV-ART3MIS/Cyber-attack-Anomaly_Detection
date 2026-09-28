import pandas as pd
import numpy as np

def generate_sample_data(n=200, scenario="Mixed Threats"):
    np.random.seed(42)

    time = np.arange(n)

    activity = 50 + 10*np.sin(time/10) + np.random.normal(0, 3, n)

    if scenario != "Normal":
        anomalies_idx = np.random.choice(n, size=10, replace=False)

        if scenario == "DDoS Attack":

            activity[anomalies_idx] += np.random.choice([40, 50, 60], size=10)
        elif scenario == "System Failure/Drop":

            activity[anomalies_idx] -= np.random.choice([30, 40], size=10)
        else: # "Mixed Threats"
            activity[anomalies_idx] += np.random.choice([35, -35], size=10)

    df = pd.DataFrame({
        "timestamp": time,
        "activity": activity
    })

    return df

def load_csv(file):
    return pd.read_csv(file)
