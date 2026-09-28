import numpy as np

def detect_anomalies(actual, predicted, threshold):

    residuals = actual - predicted
    abs_residuals = np.abs(residuals)

    anomalies = abs_residuals > threshold


    tags = []
    for i in range(len(anomalies)):
        if anomalies[i]:
            if residuals[i] > 0:
                tags.append("Sudden Spike (Possible DDoS/Brute Force)")
            else:
                tags.append("Sudden Drop (System Failure/Tampering)")
        else:
            tags.append("Normal")

    return anomalies, abs_residuals, tags
