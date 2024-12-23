import pandas as pd
import numpy as np
import mlflow

def calculate_psi(expected:pd.Series, actual:pd.Series, buckets=10):
    """Calculate Population Stability Index"""

    #using fixed interval
    
    def get_bucket_values(data:pd.Series, bucket_edges):

        bucket_values = pd.cut(data, bucket_edges, labels=False, include_lowest=True)
        
        counts = pd.value_counts(bucket_values).sort_index()
        print(counts)
        counts_dist = counts/np.sum(counts)
        print(counts_dist)
        
        return counts_dist
    
    # Calculate bucket edges using expected 
    edges = np.percentile(expected, np.linspace(0, 100, buckets+1))
    print(edges)
    edges = np.unique(edges) # to ensure that the bin division does not overlap each other
    print(edges)
    
    # Get distributions
    expected_dist = get_bucket_values(expected, edges)
    actual_dist = get_bucket_values(actual, edges)
    
    # Calculate PSI
    psi = np.sum(
        (actual_dist - expected_dist) * 
        np.log(actual_dist / expected_dist)
    )
    
    print(actual_dist)
    print(expected_dist)
    print(actual_dist - expected_dist)
    print(np.log(actual_dist / expected_dist))
    return psi

def detect_drift(reference_data_path, new_data_path, run_id):
    """Detect drift using PSI"""
    # Load data
    reference_data = pd.read_csv(reference_data_path)
    new_data = pd.read_csv(new_data_path)
    
    # Calculate PSI for each feature
    drift_metrics = {}
    for column in reference_data.columns:
        if column != 'churn':
            print("COL NAME: " +column)
            psi = calculate_psi(
                reference_data[column],
                new_data[column]
            )
            drift_metrics[f"{column}_psi"] = psi
    
    # Log drift metrics to MLflow
    with mlflow.start_run(run_name="drift_detection"):
        mlflow.log_metrics(drift_metrics)
        
        # Determine if drift is significant (PSI > 0.2 is considered significant)
        drift_detected = any(v > 0.2 for v in drift_metrics.values())
        mlflow.log_param("drift_detected", drift_detected)
        
        return drift_metrics, drift_detected
    
    
def detect_drift_psi(reference_data_path, new_data_path, run_id):
    """Detect drift using PSI"""
    # Load data
    reference_data = pd.read_csv(reference_data_path)
    new_data = pd.read_csv(new_data_path)
    
    # Calculate PSI for each feature
    drift_metrics = {}
    for column in reference_data.columns:
        if column != 'churn':
            psi = calculate_psi(
                reference_data[column],
                new_data[column]
            )
            drift_metrics[f"{column}_psi"] = psi
    
    # Log drift metrics to MLflow
    with mlflow.start_run(run_name="drift_detection"):
        mlflow.log_metrics(drift_metrics)
        
        # Determine if drift is significant (PSI > 0.2 is considered significant)
        drift_detected = any(v > 0.2 for v in drift_metrics.values())
        mlflow.log_param("drift_detected", drift_detected)
        
        return drift_metrics, drift_detected

if __name__ == "__main__":
    metrics, has_drift = detect_drift(
        "data/train.csv",
        "data/new_data.csv",
        "latest"
    )
    print("Drift Metrics:", metrics)
    print("Drift Detected:", has_drift)