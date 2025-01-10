import os
import mlflow
import pandas as pd

def log_dataset_as_artifact():
    # Define the path to the dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'updated_pollution_dataset.csv')
    
    # Check if the dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at path: {dataset_path}")

    # Read the dataset
    raw_data = pd.read_csv(dataset_path)  # Default delimiter is comma

    # Handle missing values (optional)
    raw_data.dropna(inplace=True)  # Example: Drop rows with missing values

    # Specify the target column
    target_column = "Air Quality"

    # Log the dataset as an artifact
    with mlflow.start_run() as run:
        # Log the dataset file as an artifact under the 'datasets' directory in MLflow
        mlflow.log_artifact(dataset_path, artifact_path="datasets")

        # Log additional parameters (optional)
        mlflow.log_param("num_samples", raw_data.shape[0])
        mlflow.log_param("num_features", raw_data.shape[1] - 1)  # Exclude target column

    # Retrieve the run information
    logged_run = mlflow.get_run(run.info.run_id)

    # Print run details
    print(f"Run ID: {logged_run.info.run_id}")
    print("Dataset logged as artifact under 'datasets/updated_pollution_dataset.csv'")

if __name__ == "__main__":
    log_dataset_as_artifact()
