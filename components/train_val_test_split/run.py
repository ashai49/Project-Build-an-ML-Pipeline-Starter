import argparse
import os
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

# Initialize wandb run
run = wandb.init(project="Project-Build-an-ML-Pipeline-Starter-components_train_val_test_split")

def main(args):
    # Correct file path for the clean_sample.csv
    file_path = args.input  # This is passed via the command line, so it's more flexible

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a valid file: {file_path}")

    # Log the initial artifact (input dataset)
    artifact = wandb.Artifact("clean_sample.csv", type="dataset")
    artifact.add_file(file_path)
    run.log_artifact(artifact)

    # Load the dataset
    df = pd.read_csv(file_path)

    # Retrieve parameters
    test_size = float(args.test_size)
    random_seed = int(args.random_seed)
    stratify_by = args.stratify_by

    # Check if the stratify column exists in the dataframe
    if stratify_by not in df.columns:
        raise ValueError(f"Column '{stratify_by}' not found in the dataset.")

    # Split the data
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_seed, stratify=df[stratify_by])
    train, val = train_test_split(train_val, test_size=test_size, random_state=random_seed, stratify=train_val[stratify_by])

    # Save the splits to CSV files
    train_file = "train.csv"
    val_file = "val.csv"
    test_file = "test.csv"

    train.to_csv(train_file, index=False)
    val.to_csv(val_file, index=False)
    test.to_csv(test_file, index=False)

    # Log the train, validation, and test splits as artifacts
    train_artifact = wandb.Artifact("train.csv", type="dataset")
    train_artifact.add_file(train_file)
    run.log_artifact(train_artifact)

    val_artifact = wandb.Artifact("val.csv", type="dataset")
    val_artifact.add_file(val_file)
    run.log_artifact(val_artifact)

    test_artifact = wandb.Artifact("test.csv", type="dataset")
    test_artifact.add_file(test_file)
    run.log_artifact(test_artifact)

    # Finish the wandb run
    run.finish()

if __name__ == "__main__":
    # Argument parser for the CLI
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--input", type=str, required=True, help="Path to the input dataset (CSV file)")
    parser.add_argument("--test_size", type=float, required=True, help="Proportion of the dataset to be used as the test set")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for splitting the dataset")
    parser.add_argument("--stratify_by", type=str, required=True, help="Column to stratify by when splitting the dataset")

    # Parse arguments
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(args)