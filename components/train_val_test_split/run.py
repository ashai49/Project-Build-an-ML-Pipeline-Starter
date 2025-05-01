import argparse
import os
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

def main(args):
    # Start W&B run under correct project
    run = wandb.init(project="nyc_airbnb", job_type="train_val_test_split")

    # Download and load input artifact
    artifact = run.use_artifact(args.input_artifact, type="cleaned_data")
    artifact_path = artifact.download()
    file_path = os.path.join(artifact_path, os.listdir(artifact_path)[0])  # assumes one file
    df = pd.read_csv(file_path)

    # Retrieve parameters
    test_size = float(args.test_size)
    random_seed = int(args.random_seed)
    stratify_by = args.stratify_by

    if stratify_by not in df.columns:
        raise ValueError(f"Column '{stratify_by}' not found in the dataset.")

    # Split data into train+val and test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df[stratify_by]
    )

    # Further split train_val into train and val
    train, val = train_test_split(
        train_val,
        test_size=test_size,
        random_state=random_seed,
        stratify=train_val[stratify_by]
    )

    # Save splits
    train.to_csv("train.csv", index=False)
    val.to_csv("val.csv", index=False)
    test.to_csv("test.csv", index=False)

    # Log artifacts
    for split_name in ["train", "val", "test"]:
        split_artifact = wandb.Artifact(
            name=f"{split_name}_data.csv",
            type="split_data",
            description=f"{split_name} split of the dataset"
        )
        split_artifact.add_file(f"{split_name}.csv")
        run.log_artifact(split_artifact)

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_artifact", type=str, required=True, help="Artifact name:version (e.g., clean_sample.csv:latest)")
    parser.add_argument("--test_size", type=float, required=True, help="Proportion for test and val splits")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--stratify_by", type=str, required=True, help="Column to stratify by")

    args = parser.parse_args()
    main(args)