import argparse
import pandas as pd
import mlflow
import os
import wandb

def go(input_artifact, output_artifact, output_type, output_description, min_price, max_price, sample):
    with mlflow.start_run():
        # Initialize wandb run
        run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning")

        # Download the input artifact
        artifact = run.use_artifact(input_artifact, type='raw_data')
        local_path = artifact.download()

        # Determine file path to load
        if sample:
            local_file = os.path.join(local_path, sample)
        else:
            local_file = os.path.join(local_path, os.listdir(local_path)[0])  # default: first file

        print(f"Downloaded input artifact to: {local_file}")

        # Read and clean the data
        df = pd.read_csv(local_file)

        # Filter by price
        idx = df['price'].between(min_price, max_price)
        df = df[idx].copy()

        # Filter by longitude and latitude
        idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
        df = df[idx].copy()

        # Save cleaned data locally
        output_path = os.path.join(os.getcwd(), output_artifact)
        df.to_csv(output_path, index=False)

        # Log with mlflow (optional)
        mlflow.log_artifact(output_path, artifact_path=output_type)

        # Log to wandb as new artifact
        cleaned_artifact = wandb.Artifact(
            name=output_artifact,
            type=output_type,
            description=output_description
        )
        cleaned_artifact.add_file(output_path)
        run.log_artifact(cleaned_artifact)

        print(f"Data cleaned and saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Airbnb data")

    parser.add_argument("--input_artifact", type=str, required=True, help="Artifact name:version (e.g. sample.csv:latest)")
    parser.add_argument("--output_artifact", type=str, required=True, help="Name for the output artifact (e.g. clean_sample.csv)")
    parser.add_argument("--output_type", type=str, required=True, help="Type of the output artifact (e.g. cleaned_data)")
    parser.add_argument("--output_description", type=str, required=True, help="Description of the output artifact")
    parser.add_argument("--min_price", type=float, required=True, help="Minimum price for filtering")
    parser.add_argument("--max_price", type=float, required=True, help="Maximum price for filtering")
    parser.add_argument("--sample", type=str, required=False, default="", help="Optional specific sample filename inside the artifact")

    args = parser.parse_args()

    go(
        input_artifact=args.input_artifact,
        output_artifact=args.output_artifact,
        output_type=args.output_type,
        output_description=args.output_description,
        min_price=args.min_price,
        max_price=args.max_price,
        sample=args.sample
    )