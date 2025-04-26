import argparse
import pandas as pd
import mlflow
import os


def go(input_artifact, output_artifact, output_type, output_description, min_price, max_price):

    with mlflow.start_run():

        mlflow.log_param("input_artifact", input_artifact)
        mlflow.log_param("output_artifact", output_artifact)
        mlflow.log_param("output_type", output_type)
        mlflow.log_param("output_description", output_description)
        mlflow.log_param("min_price", min_price)
        mlflow.log_param("max_price", max_price)

        local_path = mlflow.artifacts.download_artifacts(artifact_uri=input_artifact)

        df = pd.read_csv(local_path)

        idx = df['price'].between(min_price, max_price)
        df = df[idx].copy()

        idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
        df = df[idx].copy()

        output_path = os.path.join(os.getcwd(), output_artifact)
        df.to_csv(output_path, index=False)

        mlflow.log_artifact(output_path, artifact_path=output_type)

        mlflow.log_param("output_description", output_description)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Airbnb data")

    parser.add_argument("--input_artifact", type=str, required=True, help="Path to the input artifact (e.g. sample.csv:latest)")
    parser.add_argument("--output_artifact", type=str, required=True, help="Name for the output artifact (e.g. clean_sample.csv)")
    parser.add_argument("--output_type", type=str, required=True, help="The type of output data (e.g. cleaned_data)")
    parser.add_argument("--output_description", type=str, required=True, help="Description of the cleaned data")
    parser.add_argument("--min_price", type=float, required=True, help="Minimum price for filtering")
    parser.add_argument("--max_price", type=float, required=True, help="Maximum price for filtering")

    args = parser.parse_args()

    go(
        input_artifact=args.input_artifact,
        output_artifact=args.output_artifact,
        output_type=args.output_type,
        output_description=args.output_description,
        min_price=args.min_price,
        max_price=args.max_price
    )