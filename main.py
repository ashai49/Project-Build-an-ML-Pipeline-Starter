import json
import subprocess
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    "test_regression_model"
]

@hydra.main(config_path=".", config_name="config")
def go(config: DictConfig):
    wandb.init(project='nyc_airbnb')
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            subprocess.run([
                "python",
                os.path.join(hydra.utils.get_original_cwd(), "components", "get_data", "run.py"),
                "--sample", config["etl"]["sample"],
                "--artifact_name", "sample.csv",
                "--artifact_type", "raw_data",
                "--artifact_description", "Raw file as downloaded"
            ])

        if "basic_cleaning" in active_steps:
            mlflow.run(
                uri=os.path.join(hydra.utils.get_original_cwd(), "components", "basic_cleaning"),
                entry_point="main",
                parameters={
                    "input_artifact": config["etl"]["input_artifact"],
                    "output_artifact": config["etl"]["output_artifact"],
                    "output_type": config["etl"]["output_type"],
                    "output_description": config["etl"]["output_description"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_check" in active_steps:
            mlflow.run(
                uri=os.path.join(hydra.utils.get_original_cwd(), "components", "data_check"),
                entry_point="main",
                parameters={
                    "csv": config["etl"]["output_artifact"] + ":latest",
                    "ref": config["etl"]["output_artifact"] + ":reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_split" in active_steps:
            mlflow.run(
                uri=os.path.join(hydra.utils.get_original_cwd(), "components", "data_split"),
                entry_point="main",
                parameters={
                    "input": config["etl"]["output_artifact"] + ":latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"].get("stratify_by", "none")
                },
            )

        if "train_random_forest" in active_steps:
            rf_config = os.path.join(hydra.utils.get_original_cwd(), config["modeling"]["rf_config"])

            mlflow.run(
                uri=os.path.join(config["main"]["src_dir"], "train_random_forest"),
                entry_point="main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_model.pkl"
                },
            )

        if "test_regression_model" in active_steps:
            mlflow.run(
                uri=os.path.join(hydra.utils.get_original_cwd(), "components", "test_regression_model"),
                entry_point="main",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_artifact": "test_data.csv:latest"
                },
            )

if __name__ == "__main__":
    go()