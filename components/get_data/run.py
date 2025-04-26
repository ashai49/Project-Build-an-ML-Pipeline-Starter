#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "wandb_utils")))

import wandb
import mlflow
from components.wandb_utils.log_artifact import log_artifact 


def main(config):
    
    artifact_name = "raw_data"
    artifact_type = "dataset"
    artifact_description = "Raw data artifact"
    filename = config["etl"]["data_path"]

   
    wandb_run = wandb.init(project="your_project_name", job_type="data_preparation")

    log_artifact(artifact_name, artifact_type, artifact_description, filename, wandb_run)

    mlflow.log_param("artifact_name", artifact_name)

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_name="config.yaml")
    def run_etl(config: DictConfig):
        main(config)

    run_etl()