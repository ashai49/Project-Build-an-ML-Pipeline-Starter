#!/usr/bin/env python
"""
Train Random Forest model with unique output directories
"""
import argparse
import json
import logging
import os
import pandas as pd
import wandb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import mlflow
import mlflow.sklearn
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def load_rf_config(config_path):
    """Load random forest config from JSON file"""
    with open(config_path) as f:
        return json.load(f)

def go(args):
    run = wandb.init(job_type="train_random_forest")
    
    try:
        # Load RF config
        rf_config = load_rf_config(args.rf_config)
        run.config.update(rf_config)
        
        # Create unique output directory
        model_dir = tempfile.mkdtemp(prefix="random_forest_")
        logger.info(f"Using temporary model directory: {model_dir}")
        
        # Load data
        logger.info("Downloading training data")
        train_data = run.use_artifact(args.train_artifact)
        train_path = train_data.file()
        df_train = pd.read_csv(train_path)
        X_train = df_train.drop("price", axis=1)
        y_train = df_train["price"]

        logger.info("Downloading validation data")
        val_data = run.use_artifact(args.val_artifact)
        val_path = val_data.file()
        df_val = pd.read_csv(val_path)
        X_val = df_val.drop("price", axis=1)
        y_val = df_val["price"]

        # Preprocessing pipeline
        numerical_features = ["minimum_nights", "number_of_reviews", "reviews_per_month"]
        categorical_features = ["neighbourhood_group", "room_type"]

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Full pipeline
        sk_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('random_forest', RandomForestRegressor(**rf_config))
        ])

        # Train
        logger.info("Training model")
        sk_pipe.fit(X_train, y_train)

        # Evaluate
        logger.info("Evaluating model")
        y_pred = sk_pipe.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        logger.info(f"MAE: {mae:.2f}, R2: {r2:.2f}")

        # Log metrics
        run.summary["mae"] = mae
        run.summary["r2"] = r2

        # Save model
        logger.info("Exporting model")
        mlflow.sklearn.save_model(sk_pipe, model_dir)
        
        artifact = wandb.Artifact(
            args.output_artifact,
            type="model",
            description="Random Forest model with preprocessing",
            metadata=rf_config
        )
        artifact.add_dir(model_dir)
        run.log_artifact(artifact)

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        run.finish()
        # Clean up temporary directory
        if 'model_dir' in locals():
            try:
                import shutil
                shutil.rmtree(model_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest")

    parser.add_argument("--train_artifact", type=str, required=True)
    parser.add_argument("--val_artifact", type=str, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)
    parser.add_argument("--rf_config", type=str, required=True)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--stratify_by", type=str, default="none")

    args = parser.parse_args()
    go(args)