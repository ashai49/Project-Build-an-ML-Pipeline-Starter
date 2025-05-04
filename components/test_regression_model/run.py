import argparse
import logging
import os
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def validate_artifact_types(run, args):
    """Validate that artifacts have the correct types"""
    try:
        model_artifact = run.use_artifact(args.mlflow_model)
        if not model_artifact.type.startswith('model'):
            logger.warning(f"Artifact {args.mlflow_model} has type '{model_artifact.type}', expected model type")
        
        test_data_artifact = run.use_artifact(args.test_dataset)
        if not test_data_artifact.type.startswith(('test_data', 'split_data')):
            logger.warning(f"Artifact {args.test_dataset} has type '{test_data_artifact.type}', expected test_data type")
    except Exception as e:
        logger.error(f"Artifact validation failed: {str(e)}")
        raise

def go(args):
    run = wandb.init(job_type="test_model")
    run.config.update(args)

    try:
        logger.info("Validating artifacts")
        validate_artifact_types(run, args)

        logger.info("Downloading artifacts")
        # Download input artifacts
        model_local_path = run.use_artifact(args.mlflow_model).download()
        test_dataset_path = run.use_artifact(args.test_dataset).file()

        logger.info("Loading test data")
        test_df = pd.read_csv(test_dataset_path)
        if 'price' not in test_df.columns:
            raise ValueError("Test dataset must contain 'price' column")
            
        X_test = test_df.drop(columns=['price'], errors='ignore')
        y_test = test_df['price']

        logger.info("Loading model and performing inference")
        sk_pipe = mlflow.sklearn.load_model(model_local_path)
        y_pred = sk_pipe.predict(X_test)

        logger.info("Calculating metrics")
        r_squared = sk_pipe.score(X_test, y_test)
        mae = mean_absolute_error(y_test, y_pred)

        logger.info(f"R² Score: {r_squared:.4f}")
        logger.info(f"MAE: {mae:.2f}")

        # Log metrics
        run.summary['r2'] = r_squared
        run.summary['mae'] = mae

        # Log predictions as artifact
        pred_df = X_test.copy()
        pred_df['price'] = y_test
        pred_df['predicted_price'] = y_pred
        pred_path = "test_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        
        log_artifact(
            run,
            pred_path,
            "test_predictions",
            "predictions",
            "Model predictions on test set"
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        # Clean up temporary files
        if 'pred_path' in locals() and os.path.exists(pred_path):
            os.remove(pred_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided model against the test dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLFlow model (format: 'path:version' or 'path:prod')",
        required=True
    )

    parser.add_argument(
        "--test_dataset", 
        type=str,
        help="Test dataset artifact (format: 'path:version')",
        required=True
    )

    args = parser.parse_args()
    go(args)