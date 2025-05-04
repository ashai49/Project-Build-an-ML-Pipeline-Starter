import argparse
import logging
import os
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def validate_artifacts(run, mlflow_model_ref, test_dataset_ref):
    """Validate artifact types and structure"""
    try:
        # Validate model artifact
        model_artifact = run.use_artifact(mlflow_model_ref)
        if model_artifact.type != "model":
            raise ValueError(f"Artifact {mlflow_model_ref} has type '{model_artifact.type}', expected 'model'")
        
        # Validate test data artifact
        test_artifact = run.use_artifact(test_dataset_ref)
        if test_artifact.type not in ["test_data", "split_data"]:
            logger.warning(f"Test artifact has type '{test_artifact.type}', expected 'test_data' or 'split_data'")
            
        return model_artifact, test_artifact
    except Exception as e:
        logger.error(f"Artifact validation failed: {str(e)}")
        raise

def go(args):
    run = wandb.init(job_type="test_model", 
                    config=vars(args),
                    settings=wandb.Settings(start_method="fork"))
    
    try:
        # Validate artifacts
        logger.info("Validating input artifacts")
        model_artifact, test_artifact = validate_artifacts(run, args.mlflow_model, args.test_dataset)

        # Download artifacts
        logger.info("Downloading model artifact")
        model_path = model_artifact.download()
        
        logger.info("Loading test data")
        test_path = test_artifact.file()
        test_df = pd.read_csv(test_path)
        
        # Validate test data structure
        if 'price' not in test_df.columns:
            raise ValueError("Test dataset must contain 'price' column")
        
        X_test = test_df.drop(columns=['price'])
        y_test = test_df['price']

        # Load model and predict
        logger.info("Loading model and making predictions")
        model = mlflow.sklearn.load_model(model_path)
        y_pred = model.predict(X_test)

        # Calculate metrics
        logger.info("Calculating evaluation metrics")
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"Test R² score: {r2:.4f}")
        logger.info(f"Test MAE: {mae:.4f}")

        # Log metrics
        run.summary.update({
            "test_r2": r2,
            "test_mae": mae
        })

        # Save and log predictions
        logger.info("Saving predictions")
        predictions = X_test.copy()
        predictions['true_price'] = y_test
        predictions['predicted_price'] = y_pred
        pred_path = "predictions.csv"
        predictions.to_csv(pred_path, index=False)
        
        log_artifact(
            run=run,
            file_path=pred_path,
            name="model_predictions",
            type="results",
            description="Model predictions on test set with true values"
        )

    except Exception as e:
        logger.error(f"Model testing failed: {str(e)}")
        raise
    finally:
        # Clean up
        if 'pred_path' in locals() and os.path.exists(pred_path):
            os.remove(pred_path)
        run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a production model against a test dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mlflow_model",
        type=str,
        required=True,
        help="Input MLflow model reference (e.g., 'model:prod' or 'model:v1')"
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        help="Test dataset artifact reference (e.g., 'test_data:v1')"
    )

    args = parser.parse_args()
    go(args)