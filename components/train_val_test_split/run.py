#!/usr/bin/env python
"""
This script splits the provided dataframe in test and remainder
"""

import argparse
import logging
import os
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def strict_two_way_split(args):
    """Performs and verifies exactly one 2-way split"""
    # Initialize WandB with strict settings
    run = wandb.init(
        job_type="strict_two_way_split",
        project="nyc_airbnb",
        name=f"split_{os.getpid()}",
        config=dict(
            input_artifact=args.input,
            test_size=args.test_size,
            random_seed=args.random_seed,
            stratify_by=args.stratify_by,
            strict_mode=True  # Ensures no additional splits
        )
    )

    try:
        # 1. Load data with verification
        logger.info(f"STRICT MODE: Loading {args.input}")
        artifact = run.use_artifact(args.input)
        df = pd.read_csv(artifact.file())
        original_rows = len(df)
        logger.info(f"Verified input: {original_rows} rows")

        # 2. Perform SINGLE split with validation
        test_size = float(args.test_size)
        trainval, test = train_test_split(
            df,
            test_size=test_size,
            random_state=int(args.random_seed),
            stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
        )

        # 3. VERIFY split integrity
        expected_test = round(original_rows * test_size)
        actual_test = len(test)
        if abs(actual_test - expected_test) > 1:
            raise ValueError(f"Test size mismatch! Expected ~{expected_test}, got {actual_test}")

        # 4. Save with checksum verification
        artifacts = {}
        for data, name in [(trainval, "trainval_data"), (test, "test_data")]:
            with tempfile.NamedTemporaryFile("w", delete=False) as f:
                data.to_csv(f.name, index=False)
                artifacts[name] = {
                    'path': f.name,
                    'rows': len(data),
                    'checksum': pd.util.hash_pandas_object(data).sum()
                }

        # 5. Log artifacts with verification
        for name, data in artifacts.items():
            artifact = wandb.Artifact(
                name,
                type="dataset",
                description=f"STRICT MODE: {name} split",
                metadata={
                    'source': args.input,
                    'original_rows': original_rows,
                    'checksum': data['checksum'],
                    'strict_mode': True
                }
            )
            artifact.add_file(data['path'])
            run.log_artifact(artifact)
            logger.info(f"STRICT LOG: {name} with {data['rows']} rows (checksum: {data['checksum']})")

        # 6. Final verification
        logged_artifacts = [a.name for a in run.logged_artifacts()]
        if len(logged_artifacts) != 2 or "trainval_data" not in logged_artifacts or "test_data" not in logged_artifacts:
            raise RuntimeError("INVALID ARTIFACT COUNT! Expected exactly 2 artifacts")

    except Exception as e:
        logger.error(f"STRICT MODE FAILURE: {str(e)}")
        raise
    finally:
        run.finish()
        logger.info("STRICT MODE: Run completed with verification")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STRICT 2-way data split only")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--test_size", type=str, required=True)
    parser.add_argument("--random_seed", type=str, default="42")
    parser.add_argument("--stratify_by", type=str, default="none")
    
    args = parser.parse_args()
    
    # Convert and validate parameters
    try:
        test_size = float(args.test_size)
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
    except ValueError as e:
        raise ValueError(f"Invalid test_size: {args.test_size}") from e

    strict_two_way_split(args)