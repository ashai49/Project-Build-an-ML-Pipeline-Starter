import logging
import os
import shutil
import matplotlib.pyplot as plt
import mlflow
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, OneHotEncoder
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline

from omegaconf import OmegaConf
import hydra
from hydra.utils import to_absolute_path

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def delta_date_feature(dates):
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()

@hydra.main(config_path="../../config", config_name="config")
def go(cfg):
    # Log config
    logger.info(OmegaConf.to_yaml(cfg))

    run = wandb.init(project="nyc_airbnb", job_type="train_random_forest")
    run.config.update(OmegaConf.to_container(cfg, resolve=True))

    trainval_local_path = run.use_artifact(cfg.trainval_artifact).file()
    df = pd.read_csv(trainval_local_path)
    y = df.pop("price")

    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    X_train, X_val, y_train, y_val = train_test_split(
        df,
        y,
        test_size=cfg.val_size,
        stratify=df[cfg.stratify_by],
        random_state=cfg.random_seed
    )

    logger.info("Preparing sklearn pipeline")
    sk_pipe, processed_features = get_inference_pipeline(cfg.modeling.random_forest, cfg.max_tfidf_features)

    logger.info("Fitting")
    sk_pipe.fit(X_train, y_train)

    logger.info("Scoring")
    r2 = sk_pipe.score(X_val, y_val)
    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"R^2: {r2}")
    logger.info(f"MAE: {mae}")

    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    mlflow.sklearn.save_model(sk_pipe, path="random_forest_dir", input_example=X_train.iloc[:5])

    artifact = wandb.Artifact(
        cfg.output_artifact,
        type=cfg.output_type,
        description="Trained Random Forest model",
        metadata=OmegaConf.to_container(cfg.modeling.random_forest, resolve=True)
    )
    artifact.add_dir("random_forest_dir")
    run.log_artifact(artifact)

    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)
    run.log({"feature_importance": wandb.Image(fig_feat_imp)})
    run.summary["r2"] = r2
    run.summary["mae"] = mae

    run.finish()

def plot_feature_importance(pipe, feat_names):
    feat_imp = pipe["random_forest"].feature_importances_[: len(feat_names) - 1]
    nlp_importance = sum(pipe["random_forest"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.bar(range(len(feat_imp)), feat_imp, color="r", align="center")
    ax.set_xticks(range(len(feat_imp)))
    ax.set_xticklabels(np.array(feat_names), rotation=90)
    fig.tight_layout()
    return fig

def get_inference_pipeline(rf_config, max_tfidf_features):
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]

    ordinal_categorical_preproc = OrdinalEncoder()
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )

    zero_imputed = [
        "minimum_nights", "number_of_reviews", "reviews_per_month",
        "calculated_host_listings_count", "availability_365",
        "longitude", "latitude"
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    date_imputer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="2010-01-01"),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )

    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=False,
            max_features=max_tfidf_features,
            stop_words="english"
        )
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"]),
        ],
        remainder="drop",
    )

    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]

    rf = RandomForestRegressor(
        max_depth=rf_config.get("max_depth"),
        n_estimators=rf_config.get("n_estimators"),
        min_samples_split=rf_config.get("min_samples_split", 2),
        min_samples_leaf=rf_config.get("min_samples_leaf", 1),
        random_state=rf_config.get("random_state", 42)
    )

    sk_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("random_forest", rf)
    ])

    return sk_pipe, processed_features

if __name__ == "__main__":
    go()