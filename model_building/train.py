import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier
from huggingface_hub import hf_hub_download, HfApi


HF_USERNAME = "abhayfps"         
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-package-prediction"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-package-prediction-model"

LOCAL_DATA_DIR = "data"
MODEL_OUT_PATH = "model_building/best_tourism_model_v1.joblib"
DEPLOYMENT_MODEL_PATH = "deployment/best_tourism_model_v1.joblib"

EXPERIMENT_NAME = "Tourism_Package_Prediction"

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]


def load_split(name: str, hf_token: str | None) -> pd.DataFrame:
    local_path = f"{LOCAL_DATA_DIR}/{name}"
    if hf_token:
        try:
            path = hf_hub_download(
                repo_id=DATASET_REPO_ID, filename=name,
                repo_type="dataset", token=hf_token,
            )
            return pd.read_csv(path)
        except Exception as e:
            print(f"Could not fetch '{name}' from HF Hub ({e}); using local copy.")
    return pd.read_csv(local_path)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def get_candidates(pos_weight: float):
    """Return {name: (estimator, param_grid)} for a small, fast search."""
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            {"clf__C": [0.1, 1, 10]},
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 8, 12],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                eval_metric="logloss", random_state=42,
                scale_pos_weight=pos_weight,
            ),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.05, 0.1],
            },
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    hf_token = os.getenv("HF_TOKEN")

    X_train = load_split("Xtrain.csv", hf_token)
    X_test = load_split("Xtest.csv", hf_token)
    y_train = load_split("ytrain.csv", hf_token).iloc[:, 0]
    y_test = load_split("ytest.csv", hf_token).iloc[:, 0]

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    mlflow.set_experiment(EXPERIMENT_NAME)

    preprocessor = build_preprocessor()
    candidates = get_candidates(pos_weight)

    results = {}
    fitted_pipelines = {}

    for name, (estimator, param_grid) in candidates.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", estimator)])

        with mlflow.start_run(run_name=name):
            search = GridSearchCV(pipe, param_grid, scoring="f1", cv=5, n_jobs=-1)
            search.fit(X_train, y_train)

            best_pipe = search.best_estimator_
            metrics = evaluate(best_pipe, X_test, y_test)

            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("model_family", name)
            mlflow.sklearn.log_model(best_pipe, artifact_path="model")

            print(f"[{name}] best_params={search.best_params_}")
            print(f"[{name}] test_metrics={ {k: round(v, 4) for k, v in metrics.items()} }")

            results[name] = metrics
            fitted_pipelines[name] = best_pipe

    # Select the best model on test F1 (imbalanced target -> F1 over accuracy)
    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = fitted_pipelines[best_name]
    best_metrics = results[best_name]

    print(f"\nBest model: {best_name} -> {best_metrics}")
    print("\nClassification report for the best model:")
    print(classification_report(y_test, best_model.predict(X_test)))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, best_model.predict(X_test)))

    # Log + register the winning model as its own MLflow run
    with mlflow.start_run(run_name=f"BEST_{best_name}"):
        mlflow.log_metrics(best_metrics)
        mlflow.set_tag("model_family", best_name)
        mlflow.set_tag("selected_as_best", "true")
        mlflow.sklearn.log_model(
            best_model, artifact_path="model",
            registered_model_name="tourism_package_prediction_model",
        )

    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_OUT_PATH)
    os.makedirs(os.path.dirname(DEPLOYMENT_MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, DEPLOYMENT_MODEL_PATH)
    print(f"\nSaved best model to '{MODEL_OUT_PATH}' and '{DEPLOYMENT_MODEL_PATH}'")

    if hf_token:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False, exist_ok=True)
        api.upload_file(
            path_or_fileobj=MODEL_OUT_PATH,
            path_in_repo="best_tourism_model_v1.joblib",
            repo_id=MODEL_REPO_ID,
            repo_type="model",
        )
        print(f"Uploaded best model to https://huggingface.co/{MODEL_REPO_ID}")
    else:
        print("No HF_TOKEN found - model kept local only.")


if __name__ == "__main__":
    main()
