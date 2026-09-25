"""
Data Preparation
-----------------
Loads the raw tourism dataset (from the Hugging Face dataset repo if HF_TOKEN
is available, otherwise from the local data/ folder), cleans it, and produces
a stratified train/test split ready for model training.

Cleaning steps and why:
    * Drop 'Unnamed: 0' and 'CustomerID'  -> identifiers, not predictive.
    * Fix the 'Fe Male' typo in Gender    -> data entry error, same as 'Female'.
    * Merge 'Unmarried' into 'Single' in MaritalStatus -> both describe the
      same status; keeping them separate would artificially split the signal.
    * Cast the target 'ProdTaken' to int.

Usage:
    python prep.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, HfApi

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
HF_USERNAME = "abhayfps"          # <-- TODO: replace with your HF username
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-package-prediction"

LOCAL_DATA_DIR = "tourism_project/data"
LOCAL_RAW_PATH = f"{LOCAL_DATA_DIR}/tourism.csv"

TARGET = "ProdTaken"
DROP_COLS = ["Unnamed: 0", "CustomerID"]


def load_raw_data(hf_token: str | None) -> pd.DataFrame:
    """Load the raw dataset, preferring the Hugging Face Hub copy when a
    token is available so every pipeline run works off the registered
    (versioned) dataset rather than whatever happens to sit on disk."""
    if hf_token:
        try:
            path = hf_hub_download(
                repo_id=DATASET_REPO_ID,
                filename="tourism.csv",
                repo_type="dataset",
                token=hf_token,
            )
            print(f"Loaded raw data from Hugging Face Hub: {DATASET_REPO_ID}")
            return pd.read_csv(path)
        except Exception as e:
            print(f"Could not fetch dataset from HF Hub ({e}); falling back to local file.")

    print(f"Loaded raw data from local file: {LOCAL_RAW_PATH}")
    return pd.read_csv(LOCAL_RAW_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    df[TARGET] = df[TARGET].astype(int)

    return df


def main():
    hf_token = os.getenv("HF_TOKEN")

    raw_df = load_raw_data(hf_token)
    clean_df = clean_data(raw_df)

    print(f"Shape after cleaning: {clean_df.shape}")
    print(f"Target distribution:\n{clean_df[TARGET].value_counts(normalize=True).round(3)}")

    X = clean_df.drop(columns=[TARGET])
    y = clean_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    X_train.to_csv(f"{LOCAL_DATA_DIR}/Xtrain.csv", index=False)
    X_test.to_csv(f"{LOCAL_DATA_DIR}/Xtest.csv", index=False)
    y_train.to_csv(f"{LOCAL_DATA_DIR}/ytrain.csv", index=False)
    y_test.to_csv(f"{LOCAL_DATA_DIR}/ytest.csv", index=False)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv to '{LOCAL_DATA_DIR}/'")

    if hf_token:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", private=False, exist_ok=True)
        for fname in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
            api.upload_file(
                path_or_fileobj=f"{LOCAL_DATA_DIR}/{fname}",
                path_in_repo=fname,
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
            )
        print(f"Uploaded processed splits to https://huggingface.co/datasets/{DATASET_REPO_ID}")
    else:
        print("No HF_TOKEN found - processed splits kept local only.")


if __name__ == "__main__":
    main()
