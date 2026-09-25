"""
Data Registration
------------------
Registers the raw tourism.csv dataset on the Hugging Face Hub so that every
later pipeline stage (data prep, training) pulls from a single versioned
source of truth instead of a local file.

Usage:
    python data_register.py

Requires:
    HF_TOKEN        - Hugging Face access token with "write" scope
                       (set as an environment variable / GitHub secret)
    HF_USERNAME     - your Hugging Face username (edit the constant below)
"""

import os
from huggingface_hub import HfApi

# ---------------------------------------------------------------------------
# CONFIG - replace with your own Hugging Face username
# ---------------------------------------------------------------------------
HF_USERNAME = "abhayfps"          # <-- TODO: replace with your HF username
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-package-prediction"

LOCAL_DATA_PATH = "tourism_project/data/tourism.csv"


def main():
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        print(
            "No HF_TOKEN found in the environment. Skipping the upload to the "
            "Hugging Face Hub - the raw file is already available locally at "
            f"'{LOCAL_DATA_PATH}'. Set HF_TOKEN (and HF_USERNAME above) to "
            "register the dataset online."
        )
        return

    api = HfApi(token=hf_token)

    # Create the dataset repo if it does not already exist
    api.create_repo(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )

    # Upload the raw CSV
    api.upload_file(
        path_or_fileobj=LOCAL_DATA_PATH,
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
    )

    print(f"Raw dataset uploaded to https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
