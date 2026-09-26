import os
from huggingface_hub import HfApi


HF_USERNAME = "abhayfps"          
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-package-prediction"

LOCAL_DATA_PATH = "data/tourism.csv"


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
