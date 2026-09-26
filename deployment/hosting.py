
import os
from huggingface_hub import HfApi

HF_USERNAME = "abhayfps"         
SPACE_REPO_ID = f"{HF_USERNAME}/tourism-package-prediction-app"

DEPLOYMENT_DIR = "deployment"


def main():
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        print(
            "No HF_TOKEN found in the environment - skipping the push to "
            "Hugging Face Spaces. Set HF_TOKEN (and HF_USERNAME above) and "
            "re-run this script to deploy the app."
        )
        return

    api = HfApi(token=hf_token)

    api.create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=DEPLOYMENT_DIR,
        repo_id=SPACE_REPO_ID,
        repo_type="space",
    )

    print(f"Deployment files pushed to https://huggingface.co/spaces/{SPACE_REPO_ID}")


if __name__ == "__main__":
    main()
