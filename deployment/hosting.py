"""
Hosting - push the Streamlit app to a Hugging Face Space
-----------------------------------------------------------
Uploads everything in tourism_project/deployment/ (app.py, Dockerfile,
requirements.txt, and the trained model file) to a public Hugging Face
Space so it goes live as the project's frontend.

Usage:
    python hosting.py

Requires:
    HF_TOKEN     - Hugging Face access token with "write" scope
"""

import os
from huggingface_hub import HfApi

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
HF_USERNAME = "abhayfps"          # <-- TODO: replace with your HF username
SPACE_REPO_ID = f"{HF_USERNAME}/tourism-package-prediction-app"

DEPLOYMENT_DIR = "tourism_project/deployment"


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
