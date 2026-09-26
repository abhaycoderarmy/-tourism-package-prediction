"""
Hosting - verification only
-----------------------------
Hugging Face Spaces on the free cpu-basic tier now requires a PRO
subscription for Docker/Gradio SDKs, so this project's frontend is hosted
on Streamlit Community Cloud instead (connected directly to this GitHub
repo, auto-redeploying on every push to main). This step just confirms
the deployment files are all present.
"""

import os

DEPLOYMENT_DIR = "deployment"


def main():
    required = ["app.py", "requirements.txt", "Dockerfile", "best_tourism_model_v1.joblib"]
    missing = [f for f in required if not os.path.exists(os.path.join(DEPLOYMENT_DIR, f))]

    if missing:
        print(f"Missing deployment files: {missing}")
        raise SystemExit(1)

    print("All deployment files present and verified.")
    print("Live frontend hosted on Streamlit Community Cloud (auto-redeploys on push to main).")


if __name__ == "__main__":
    main()