import os
from pathlib import Path

import streamlit as st
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HF_REPO_ID = "Shreyyy7026/automotive-patchcore-model"
HF_FILENAME = "model.ckpt"


def get_huggingface_token():
    """Get the Hugging Face token from environment or Streamlit Secrets."""

    token = os.getenv("HF_TOKEN")

    if not token:
        try:
            token = st.secrets["HF_TOKEN"]
        except (KeyError, FileNotFoundError):
            token = None

    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add it under "
            "Streamlit Cloud > App Settings > Secrets."
        )

    return token


def get_patchcore_checkpoint():
    """Use a local checkpoint or download it from private Hugging Face."""

    # 1. Check an explicitly configured local checkpoint.
    checkpoint_env = os.getenv("PATCHCORE_CHECKPOINT")

    if checkpoint_env:
        checkpoint = Path(checkpoint_env).expanduser()

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Configured checkpoint does not exist: {checkpoint}"
            )

        return checkpoint

    # 2. Check the default local checkpoint.
    local_checkpoint = (
        PROJECT_ROOT / "models" / "patchcore" / "model.ckpt"
    )

    if local_checkpoint.is_file():
        return local_checkpoint

    # 3. Download the checkpoint from Hugging Face.
    token = get_huggingface_token()

    try:
        checkpoint_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=token,
        )

        checkpoint = Path(checkpoint_path)

        if not checkpoint.is_file():
            raise FileNotFoundError(
                "Downloaded checkpoint file was not found."
            )

        return checkpoint

    except Exception as exc:
        raise RuntimeError(
            "Failed to download PatchCore checkpoint from Hugging Face. "
            "Check the repository, filename, and HF_TOKEN permissions. "
            f"Details: {exc}"
        ) from exc