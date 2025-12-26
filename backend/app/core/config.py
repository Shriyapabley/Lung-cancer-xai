# backend/app/core/config.py

import os
from functools import lru_cache


class Settings:
    PROJECT_NAME: str = "Lung Cancer CT XAI API"
    MODEL_CHECKPOINT: str = os.getenv(
        "CT_MODEL_PATH",
        # absolute path is safest; adjust to your real path
        r"C:\Users\shriy\OneDrive\Desktop\Lung_cancer_detectionV2\ml\checkpoints\ct_resnet18_best.pth",
    )
    NUM_CLASSES: int = int(os.getenv("CT_NUM_CLASSES", "4"))


@lru_cache()
def get_settings() -> Settings:
    return Settings()
