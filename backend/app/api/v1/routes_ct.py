# backend/app/api/v1/routes_ct.py

import os
import uuid
import base64
from typing import Tuple

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.schemas.ct_responses import CTPredictionResponse, CTExplainResponse
from app.models.ct_model_loader import predict_ct, explain_ct


router = APIRouter()


def _save_temp_file(upload: UploadFile) -> str:
    ext = os.path.splitext(upload.filename)[-1]
    tmp_name = f"tmp_{uuid.uuid4().hex}{ext}"
    tmp_dir = "tmp_uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, tmp_name)
    with open(tmp_path, "wb") as f:
        f.write(upload.file.read())
    return tmp_path


@router.post("/predict_ct", response_model=CTPredictionResponse)
async def predict_ct_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code=400, detail="Only PNG/JPEG images supported")

    tmp_path = _save_temp_file(file)
    pred_class, probs = predict_ct(tmp_path)
    os.remove(tmp_path)

    return CTPredictionResponse(predicted_class=pred_class, probabilities=probs)


@router.post("/explain_ct", response_model=CTExplainResponse)
async def explain_ct_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code=400, detail="Only PNG/JPEG images supported")

    tmp_path = _save_temp_file(file)
    pred_class, probs = predict_ct(tmp_path)

    cam_bgr = explain_ct(tmp_path)
    os.remove(tmp_path)

    # encode heatmap as base64
    import cv2
    import numpy as np

    success, buffer = cv2.imencode(".png", cam_bgr)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode heatmap")

    heatmap_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return CTExplainResponse(
        predicted_class=pred_class,
        probabilities=probs,
        heatmap_base64=heatmap_b64,
    )
