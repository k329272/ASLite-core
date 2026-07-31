"""HTTP API adapter for hosted ASLite inference.

The original server.py reads from the server machine webcam. This adapter accepts
browser-captured frames over HTTPS so ASLite can run on a cloud host.
"""

from __future__ import annotations

import base64
import time
import uuid
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import PipelineConfig
from pipeline import ASLSpeechPipeline

ConfidenceLevel = Literal["high", "medium", "low"]


class FrameRequest(BaseModel):
    image_base64: str


class TranslationAlternative(BaseModel):
    id: str
    text: str
    confidence: float


class DetectedSign(BaseModel):
    label: str
    confidence: float
    startTime: float
    endTime: float


class RecognitionResult(BaseModel):
    id: str
    text: str
    confidence: float
    confidenceLevel: ConfidenceLevel
    alternatives: list[TranslationAlternative]
    requiresConfirmation: bool
    detectedSigns: list[DetectedSign]
    timestamp: str


app = FastAPI(title="ASLite Core API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Optional[ASLSpeechPipeline] = None
_latest = RecognitionResult(
    id="startup",
    text="",
    confidence=0,
    confidenceLevel="low",
    alternatives=[],
    requiresConfirmation=True,
    detectedSigns=[],
    timestamp="",
)


def confidence_level(confidence: float) -> ConfidenceLevel:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"


def get_pipeline() -> ASLSpeechPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ASLSpeechPipeline(PipelineConfig())
        _pipeline.start()
    return _pipeline


def decode_frame(image_base64: str):
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    try:
        payload = base64.b64decode(image_base64)
        array = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid base64 image") from exc
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return frame


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": _pipeline is not None}


@app.post("/recognize-frame", response_model=RecognitionResult)
def recognize_frame(request: FrameRequest):
    global _latest
    pipeline = get_pipeline()
    frame = decode_frame(request.image_base64)

    prediction = pipeline.on_frame(frame)
    text = pipeline.last_spoken_text or ""
    confidence = 0.0
    detected: list[DetectedSign] = []

    if prediction is not None and prediction.token is not None:
        text = prediction.token
        confidence = float(prediction.confidence)
        detected.append(
            DetectedSign(
                label=prediction.token,
                confidence=confidence,
                startTime=prediction.timestamp,
                endTime=prediction.timestamp,
            )
        )

    level = confidence_level(confidence)
    _latest = RecognitionResult(
        id=f"aslite-{uuid.uuid4().hex[:10]}",
        text=text,
        confidence=confidence,
        confidenceLevel=level,
        alternatives=[],
        requiresConfirmation=level != "high",
        detectedSigns=detected,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    return _latest


@app.get("/translation/latest", response_model=RecognitionResult)
def latest_translation():
    return _latest


@app.on_event("shutdown")
def shutdown():
    global _pipeline
    if _pipeline is not None:
        _pipeline.stop()
        _pipeline = None
