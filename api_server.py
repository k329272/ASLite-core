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

from asl_recognizer import ASLRecognizer, ASLPrediction
from config import PipelineConfig

ConfidenceLevel = Literal["high", "medium", "low"]


class FrameRequest(BaseModel):
    image_base64: str | None = None
    frames_base64: list[str] | None = None


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

_recognizer: Optional[ASLRecognizer] = None
_latest = RecognitionResult(
    id="startup",
    text="Camera ready. Start a scan to recognize a sign.",
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


def get_recognizer() -> ASLRecognizer:
    global _recognizer
    if _recognizer is None:
        cfg = PipelineConfig().asl
        cfg.input_mode = "frame"
        cfg.min_stable_frames = 1
        cfg.confidence_threshold = 0.2
        _recognizer = ASLRecognizer(cfg)
    return _recognizer


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


def result_from_prediction(prediction: ASLPrediction | None) -> RecognitionResult:
    if prediction is None or prediction.token is None:
        return RecognitionResult(
            id=f"aslite-{uuid.uuid4().hex[:10]}",
            text="No clear sign detected yet.",
            confidence=0,
            confidenceLevel="low",
            alternatives=[],
            requiresConfirmation=True,
            detectedSigns=[],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    confidence = float(prediction.confidence)
    level = confidence_level(confidence)
    label = prediction.token.replace("_", " ")
    return RecognitionResult(
        id=f"aslite-{uuid.uuid4().hex[:10]}",
        text=label,
        confidence=confidence,
        confidenceLevel=level,
        alternatives=[],
        requiresConfirmation=level != "high",
        detectedSigns=[
            DetectedSign(
                label=label,
                confidence=confidence,
                startTime=prediction.timestamp,
                endTime=prediction.timestamp,
            )
        ],
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": _recognizer is not None}


@app.post("/recognize-frame", response_model=RecognitionResult)
def recognize_frame(request: FrameRequest):
    global _latest
    recognizer = get_recognizer()
    frames = request.frames_base64 or ([request.image_base64] if request.image_base64 else [])
    if not frames:
        raise HTTPException(status_code=400, detail="No frame payload provided")

    prediction: ASLPrediction | None = None
    for frame_payload in frames:
        frame = decode_frame(frame_payload)
        next_prediction = recognizer.process_frame(frame)
        if next_prediction is not None:
            prediction = next_prediction

    _latest = result_from_prediction(prediction)
    return _latest


@app.get("/translation/latest", response_model=RecognitionResult)
def latest_translation():
    return _latest
