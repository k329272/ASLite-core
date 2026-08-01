"""
ASL sign recognition using OpenVINO Runtime.

Two input modes are supported (set in config.ASLConfig.input_mode):
  - "landmarks": model expects a flattened MediaPipe hand-landmark vector
                 (typically 21 landmarks * 3 coords * up to 2 hands = 126 floats).
                 This is what most lightweight real-time ASL classifiers use,
                 since it's far cheaper than running a CNN over raw frames.
  - "frame":     model expects a resized/normalized RGB frame (a small CNN).

Swap in your actual compiled model + label file via config.py. This class only
assumes a single input / single output classification model; adjust
`_infer` if your model's signature differs (e.g. multiple inputs).
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2
import openvino as ov
import mediapipe as mp
from config import ASLConfig

logger = logging.getLogger(__name__)


@dataclass
class ASLPrediction:
    """Container for one inference result and its confidence."""

    token: Optional[str]  # gloss token, or None if below confidence threshold
    confidence: float
    timestamp: float


class ASLRecognizer:
    """Run per-frame sign classification with a compiled OpenVINO model."""

    def __init__(self, cfg: ASLConfig):
        self.cfg = cfg
        self._min_interval = 1.0 / cfg.inference_fps
        self._last_infer_time = 0.0

        # Stability-smoothing state: a sign must be the top prediction for
        # `cfg.min_stable_frames` consecutive inference passes before it's
        # accepted and returned as a new token.
        self._pending_label: Optional[str] = None
        self._pending_count = 0
        self._confirmed_label: Optional[str] = None

        if np is None or cv2 is None or ov is None:
            raise RuntimeError(
                "ASL recognizer requires numpy, opencv-python, and openvino"
            )

        with open(cfg.labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        core = ov.Core()
        model = core.read_model(cfg.model_xml)
        self.compiled_model = core.compile_model(model, cfg.device)
        self.output_layer = self.compiled_model.output(0)
        self.input_layer = self.compiled_model.input(0)
        input_shape = [int(dim) for dim in self.input_layer.shape]
        self._expects_clip = len(input_shape) == 5
        self._clip_len = input_shape[2] if self._expects_clip else 1
        self._clip_frames = deque(maxlen=self._clip_len)

        if cfg.input_mode == "landmarks":
            if mp is None:
                raise RuntimeError("mediapipe is required for landmark input mode")

            self._mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def _extract_landmarks(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mp_hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        vecs = []
        for hand_landmarks in result.multi_hand_landmarks[:2]:
            for lm in hand_landmarks.landmark:
                vecs.extend([lm.x, lm.y, lm.z])

        # Pad to a fixed length (2 hands * 21 landmarks * 3 coords = 126) so the
        # model always sees a consistent shape even with one hand visible.
        target_len = 126
        if len(vecs) < target_len:
            vecs.extend([0.0] * (target_len - len(vecs)))
        return np.array(vecs[:target_len], dtype=np.float32)

    def _prep_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_bgr, self.cfg.frame_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        return chw

    def _infer(self, input_tensor: np.ndarray) -> ASLPrediction:
        if self._expects_clip:
            self._clip_frames.append(input_tensor)
            while len(self._clip_frames) < self._clip_len:
                self._clip_frames.append(input_tensor)
            model_input = np.stack(list(self._clip_frames), axis=1)
            batched = np.expand_dims(model_input, 0)
        else:
            batched = np.expand_dims(input_tensor, 0)
        logits = self.compiled_model([batched])[self.output_layer]
        probs = self._softmax(logits[0])
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        now = time.time()

        if conf < self.cfg.confidence_threshold:
            return ASLPrediction(token=None, confidence=conf, timestamp=now)
        return ASLPrediction(token=self.labels[idx], confidence=conf, timestamp=now)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def _stabilize(self, raw: ASLPrediction) -> ASLPrediction:
        """Only lets a prediction through as a real token once the same
        label has been the top prediction for several consecutive passes,
        and only once per stable run (not on every subsequent frame)."""
        if raw.token is None:
            self._pending_label = None
            self._pending_count = 0
            return raw

        if raw.token == self._pending_label:
            self._pending_count += 1
        else:
            self._pending_label = raw.token
            self._pending_count = 1
        if (
            raw.token == self._confirmed_label
            or self._pending_count < self.cfg.min_stable_frames
        ):
            # Still holding the same already-accepted sign -- nothing new to emit.
            return ASLPrediction(
                token=None, confidence=raw.confidence, timestamp=raw.timestamp
            )

        self._confirmed_label = raw.token
        return raw

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[ASLPrediction]:
        """Call once per captured video frame. Internally throttled to
        cfg.inference_fps and returns None on frames that are skipped or
        where no sign was detected/confident/stable yet."""
        now = time.time()
        if now - self._last_infer_time < self._min_interval:
            return None
        self._last_infer_time = now
        if self.cfg.input_mode == "landmarks":
            vec = self._extract_landmarks(frame_bgr)
            if vec is None:
                self._pending_label = None
                self._pending_count = 0
                return None
            raw = self._infer(vec)
        else:
            tensor = self._prep_frame(frame_bgr)
            raw = self._infer(tensor)
        return self._stabilize(raw)
