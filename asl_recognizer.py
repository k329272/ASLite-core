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
from typing import Optional, List, Union

import numpy as np
import cv2
import openvino as ov
from config import ASLConfig

logger = logging.getLogger(__name__)


@dataclass
class ASLPrediction:
    """Container for one inference result and its confidence."""

    token: Optional[str]  # gloss token, or None if below confidence threshold
    confidence: float
    timestamp: float


class ASLRecognizer:
    """Run action/gesture classification matching OpenVINO model zoo signature."""

    def __init__(self, cfg: ASLConfig):
        self.cfg = cfg
        self._min_interval = 1.0 / cfg.inference_fps
        self._last_infer_time = 0.0

        # Stability-smoothing state
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

        # Inspect model shape: expected shape is typically [B, C, T, H, W] or [B, T, C, H, W]
        input_shape = [int(dim) for dim in self.input_layer.shape]

        # Gesture recognition models expect 5D clip inputs [Batch, Channels, Temporal_Frames, Height, Width]
        if len(input_shape) == 5:
            self._clip_len = input_shape[2]
            self._target_height = input_shape[3]
            self._target_width = input_shape[4]
        else:
            self._clip_len = getattr(cfg, "clip_len", 16)
            self._target_height = cfg.frame_size[1]
            self._target_width = cfg.frame_size[0]

        # Queue to accumulate temporal frame window
        self._clip_frames = deque(maxlen=self._clip_len)

    def _prep_frame(
        self, frame_bgr: np.ndarray, roi: Optional[Union[List[int], np.ndarray]] = None
    ) -> np.ndarray:
        """Crops ROI (if provided), resizes, converts to RGB, and transposes to (C, H, W)."""
        if roi is not None and len(roi) == 4:
            x1, y1, x2, y2 = [int(v) for v in roi]
            # Ensure ROI boundaries remain inside frame limits
            h, w, _ = frame_bgr.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                frame_bgr = frame_bgr[y1:y2, x1:x2]

        resized = cv2.resize(frame_bgr, (self._target_width, self._target_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Transpose from (H, W, C) to (C, H, W)
        chw = np.transpose(rgb, (2, 0, 1))
        return chw

    def _infer(self, frame_tensor: np.ndarray) -> ASLPrediction:
        """Appends preprocessed frame to clip buffer and executes 5D tensor inference."""
        self._clip_frames.append(frame_tensor)

        # Fill buffer on startup if not fully populated yet
        while len(self._clip_frames) < self._clip_len:
            self._clip_frames.append(frame_tensor)

        # Stack temporal dimension: list of (C, H, W) -> (C, T, H, W)
        clip_tensor = np.stack(list(self._clip_frames), axis=1)

        # Add batch dimension -> (1, C, T, H, W)
        batched = np.expand_dims(clip_tensor, axis=0)

        # Execute OpenVINO inference pass
        logits = self.compiled_model([batched])[self.output_layer]
        probs = self._softmax(logits[0])
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        now = time.time()

        if conf < self.cfg.confidence_threshold:
            return ASLPrediction(token=None, confidence=conf, timestamp=now)

        label = (
            self.labels[idx]
            if isinstance(self.labels, list)
            else self.labels.get(str(idx), self.labels.get(idx))
        )
        return ASLPrediction(token=label, confidence=conf, timestamp=now)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def _stabilize(self, raw: ASLPrediction) -> ASLPrediction:
        """Filters out unstable predictions until top prediction remains stable over frames."""
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
            return ASLPrediction(
                token=None, confidence=raw.confidence, timestamp=raw.timestamp
            )

        self._confirmed_label = raw.token
        return raw

    def process_frame(
        self, frame_bgr: np.ndarray, roi: Optional[Union[List[int], np.ndarray]] = None
    ) -> Optional[ASLPrediction]:
        """Main entry point. Accepts frame and optional bounding box ROI."""
        now = time.time()
        if now - self._last_infer_time < self._min_interval:
            return None
        self._last_infer_time = now

        tensor = self._prep_frame(frame_bgr, roi=roi)
        raw = self._infer(tensor)
        return self._stabilize(raw)
