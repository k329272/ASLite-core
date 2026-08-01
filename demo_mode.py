import os
import tempfile
from typing import Optional

import cv2

from config import PipelineConfig
from tts_engine import KittenTTSSpeaker


def _write_caption_video(
    input_path: str, output_path: str, caption: str, audio_path: Optional[str] = None
) -> None:
    """Render a caption and optional audio onto a copied video file using OpenCV."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to read input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        raise RuntimeError("Unable to determine video dimensions")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError("Unable to create output video writer")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            overlay = frame.copy()
            cv2.putText(
                overlay,
                caption,
                (20, max(30, height - 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            out.write(overlay)
    finally:
        cap.release()
        out.release()


def run_video_demo(input_path: str, output_path: str, text: str, caption: str) -> str:
    """Create a demo video by replacing or augmenting the original audio with TTS output."""
    cfg = PipelineConfig()
    speaker = KittenTTSSpeaker(cfg.tts, None, None)  # type: ignore[arg-type]

    audio_bytes = speaker.synthesize_text_to_wav(text)
    if not audio_bytes:
        raise RuntimeError("Unable to synthesize demo audio")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_audio.write(audio_bytes)
        tmp_audio_path = tmp_audio.name

    try:
        _write_caption_video(input_path, output_path, caption, tmp_audio_path)
    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

    return output_path
