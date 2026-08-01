"""Build a captioned, narrated demo video from a source clip.

The docstring on the old version of `_write_caption_video` promised it
would burn in audio too, but OpenCV's VideoWriter can only write image
frames -- it has no audio track support -- so the `audio_path` argument
was silently ignored and the synthesized narration was thrown away. This
version keeps captioning and audio as two explicit steps: burn in the
caption with OpenCV, then mux the narration on with ffmpeg. If ffmpeg
isn't available, the caller still gets a valid (silent) captioned video
instead of a hard failure.
"""

import logging
import os
import shutil
import subprocess
import tempfile

import cv2

from config import PipelineConfig
from tts_engine import KittenTTSSpeaker

logger = logging.getLogger(__name__)


def _write_caption_video(input_path: str, output_path: str, caption: str) -> None:
    """Render a caption onto a copy of the input video using OpenCV."""
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
            cv2.putText(
                frame,
                caption,
                (20, max(30, height - 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            out.write(frame)
    finally:
        cap.release()
        out.release()


def _mux_audio(video_path: str, audio_path: str, output_path: str) -> bool:
    """Combine the captioned (silent) video with the narration track via
    ffmpeg. Returns False if ffmpeg is unavailable or the mux fails, in
    which case the caller should fall back to the silent video."""
    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not found; demo video will have no audio")
        return False
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-i",
                audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg mux failed (%s); demo video will have no audio", exc)
        return False


def run_video_demo(input_path: str, output_path: str, text: str, caption: str) -> str:
    """Create a demo video: caption burned in, narrated with synthesized TTS audio."""
    cfg = PipelineConfig()
    speaker = KittenTTSSpeaker(cfg.tts, None, None)  # type: ignore[arg-type]

    audio_bytes = speaker.synthesize_text_to_wav(text)
    if not audio_bytes:
        raise RuntimeError("Unable to synthesize demo audio")

    with tempfile.TemporaryDirectory() as tmp_dir:
        silent_video_path = os.path.join(tmp_dir, "silent.avi")
        audio_path = os.path.join(tmp_dir, "narration.wav")

        _write_caption_video(input_path, silent_video_path, caption)
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        if not _mux_audio(silent_video_path, audio_path, output_path):
            shutil.copyfile(silent_video_path, output_path)

    return output_path
