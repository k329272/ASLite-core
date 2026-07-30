"""Simple live server for streaming ASL recognition text and synthesized audio."""

import base64
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2

from config import PipelineConfig
from pipeline import ASLSpeechPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


class LiveStreamState:
    """Shared state for the live server and camera loop."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_text = ""
        self.latest_audio_b64 = ""
        self.latest_audio_sample_rate = 16000
        self.latest_frame = None
        self.running = True

    def update(self, text: str, audio_payload: bytes, sample_rate: int, frame) -> None:
        with self.lock:
            self.latest_text = text
            if audio_payload:
                self.latest_audio_b64 = base64.b64encode(audio_payload).decode("ascii")
            else:
                self.latest_audio_b64 = ""
            self.latest_audio_sample_rate = sample_rate
            self.latest_frame = frame


class LiveRequestHandler(BaseHTTPRequestHandler):
    """Serve a simple JSON payload with the latest transcript and audio."""

    server_version = "ASLiteLive/1.0"

    def do_GET(self):
        if self.path != "/":
            self.send_error(404, "Not Found")
            return

        state: LiveStreamState = self.server.state
        with state.lock:
            payload = {
                "text": state.latest_text,
                "audio_base64": state.latest_audio_b64,
                "audio_sample_rate": state.latest_audio_sample_rate,
            }
            body = json.dumps(payload).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        logger.debug("%s - - [%s] %s", self.address_string(), self.log_date_time_string(), format % args)


class LiveHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, state: LiveStreamState):
        super().__init__(server_address, handler_class)
        self.state = state


def _shutdown_resources(cap, pipeline, server, state) -> None:
    """Stop the camera, pipeline, and HTTP server cleanly."""
    state.running = False
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    if pipeline is not None:
        pipeline.stop()
    if server is not None:
        server.shutdown()
        server.server_close()


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the live server and read frames from the webcam."""
    state = LiveStreamState()
    server = LiveHTTPServer((host, port), LiveRequestHandler, state)

    cfg = PipelineConfig()
    pipeline = ASLSpeechPipeline(cfg)
    pipeline.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    logger.info("Live server listening on http://%s:%s/", host, port)

    try:
        while state.running:
            ok, frame = cap.read()
            if not ok:
                break

            try:
                prediction = pipeline.on_frame(frame)
            except (RuntimeError, ValueError, TypeError, AttributeError):
                logger.exception("Error processing frame; skipping it")
                prediction = None

            text = pipeline.last_spoken_text or ""
            if prediction is not None and prediction.token is not None:
                text = prediction.token
            if text:
                state.update(text, pipeline.last_audio_payload, pipeline.last_audio_sample_rate, frame)
            else:
                state.update(text, b"", 16000, frame)

            time.sleep(0.03)
    finally:
        _shutdown_resources(cap, pipeline, server, state)


if __name__ == "__main__":
    run_server()
