"""
Real-time ASL -> speech demo.

Usage:
    python main.py
    python main.py --demo-video path/to/input.mp4 --demo-output path/to/output.mp4 --demo-text "Hello world"

Press 'q' to quit in webcam mode.
"""

import argparse
import logging

import cv2

from config import PipelineConfig
from demo_mode import run_video_demo
from pipeline import ASLSpeechPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    """Launch either the webcam demo or the video demo export workflow."""
    parser = argparse.ArgumentParser(description="ASL speech demo")
    parser.add_argument("--demo-video", help="Input video file for the demo export")
    parser.add_argument("--demo-output", help="Output video file for the demo export")
    parser.add_argument(
        "--demo-text",
        help="Text to synthesize for the video-demo export",
        default="",
    )
    parser.add_argument(
        "--demo-caption",
        help="Caption text to render over the exported video",
        default="",
    )
    args = parser.parse_args()

    if args.demo_video or args.demo_output:
        if not args.demo_video or not args.demo_output:
            raise ValueError("Both --demo-video and --demo-output are required")
        run_video_demo(
            args.demo_video,
            args.demo_output,
            args.demo_text or args.demo_caption or "Demo mode",
            args.demo_caption or args.demo_text or "Demo mode",
        )
        return

    cfg = PipelineConfig()
    pipeline = ASLSpeechPipeline(cfg)
    pipeline.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    last_token_shown = ""
    last_sentence_shown = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            try:
                prediction = pipeline.on_frame(frame)
            except (RuntimeError, ValueError, TypeError, AttributeError):
                logging.getLogger(__name__).exception(
                    "Error processing frame; skipping it"
                )
                prediction = None

            if prediction is not None:
                last_token_shown = f"{prediction.token} ({prediction.confidence:.2f})"
            if pipeline.last_spoken_text:
                last_sentence_shown = pipeline.last_spoken_text

            cv2.putText(
                frame,
                f"Sign: {last_token_shown}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Last: {last_sentence_shown}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )
            cv2.imshow("ASL -> Speech", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.stop()


if __name__ == "__main__":
    main()
