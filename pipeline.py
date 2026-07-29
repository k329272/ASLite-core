"""
Wires the four stages together:

  camera frames
      -> ASLRecognizer (OpenVINO)                [per-frame gloss token / None]
      -> AdaptiveGlossQueue                       [buffers into sentence-like chunks]
      -> GlossTranslator (your fine-tuned T5)     [gloss chunk -> English sentence]
      -> TTSSpeaker (pyttsx3)                     [English sentence -> speech]

Each stage after the recognizer runs on its own thread and communicates via
queue.Queue, so a slow T5 generate() call never blocks frame capture, and TTS
playback never blocks translation.
"""

import queue

from config import PipelineConfig
from asl_recognizer import ASLRecognizer
from adaptive_queue import AdaptiveGlossQueue
from translator import GlossTranslator
from tts_engine import TTSSpeaker


class ASLSpeechPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

        self.gloss_chunk_queue: "queue.Queue" = queue.Queue()
        self.text_queue: "queue.Queue" = queue.Queue()

        self.recognizer = ASLRecognizer(cfg.asl)
        self.gloss_queue = AdaptiveGlossQueue(cfg.queue, self.gloss_chunk_queue)
        self.translator = GlossTranslator(cfg.translator, self.gloss_chunk_queue, self.text_queue)
        self.speaker = TTSSpeaker(cfg.tts, self.text_queue)

    def start(self):
        self.gloss_queue.start()
        self.translator.start()
        self.speaker.start()

    def stop(self):
        self.gloss_queue.stop()
        self.translator.stop()
        self.speaker.stop()

    @property
    def last_spoken_text(self) -> str:
        return self.speaker.last_spoken_text

    def on_frame(self, frame_bgr):
        """Call this once per captured video frame."""
        prediction = self.recognizer.process_frame(frame_bgr)
        if prediction is not None and prediction.token is not None:
            self.gloss_queue.push(prediction.token)
            return prediction
        return None
