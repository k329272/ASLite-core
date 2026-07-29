"""
Lightweight, fully-offline text-to-speech using pyttsx3 (wraps SAPI5 on
Windows, NSSpeechSynthesizer on macOS, espeak on Linux). No model download,
low latency, minimal CPU/RAM footprint -- appropriate for the tail end of a
real-time pipeline.

Runs its own thread pulling finished sentences off a queue and speaking them
one at a time, so playback never overlaps.
"""

import queue
import threading

import pyttsx3

from config import TTSConfig


def list_voices():
    """Utility: run this once to see available voice ids/names on your machine."""
    engine = pyttsx3.init()
    for v in engine.getProperty("voices"):
        print(v.id, "-", v.name)
    engine.stop()


class TTSSpeaker:
    def __init__(self, cfg: TTSConfig, input_queue: "queue.Queue[str]"):
        self.cfg = cfg
        self.input_queue = input_queue

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", cfg.rate)
        self.engine.setProperty("volume", cfg.volume)
        if cfg.voice_id:
            self.engine.setProperty("voice", cfg.voice_id)

        self.last_spoken_text = ""  # readable by other threads for UI display only

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.input_queue.put(None)
        self._thread.join(timeout=5)

    def _run(self):
        while not self._stop_event.is_set():
            text = self.input_queue.get()
            if text is None:
                break
            self.last_spoken_text = text
            self.engine.say(text)
            self.engine.runAndWait()
