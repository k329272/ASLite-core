"""
Word-by-word speech synthesis using TinyTTS -- a ~1.6M-parameter ONNX
neural TTS model (github.com/tronghieuit/tiny-tts, `pip install tiny-tts`),
chosen for "lightweight NN-based TTS" since it's a genuine small neural
model rather than a rule-based engine, but still runs comfortably on CPU
with no GPU requirement.

As each word is spoken it's locked via SharedSentenceState.lock_next_word(),
so the translator can never revise it in a later re-translation pass. Once
every candidate word for the sentence has been spoken and no more signs are
coming, the whole pipeline state is cleared and the next sentence starts
fresh.

NOTE: tiny-tts is a small, actively-changing project. The constructor/
method signature below is inferred from its published CLI
(`tiny-tts --text ... --checkpoint G.pth --speaker MALE --speed 1.0 --device cpu`)
and README examples. Check `pip show tiny-tts` / the installed package's
actual Python API and adjust `_TinyTTSEngine(...)` / `.speak(...)` kwargs
if they differ in your installed version.
"""

import os
import tempfile
import threading
import time

import soundfile as sf
import sounddevice as sd
from tiny_tts import TinyTTS as _TinyTTSEngine

from config import TTSConfig
from streaming_state import SharedSentenceState, StreamingGlossBuffer


class TinyTTSSpeaker:
    def __init__(
        self,
        cfg: TTSConfig,
        sentence_state: SharedSentenceState,
        gloss_buffer: StreamingGlossBuffer,
    ):
        self.cfg = cfg
        self.sentence_state = sentence_state
        self.gloss_buffer = gloss_buffer

        self.engine = _TinyTTSEngine(
            checkpoint=cfg.tinytts_checkpoint,
            device=cfg.tinytts_device,
        )

        self.last_spoken_text = ""  # readable by other threads for UI display only
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5)

    def _speak_word(self, word: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = os.path.join(tmp_dir, "word.wav")
            self.engine.speak(
                word,
                output_path=wav_path,
                speaker=self.cfg.tinytts_speaker,
                speed=self.cfg.tinytts_speed,
            )
            audio, sample_rate = sf.read(wav_path, dtype="float32")
            sd.play(audio, sample_rate)
            sd.wait()

    def _run(self):
        while not self._stop_event.is_set():
            candidate_words, locked_count = self.sentence_state.get_candidate_and_locked_count()

            if locked_count < len(candidate_words):
                word = self.sentence_state.lock_next_word()
                if word:
                    self._speak_word(word)
                    self.last_spoken_text = " ".join(self.sentence_state.get_locked_words())
                continue

            if self.sentence_state.fully_spoken():
                # Every candidate word has been voiced and the sentence is
                # marked ended (the signer paused) -- reset everything so
                # the next sentence starts from a clean slate.
                self.sentence_state.reset()
                self.gloss_buffer.clear()
                self.last_spoken_text = ""
                continue

            time.sleep(0.02)  # nothing new to say yet; avoid a hot spin loop
