"""
Word-by-word speech synthesis using TinyTTS -- a ~1.6M-parameter ONNX
neural TTS model (github.com/tronghieuit/tiny-tts, `pip install tiny-tts`),
chosen for "lightweight NN-based TTS" since it's a genuine small neural
model rather than a rule-based engine, but still runs comfortably on CPU
with no GPU requirement.

Two threads, pipelined:
  - _decision_loop: decides which word to commit to next (requiring it to
    be "stable" across recent re-translations -- see SharedSentenceState.
    get_stable_next_word -- unless the sentence has ended and translation
    has caught up, in which case remaining words are final and spoken
    immediately), synthesizes its audio, and hands it off without waiting
    for playback to finish. This lets synthesis of word N+1 overlap with
    playback of word N instead of leaving dead air between every word.
  - _playback_loop: plays each synthesized word in order, and performs the
    end-of-sentence reset once a sentinel confirms every word for that
    sentence has actually been played.

Locking happens the moment a word is committed to the playback queue (not
the exact instant its audio starts), since from that point on it's
guaranteed to be spoken next in order and the translator must already
treat it as fixed.

NOTE: tiny-tts is a small, actively-changing project. The constructor/
method signature below is inferred from its published CLI
(`tiny-tts --text ... --checkpoint G.pth --speaker MALE --speed 1.0 --device cpu`)
and README examples. Check `pip show tiny-tts` / the installed package's
actual Python API and adjust `_TinyTTSEngine(...)` / `.speak(...)` kwargs
if they differ in your installed version.
"""

import logging
import os
import queue
import tempfile
import threading
import time
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import sounddevice as sd
from tiny_tts import TinyTTS as _TinyTTSEngine

from config import TTSConfig
from streaming_state import SharedSentenceState, StreamingGlossBuffer

logger = logging.getLogger(__name__)

_SENTENCE_DONE = object()  # sentinel: everything for the current sentence has been queued


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
        self._play_queue: "queue.Queue" = queue.Queue()
        self._stop_event = threading.Event()
        self._decision_thread = threading.Thread(target=self._decision_loop, daemon=True)
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)

    def start(self):
        self._decision_thread.start()
        self._playback_thread.start()

    def stop(self):
        self._stop_event.set()
        self._play_queue.put(None)  # unblock playback thread if it's waiting
        self._decision_thread.join(timeout=5)
        self._playback_thread.join(timeout=5)

    def _synthesize(self, word: str) -> Optional[Tuple[np.ndarray, int]]:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                wav_path = os.path.join(tmp_dir, "word.wav")
                self.engine.speak(
                    word,
                    output_path=wav_path,
                    speaker=self.cfg.tinytts_speaker,
                    speed=self.cfg.tinytts_speed,
                )
                audio, sample_rate = sf.read(wav_path, dtype="float32")
                return audio, sample_rate
        except Exception:
            logger.exception("TinyTTS synthesis failed for word %r; skipping it", word)
            return None

    def _decision_loop(self):
        while not self._stop_event.is_set():
            try:
                word = self.sentence_state.get_stable_next_word(self.cfg.min_word_stability)
                if word is not None:
                    locked = self.sentence_state.lock_next_word()
                    if locked is None:
                        continue  # lost a race with a concurrent reset; just retry
                    synth_result = self._synthesize(locked)
                    if synth_result is not None:
                        audio, sample_rate = synth_result
                        self._play_queue.put((locked, audio, sample_rate))
                    continue

                if self.sentence_state.fully_spoken():
                    self._play_queue.put(_SENTENCE_DONE)
                    # Block here until playback has actually drained this
                    # sentence and reset the shared state, so we don't spin
                    # re-detecting "fully_spoken" against a sentence that's
                    # already been cleared out from under us.
                    while not self._stop_event.is_set() and self.sentence_state.is_sentence_ended():
                        time.sleep(0.02)
                    continue

                time.sleep(0.02)  # nothing new is stable enough to commit to yet
            except Exception:
                logger.exception("Error in TTS decision loop; continuing")
                time.sleep(0.1)

    def _playback_loop(self):
        while not self._stop_event.is_set():
            item = self._play_queue.get()
            if item is None:
                break
            try:
                if item is _SENTENCE_DONE:
                    self.sentence_state.reset()
                    self.gloss_buffer.clear()
                    self.last_spoken_text = ""
                    continue

                word, audio, sample_rate = item
                sd.play(audio, sample_rate)
                sd.wait()
                self.last_spoken_text = " ".join(self.sentence_state.get_locked_words())
            except Exception:
                logger.exception("Error in TTS playback loop; continuing")
