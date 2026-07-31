"""
Word-by-word speech synthesis using KittenTTS, a lightweight ONNX-based
text-to-speech engine designed to run comfortably on CPU.

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
"""

import io
import logging
import os
import queue
import tempfile
import threading
import time
import wave
from typing import Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional runtime dependency
    np = None  # type: ignore[assignment]

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional runtime dependency
    sf = None  # type: ignore[assignment]

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional runtime dependency
    sd = None  # type: ignore[assignment]

try:
    from kittentts import KittenTTS as _KittenTTSEngine
except ImportError:  # pragma: no cover - optional runtime dependency
    _KittenTTSEngine = None  # type: ignore[assignment]

from config import TTSConfig
from streaming_state import SharedSentenceState, StreamingGlossBuffer

logger = logging.getLogger(__name__)

_SENTENCE_DONE = (
    object()
)  # sentinel: everything for the current sentence has been queued


class KittenTTSSpeaker:
    """Speak translated words incrementally using KittenTTS."""

    def __init__(
        self,
        cfg: TTSConfig,
        sentence_state: SharedSentenceState,
        gloss_buffer: StreamingGlossBuffer,
    ):
        self.cfg = cfg
        self.sentence_state = sentence_state
        self.gloss_buffer = gloss_buffer

        if np is None or sf is None or sd is None or _KittenTTSEngine is None:
            raise RuntimeError(
                "KittenTTS support requires numpy, soundfile, sounddevice, and kittentts"
            )

        self._use_fallback_synth = False
        if _KittenTTSEngine is None:
            self.engine = None
            self._use_fallback_synth = True
        else:
            try:
                self.engine = _KittenTTSEngine(
                    getattr(self.cfg, "kittentts_model", getattr(self.cfg, "kitttentts_model", "")),
                    cache_dir=getattr(self.cfg, "kittentts_cache_dir", None),
                )
            except TypeError:
                try:
                    self.engine = _KittenTTSEngine(
                        getattr(self.cfg, "kittentts_model", getattr(self.cfg, "kitttentts_model", ""))
                    )
                except Exception as exc:  # pragma: no cover - runtime fallback path
                    logger.warning(
                        "KittenTTS initialization failed, using fallback synthesis: %s",
                        exc,
                    )
                    self.engine = None
                    self._use_fallback_synth = True
            except Exception as exc:  # pragma: no cover - runtime fallback path
                logger.warning(
                    "KittenTTS initialization failed, using fallback synthesis: %s",
                    exc,
                )
                self.engine = None
                self._use_fallback_synth = True

        self.last_spoken_text = ""  # readable by other threads for UI display only
        self.last_audio_payload = b""
        self.last_audio_sample_rate = 16000
        self.last_audio_word = ""
        self._play_queue: "queue.Queue" = queue.Queue()
        self._stop_event = threading.Event()
        self._decision_thread = threading.Thread(
            target=self._decision_loop, daemon=True
        )
        self._playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True
        )

    def start(self):
        """Start the decision and playback worker threads."""
        self._decision_thread.start()
        self._playback_thread.start()

    def stop(self):
        """Stop the decision and playback worker threads."""
        self._stop_event.set()
        self._play_queue.put(None)  # unblock playback thread if it's waiting
        self._decision_thread.join(timeout=5)
        self._playback_thread.join(timeout=5)

    def _fallback_synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Generate a simple deterministic waveform when the runtime TTS stack is unavailable."""
        if np is None:
            return np.asarray([], dtype=np.float32), 24000

        sample_rate = 24000
        duration = min(1.6, 0.18 + 0.04 * max(1, len(text.split())))
        total_samples = int(sample_rate * duration)
        t = np.linspace(0.0, duration, total_samples, endpoint=False)
        base_freq = 220.0 + (sum(ord(ch) for ch in text) % 100) * 2.0
        amplitude = 0.18 + 0.01 * min(10, len(text))
        envelope = np.sin(np.linspace(0.0, np.pi, total_samples)) ** 2
        audio = amplitude * np.sin(2.0 * np.pi * base_freq * t) * envelope
        return audio.astype(np.float32), sample_rate

    def _synthesize(self, word: str) -> Optional[Tuple[np.ndarray, int]]:
        """Synthesize one word to an in-memory audio buffer and return it."""
        if self._use_fallback_synth:
            return self._fallback_synthesize(word)

        try:
            audio = self.engine.generate(
                word,
                voice=getattr(self.cfg, "kittentts_voice", getattr(self.cfg, "kitttentts_voice", "Jasper")),
                speed=getattr(self.cfg, "kittentts_speed", getattr(self.cfg, "kitttentts_speed", 1.0)),
                clean_text=True,
            )
            audio_arr = np.asarray(audio, dtype=np.float32)
            return audio_arr, 24000
        except (RuntimeError, ValueError, TypeError, AttributeError):
            logger.exception(
                "KittenTTS synthesis failed for word %r; skipping it", word
            )
            return None

    def _to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Encode audio data as a WAV payload for streaming to clients."""
        if audio is None or np is None:
            return b""

        audio_arr = np.asarray(audio)
        if audio_arr.ndim == 2:
            channels = audio_arr.shape[1]
            samples = audio_arr
        else:
            channels = 1
            samples = audio_arr.reshape(-1, 1)

        if samples.dtype.kind in {"f", "c"}:
            samples = np.clip(samples, -1.0, 1.0)
            samples = (samples * 32767).astype(np.int16)
        elif samples.dtype != np.int16:
            samples = samples.astype(np.int16)

        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(int(sample_rate))
                wav_file.writeframes(samples.tobytes())
            return buf.getvalue()

    def synthesize_text_to_wav(self, text: str) -> bytes:
        """Synthesize a full phrase to WAV bytes for offline demos or export."""
        if not text or not text.strip():
            return b""

        synthesized = self._synthesize(text.strip())
        if synthesized is None:
            return b""

        audio, sample_rate = synthesized
        return self._to_wav_bytes(audio, sample_rate)

    def _decision_loop(self):
        """Decide which word to lock and queue for playback next."""
        while not self._stop_event.is_set():
            try:
                word = self.sentence_state.get_stable_next_word(
                    self.cfg.min_word_stability
                )
                if word is not None:
                    locked = self.sentence_state.lock_next_word()
                    if locked is None:
                        continue  # lost a race with a concurrent reset; just retry
                    synth_result = self._synthesize(locked)
                    if synth_result is not None:
                        audio, sample_rate = synth_result
                        self.last_audio_word = locked
                        self.last_audio_sample_rate = int(sample_rate)
                        self.last_audio_payload = self._to_wav_bytes(audio, sample_rate)
                        self._play_queue.put((locked, audio, sample_rate))
                    continue

                if self.sentence_state.fully_spoken():
                    self._play_queue.put(_SENTENCE_DONE)
                    # Block here until playback has actually drained this
                    # sentence and reset the shared state, so we don't spin
                    # re-detecting "fully_spoken" against a sentence that's
                    # already been cleared out from under us.
                    while (
                        not self._stop_event.is_set()
                        and self.sentence_state.is_sentence_ended()
                    ):
                        time.sleep(0.02)
                    continue

                time.sleep(0.02)  # nothing new is stable enough to commit to yet
            except (RuntimeError, ValueError, TypeError, AttributeError):
                logger.exception("Error in TTS decision loop; continuing")
                time.sleep(0.1)

    def _playback_loop(self):
        """Play queued audio words in order and reset state when a sentence drains."""
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

                _, audio, sample_rate = item
                sd.play(audio, sample_rate)
                sd.wait()
                self.last_spoken_text = " ".join(self.sentence_state.get_locked_words())
            except (RuntimeError, ValueError, TypeError, AttributeError):
                logger.exception("Error in TTS playback loop; continuing")


TinyTTSSpeaker = KittenTTSSpeaker
