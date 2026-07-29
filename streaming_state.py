"""
Continuous streaming buffer + locked-word state for simultaneous
sign-to-speech translation.

Unlike a "buffer until pause, then translate once" design, T5 now
re-translates the *entire* gloss buffer every time a new sign arrives, so
the candidate sentence keeps improving as more context comes in. But
words TTS has already spoken are locked: once TinyTTS has said a word out
loud, it can never be un-said, so T5 is forced to keep that word fixed in
every subsequent re-translation and may only edit the still-unspoken tail.

A background thread watches for a pause in incoming signs (using the same
adaptive idea as before: the pause threshold shrinks/grows with the
signer's recent pace) and marks the sentence as ended. Once TTS finishes
speaking every locked word, the whole buffer + locked state is cleared and
the next sentence starts from scratch.
"""

import logging
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

from config import AdaptiveQueueConfig

logger = logging.getLogger(__name__)


class LatestSlot:
    """A single-slot mailbox: only the most recent value matters, and a
    consumer blocks until a new one arrives. Used so the translator always
    works on the freshest gloss buffer instead of a backlog of stale ones."""

    def __init__(self):
        self._cond = threading.Condition()
        self._value = None
        self._has_value = False

    def put(self, value):
        with self._cond:
            self._value = value
            self._has_value = True
            self._cond.notify_all()

    def get(self, timeout: Optional[float] = None):
        with self._cond:
            got = self._cond.wait_for(lambda: self._has_value, timeout=timeout)
            if not got:
                return None
            value = self._value
            self._has_value = False
            return value


class SharedSentenceState:
    """Thread-safe holder for the current sentence's locked (already
    spoken) words and the translator's latest full candidate sentence."""

    def __init__(self):
        self._lock = threading.Lock()
        self.locked_words: List[str] = []
        self.candidate_words: List[str] = []
        self.sentence_ended = False  # True once no more new signs are coming

        # Lets the speaker know the translator has actually processed the
        # most recent gloss token before declaring a sentence fully spoken
        # (otherwise "fully spoken" could fire one token too early, while
        # the translator is still mid-inference on the latest buffer).
        self.expected_token_count = 0
        self.translated_token_count = 0

        # Tracks how many consecutive re-translations produced the same
        # word as the current tail (first unlocked) position, so the
        # speaker can require a word to be "stable" before committing to it.
        self._tail_stability_word: Optional[str] = None
        self._tail_stability_count = 0

    def record_expected_token_count(self, n: int):
        with self._lock:
            self.expected_token_count = n

    def record_translated_token_count(self, n: int):
        with self._lock:
            self.translated_token_count = n

    def set_candidate(self, words: List[str]):
        with self._lock:
            n = len(self.locked_words)
            # Safety net: a retranslation must never alter an already-spoken word,
            # even if tokenizer round-tripping drifted slightly.
            if words[:n] != self.locked_words:
                words = self.locked_words + words[n:]
            self.candidate_words = words

            tail_word = words[n] if n < len(words) else None
            if tail_word == self._tail_stability_word:
                self._tail_stability_count += 1
            else:
                self._tail_stability_word = tail_word
                self._tail_stability_count = 1

    def get_candidate_and_locked_count(self) -> Tuple[List[str], int]:
        with self._lock:
            return list(self.candidate_words), len(self.locked_words)

    def get_stable_next_word(self, min_stability: int) -> Optional[str]:
        """Returns the next unlocked word iff it's safe to commit to: either
        it's survived `min_stability` consecutive re-translations unchanged,
        or the sentence has ended and translation has caught up (final
        drain -- nothing more will change it, so there's no reason to wait)."""
        with self._lock:
            n = len(self.locked_words)
            if n >= len(self.candidate_words):
                return None
            word = self.candidate_words[n]
            caught_up = self.translated_token_count >= self.expected_token_count
            if self.sentence_ended and caught_up:
                return word
            if self._tail_stability_count >= min_stability:
                return word
            return None

    def lock_next_word(self) -> Optional[str]:
        """Call right after TTS actually commits to saying the next word."""
        with self._lock:
            n = len(self.locked_words)
            if n < len(self.candidate_words):
                word = self.candidate_words[n]
                self.locked_words.append(word)
                # Reset stability tracking for the new tail position.
                self._tail_stability_word = None
                self._tail_stability_count = 0
                return word
            return None

    def get_locked_words(self) -> List[str]:
        with self._lock:
            return list(self.locked_words)

    def mark_sentence_ended(self):
        with self._lock:
            self.sentence_ended = True

    def is_sentence_ended(self) -> bool:
        with self._lock:
            return self.sentence_ended

    def fully_spoken(self) -> bool:
        with self._lock:
            return (
                self.sentence_ended
                and self.translated_token_count >= self.expected_token_count
                and len(self.locked_words) == len(self.candidate_words)
            )

    def reset(self):
        with self._lock:
            self.locked_words = []
            self.candidate_words = []
            self.sentence_ended = False
            self.expected_token_count = 0
            self.translated_token_count = 0
            self._tail_stability_word = None
            self._tail_stability_count = 0


class StreamingGlossBuffer:
    """Accumulates gloss tokens for the current sentence and detects when
    the signer has paused long enough to call it a sentence boundary.
    Every accepted token pushes the updated buffer onto `retranslate_slot`
    so the translator immediately re-runs T5 on it."""

    def __init__(
        self,
        cfg: AdaptiveQueueConfig,
        retranslate_slot: LatestSlot,
        sentence_state: SharedSentenceState,
    ):
        self.cfg = cfg
        self.retranslate_slot = retranslate_slot
        self.sentence_state = sentence_state

        self._tokens: List[str] = []
        self._gap_history: deque = deque(maxlen=cfg.gap_history_len)
        self._last_token_time: Optional[float] = None
        self._sentence_started_at: Optional[float] = None

        self._stop_event = threading.Event()
        self._new_token_event = threading.Event()
        self._thread = threading.Thread(target=self._watch_pauses, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._new_token_event.set()
        self._thread.join(timeout=2)

    def push(self, token: str):
        now = time.time()
        if self._last_token_time is not None:
            self._gap_history.append(now - self._last_token_time)
        self._last_token_time = now

        if self.cfg.dedup_repeats and self._tokens and self._tokens[-1] == token:
            return  # collapse a held/repeated sign into one token

        self._tokens.append(token)
        if self._sentence_started_at is None:
            self._sentence_started_at = now

        self.sentence_state.record_expected_token_count(len(self._tokens))
        self.retranslate_slot.put(("tokens", list(self._tokens)))
        self._new_token_event.set()

        if len(self._tokens) >= self.cfg.max_buffer_size:
            self._end_sentence()

    def clear(self):
        """Call once the current sentence has been fully spoken, to start
        the next one from a clean slate."""
        self._tokens = []
        self._gap_history.clear()
        self._last_token_time = None
        self._sentence_started_at = None

    def _current_pause_threshold(self) -> float:
        if not self._gap_history:
            return self.cfg.base_pause_seconds
        avg_gap = sum(self._gap_history) / len(self._gap_history)
        factor = min(
            max(avg_gap / self.cfg.base_pause_seconds, self.cfg.min_adaptive_factor),
            self.cfg.max_adaptive_factor,
        )
        return self.cfg.base_pause_seconds * factor

    def _end_sentence(self):
        self.sentence_state.mark_sentence_ended()
        self.retranslate_slot.put(("sentence_end", None))

    def _watch_pauses(self):
        while not self._stop_event.is_set():
            try:
                self._new_token_event.wait(timeout=0.2)
                self._new_token_event.clear()

                if self._sentence_started_at is None or self.sentence_state.is_sentence_ended():
                    continue

                pause_threshold = self._current_pause_threshold()
                elapsed_since_last = time.time() - (self._last_token_time or time.time())
                elapsed_since_start = time.time() - self._sentence_started_at

                if (
                    elapsed_since_last >= pause_threshold
                    or elapsed_since_start >= self.cfg.max_wait_seconds
                ):
                    self._end_sentence()
            except Exception:
                logger.exception("Error in gloss-buffer pause watcher; continuing")
