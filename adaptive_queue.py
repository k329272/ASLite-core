"""
Adaptive gloss queue.

Raw per-frame ASL predictions arrive fast and noisy: the same sign repeats
across many consecutive frames, and there's no explicit "end of sentence"
marker. This module buffers incoming gloss tokens and decides *when* to
flush a chunk to the translator, using a pause threshold that adapts to
how quickly the signer is actually signing:

  - Consecutive duplicate tokens are collapsed (a held sign shouldn't become
    "HELLO HELLO HELLO" in the gloss sequence fed to T5).
  - The flush pause threshold shrinks for a fast signer and grows for a slow
    one, based on a rolling average of recent inter-token gaps.
  - Hard ceilings (max_buffer_size, max_wait_seconds) guarantee the buffer
    is never held indefinitely, even if the pause condition never fires.

Runs its own background thread: tokens go in via `push()`, flushed gloss
sequences come out on `output_queue`.
"""

import queue
import threading
import time
from collections import deque
from typing import List, Optional

from config import AdaptiveQueueConfig


class AdaptiveGlossQueue:
    def __init__(
        self, cfg: AdaptiveQueueConfig, output_queue: "queue.Queue[List[str]]"
    ):
        self.cfg = cfg
        self.output_queue = output_queue

        self._input_queue: "queue.Queue[str]" = queue.Queue()
        self._buffer: List[str] = []
        self._gap_history: deque = deque(maxlen=cfg.gap_history_len)
        self._last_token_time: Optional[float] = None
        self._buffer_started_at: Optional[float] = None

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._input_queue.put(None)  # unblock the loop
        self._thread.join(timeout=2)

    def push(self, token: str):
        """Call from the ASL recognizer thread whenever a confident token arrives."""
        self._input_queue.put(token)

    def _current_pause_threshold(self) -> float:
        if not self._gap_history:
            return self.cfg.base_pause_seconds

        avg_gap = sum(self._gap_history) / len(self._gap_history)
        # Fast signing (small avg_gap) -> smaller factor -> shorter pause needed.
        # Slow signing (large avg_gap) -> larger factor -> more patience before flushing.
        factor = min(
            max(avg_gap / self.cfg.base_pause_seconds, self.cfg.min_adaptive_factor),
            self.cfg.max_adaptive_factor,
        )
        return self.cfg.base_pause_seconds * factor

    def _flush(self):
        if not self._buffer:
            return
        self.output_queue.put(list(self._buffer))
        self._buffer.clear()
        self._buffer_started_at = None

    def _run(self):
        while not self._stop_event.is_set():
            timeout = self._current_pause_threshold()
            try:
                item = self._input_queue.get(timeout=timeout)
            except queue.Empty:
                # No new token arrived within the pause threshold -> sentence boundary.
                self._flush()
                continue

            if item is None:
                break

            now = time.time()
            if self._last_token_time is not None:
                self._gap_history.append(now - self._last_token_time)
            self._last_token_time = now

            if self.cfg.dedup_repeats and self._buffer and self._buffer[-1] == item:
                pass  # collapse held/repeated sign
            else:
                self._buffer.append(item)
                if self._buffer_started_at is None:
                    self._buffer_started_at = now

            # Hard ceilings so a buffer never grows or waits forever.
            if len(self._buffer) >= self.cfg.max_buffer_size:
                self._flush()
            elif (
                self._buffer_started_at is not None
                and now - self._buffer_started_at >= self.cfg.max_wait_seconds
            ):
                self._flush()

        self._flush()  # drain on shutdown
