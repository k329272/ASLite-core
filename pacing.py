"""
Shared pacing helper for gloss-buffering queues.

Both AdaptiveGlossQueue (adaptive_queue.py) and StreamingGlossBuffer
(streaming_state.py) need to decide, from a rolling window of recent
inter-token gaps, how long a pause currently counts as "the signer
stopped." They used identical logic; it now lives in one place.
"""

from collections import deque

from config import AdaptiveQueueConfig


def adaptive_pause_threshold(gap_history: deque, cfg: AdaptiveQueueConfig) -> float:
    """Return the pause (seconds) that currently counts as a sentence
    boundary, based on the signer's recent pace. Shrinks for a fast
    signer and grows for a slow one, clamped to
    [min_adaptive_factor, max_adaptive_factor] * base_pause_seconds.
    """
    if not gap_history:
        return cfg.base_pause_seconds

    avg_gap = sum(gap_history) / len(gap_history)
    factor = min(
        max(avg_gap / cfg.base_pause_seconds, cfg.min_adaptive_factor),
        cfg.max_adaptive_factor,
    )
    return cfg.base_pause_seconds * factor
