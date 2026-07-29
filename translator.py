"""
Continuously re-translates the growing gloss buffer with T5 (the model
your notebook fine-tuned and quantized), honoring already-spoken (locked)
words as a fixed prefix the model isn't allowed to change.

Mechanism: on every new gloss token, we re-tokenize the *whole* buffer and
generate again, but force generation to start from the locked words by
passing them in as `decoder_input_ids`. T5 then only gets to choose new
wording for the tail beyond what's already been spoken.
"""

import logging
import threading
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import TranslatorConfig
from streaming_state import LatestSlot, SharedSentenceState

logger = logging.getLogger(__name__)


class GlossTranslator:
    def __init__(
        self,
        cfg: TranslatorConfig,
        retranslate_slot: LatestSlot,
        sentence_state: SharedSentenceState,
    ):
        self.cfg = cfg
        self.retranslate_slot = retranslate_slot
        self.sentence_state = sentence_state

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_path)
        self.model.to(cfg.device)
        self.model.eval()

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.retranslate_slot.put(("stop", None))
        self._thread.join(timeout=5)

    def _forced_decoder_ids(self, locked_words: List[str]) -> Optional[torch.Tensor]:
        if not locked_words:
            return None
        text = " ".join(locked_words)
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        decoder_start = self.model.config.decoder_start_token_id
        return torch.tensor([[decoder_start] + ids], device=self.cfg.device)

    def _retranslate(self, gloss_tokens: List[str]):
        if not gloss_tokens:
            return

        # Read the locked prefix fresh each pass -- it may have grown since
        # the last retranslation while TTS kept speaking in parallel.
        locked_words = self.sentence_state.get_locked_words()

        inputs = self.tokenizer(
            " ".join(gloss_tokens),
            max_length=self.cfg.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.cfg.device)

        gen_kwargs = dict(max_new_tokens=self.cfg.max_new_tokens, num_beams=self.cfg.num_beams)
        forced_ids = self._forced_decoder_ids(locked_words)
        if forced_ids is not None:
            gen_kwargs["decoder_input_ids"] = forced_ids

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        words = full_text.split()

        # Belt-and-suspenders: guarantee the locked prefix survives verbatim
        # even if decoding drifted (e.g. subword/spacing quirks).
        if words[: len(locked_words)] != locked_words:
            words = locked_words + words[len(locked_words):]

        self.sentence_state.set_candidate(words)
        self.sentence_state.record_translated_token_count(len(gloss_tokens))

    def _run(self):
        while not self._stop_event.is_set():
            msg = self.retranslate_slot.get(timeout=0.5)
            if msg is None:
                continue
            kind, payload = msg
            if kind == "stop":
                break
            try:
                if kind == "tokens":
                    self._retranslate(payload)
                elif kind == "sentence_end":
                    # No more gloss tokens will arrive for this sentence; the
                    # last "tokens" retranslation already produced the final
                    # candidate. The speaker finishes voicing it and triggers
                    # the reset once every word has been spoken.
                    continue
            except Exception:
                logger.exception("Retranslation failed; keeping previous candidate")
