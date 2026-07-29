"""
Loads the gloss->English T5 model produced at the end of your training
notebook (the quantized + pruned model saved to `optimized_t5_model`) and
translates buffered gloss token sequences into natural English text.
"""

import threading
import queue
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import TranslatorConfig


class GlossTranslator:
    def __init__(
        self,
        cfg: TranslatorConfig,
        input_queue: "queue.Queue[List[str]]",
        output_queue: "queue.Queue[str]",
    ):
        self.cfg = cfg
        self.input_queue = input_queue
        self.output_queue = output_queue

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
        self.input_queue.put(None)
        self._thread.join(timeout=5)

    def translate(self, gloss_tokens: List[str]) -> str:
        gloss_text = " ".join(gloss_tokens)
        inputs = self.tokenizer(
            gloss_text,
            max_length=self.cfg.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.cfg.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                num_beams=self.cfg.num_beams,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _run(self):
        while not self._stop_event.is_set():
            item = self.input_queue.get()
            if item is None:
                break
            if not item:
                continue

            text = self.translate(item)
            if text.strip():
                self.output_queue.put(text)
