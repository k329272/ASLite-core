# ASL → Speech Pipeline

Real-time pipeline: webcam → OpenVINO ASL sign recognition → adaptive
gloss-chunking queue → your fine-tuned T5 (gloss → English) → offline TTS.

```
camera frame
   │
   ▼
ASLRecognizer (OpenVINO)                  one gloss token per confident sign
   │
   ▼
AdaptiveGlossQueue                        buffers tokens into sentence-like chunks,
   │                                      flushing on a pause that adapts to signing speed
   ▼
GlossTranslator (T5, from your notebook)  gloss chunk -> English sentence
   │
   ▼
TTSSpeaker (pyttsx3)                      speaks each sentence, one at a time
```

Each stage after frame capture runs on its own thread and hands off through a
`queue.Queue`, so a slow translation or TTS call never stalls sign
recognition.

## Files

| File | Purpose |
|---|---|
| `config.py` | All tunables in one place — model paths, thresholds, timing |
| `asl_recognizer.py` | Loads your OpenVINO IR model, runs inference per frame |
| `adaptive_queue.py` | Buffers/dedups gloss tokens, adaptive pause-based flushing |
| `translator.py` | Loads `optimized_t5_model` (from your training notebook) and translates |
| `tts_engine.py` | Offline TTS via `pyttsx3` |
| `pipeline.py` | Wires the stages together |
| `main.py` | Webcam demo / entry point |

## Setup

```bash
pip install -r requirements.txt
```

1. **T5 model**: copy the `optimized_t5_model/` folder your notebook's last
   cell produces (`quantized_model.save_pretrained(...)` +
   `tokenizer.save_pretrained(...)`) into this project, or point
   `TranslatorConfig.model_path` at it.

2. **ASL model**: place your OpenVINO IR files (`.xml`/`.bin`) at the path in
   `ASLConfig.model_xml`, and a JSON array of gloss labels (index → token, in
   the same order as your model's output classes) at `ASLConfig.labels_path`.

3. **Input mode**: `ASLConfig.input_mode` controls what the recognizer feeds
   the model:
   - `"landmarks"` (default) — extracts MediaPipe hand landmarks per frame and
     feeds a 126-float vector (2 hands × 21 landmarks × xyz). Cheapest and
     most common approach for real-time sign classifiers.
   - `"frame"` — feeds a resized/normalized RGB frame directly, for CNN-style
     models. Adjust `frame_size` to match your model's expected input.

   If your actual model's signature differs (multiple inputs, different
   preprocessing, CTC-style sequence output instead of single-frame
   classification), adjust `ASLRecognizer._infer` / `_prep_frame` /
   `_extract_landmarks` accordingly — the rest of the pipeline only cares
   that `process_frame()` returns an `ASLPrediction(token, confidence, timestamp)`.

## Run

```bash
python main.py
```

Press `q` to quit. On-screen you'll see the current recognized sign and the
most recently spoken sentence.

## Tuning the adaptive queue

`AdaptiveQueueConfig` in `config.py`:
- `base_pause_seconds` / `min_adaptive_factor` / `max_adaptive_factor` control
  how long a pause in signing must be before a buffered gloss chunk is sent
  to translation — this threshold shrinks for fast signers and grows for slow
  ones, based on a rolling average of recent inter-sign gaps.
- `max_buffer_size` / `max_wait_seconds` are hard ceilings so a chunk is
  never held indefinitely even if no natural pause occurs.
- `dedup_repeats` collapses a held sign across consecutive frames into a
  single gloss token (otherwise you'd get `"HELLO HELLO HELLO"` instead of
  `"HELLO"`), matching the token granularity T5 was trained on.

## Notes / things you'll likely want to change

- `pyttsx3` is fully offline and near-zero latency, which is why it was
  chosen for "lightweight TTS" — but voice quality is robotic. If you later
  want more natural speech at the cost of some latency/footprint, swap
  `tts_engine.py` for something like Piper (still lightweight, ONNX-based,
  runs well on CPU) without touching any other file — it only needs to
  consume strings off `text_queue`.
- The confidence threshold (`ASLConfig.confidence_threshold`) is what
  prevents transition/blur frames between signs from being treated as real
  tokens. Tune it against your model's actual softmax behavior.
- `GlossTranslator.translate()` uses beam search (`num_beams=4`) to match
  higher-quality generation; drop to `num_beams=1` (greedy) if you need
  lower latency over quality on weaker hardware.
