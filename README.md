# ASL → Speech Pipeline

Real-time pipeline: webcam → OpenVINO ASL sign recognition → continuous
streaming translation with word-locking → TinyTTS (lightweight neural TTS).

```
camera frame
   │
   ▼
ASLRecognizer (OpenVINO)         one gloss token per confident sign
   │
   ▼
StreamingGlossBuffer              accumulates tokens for the current sentence,
   │                              detects a pause (adaptive to signing speed) to
   │                              mark the sentence as ended
   ▼
GlossTranslator (T5, from your notebook)
   │        re-translates the ENTIRE buffer on every new token, but is forced
   │        (via decoder_input_ids) to keep already-spoken words fixed and can
   │        only choose new wording for the still-unspoken tail
   ▼
TinyTTSSpeaker (word-by-word)
   │        speaks the next unspoken word, then locks it -- T5 can never
   │        revise it in a later pass
   ▼
once the sentence pause fires AND every candidate word has been spoken:
   state resets (buffer + locked words cleared) -> next sentence starts clean
```

### The locking mechanism

This isn't "translate once, then speak the sentence." T5 keeps re-guessing
the whole sentence as new signs come in — the wording after the current
point can still change — but anything TTS has already said out loud is
permanently fixed. Concretely:

- `SharedSentenceState` holds `locked_words` (already spoken) and
  `candidate_words` (T5's latest full guess, always starting with
  `locked_words` unchanged).
- Every re-translation forces T5's decoder to start from `locked_words`
  (via `decoder_input_ids`), so it can only generate new tokens for the
  tail — it's structurally incapable of rewriting what's locked.
- `TinyTTSSpeaker` picks off `candidate_words[len(locked_words)]`, speaks
  it, and only then calls `lock_next_word()` to commit it.
- Once the signer pauses (sentence boundary) and every candidate word has
  been voiced, `reset()` clears `locked_words`/`candidate_words` and
  `StreamingGlossBuffer.clear()` empties the gloss buffer — a fresh start
  for the next sentence.

## Files

| File | Purpose |
|---|---|
| `config.py` | All tunables — model paths, thresholds, timing |
| `asl_recognizer.py` | Loads your OpenVINO IR model, runs inference per frame |
| `streaming_state.py` | `StreamingGlossBuffer` (adaptive pause/sentence-boundary detection) + `SharedSentenceState` (locked/candidate words) + `LatestSlot` (single-slot mailbox for the translator) |
| `translator.py` | Loads `optimized_t5_model` (from your training notebook) and continuously re-translates with a locked decoder prefix |
| `tts_engine.py` | `TinyTTSSpeaker` — word-by-word playback via TinyTTS, locking each word as it's spoken |
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

3. **TinyTTS**: `pip install tiny-tts` downloads the tokenizer/G2P assets;
   the ~17 MB checkpoint (`G.pth`) needs to be placed at the path in
   `TTSConfig.tinytts_checkpoint`. TinyTTS is a small, fast-moving project —
   check the installed version's actual Python API (`pip show tiny-tts`)
   against `tts_engine.py`'s `_TinyTTSEngine(...)` / `.speak(...)` calls,
   which are inferred from its published CLI flags
   (`--checkpoint --speaker --speed --device`), and adjust kwargs if they've
   changed.

4. **Input mode**: `ASLConfig.input_mode` controls what the recognizer feeds
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

## Tuning the sentence boundary

`AdaptiveQueueConfig` in `config.py` (used by `StreamingGlossBuffer`):
- `base_pause_seconds` / `min_adaptive_factor` / `max_adaptive_factor` control
  how long a pause in signing must be before the sentence is marked ended —
  this threshold shrinks for fast signers and grows for slow ones, based on
  a rolling average of recent inter-sign gaps.
- `max_buffer_size` / `max_wait_seconds` are hard ceilings so a sentence is
  never left open indefinitely even if no natural pause occurs.
- `dedup_repeats` collapses a held sign across consecutive frames into a
  single gloss token (otherwise you'd get `"HELLO HELLO HELLO"` instead of
  `"HELLO"`), matching the token granularity T5 was trained on.

## Notes / things you'll likely want to change

- **Locked-word risk**: because TTS commits to a word the moment it's
  spoken, an early greedy commitment can occasionally lock in a word T5
  would have phrased differently with more context (a well-known tradeoff
  in simultaneous/streaming translation). If this happens too often for
  your gloss vocabulary, consider adding a short "look-ahead" delay before
  locking (e.g. only lock a word once it has survived N consecutive
  re-translations unchanged) — that logic would live in
  `TinyTTSSpeaker._run()`.
- **TinyTTS latency**: the current implementation synthesizes each word to
  a temp WAV file and plays it back via `sounddevice`/`soundfile`. If
  TinyTTS's installed API exposes a raw-array method (returning audio
  directly instead of writing a file), swap that in for lower per-word
  latency.
- The confidence threshold (`ASLConfig.confidence_threshold`) is what
  prevents transition/blur frames between signs from being treated as real
  tokens. Tune it against your model's actual softmax behavior.
- `GlossTranslator._retranslate()` uses beam search (`num_beams=4`); drop to
  `num_beams=1` (greedy) for lower per-token latency if re-translation is
  falling behind the signer's pace (watch for the translator's queue never
  catching up — `LatestSlot` will silently drop stale updates in that case).
