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
StreamingGlossBuffer             accumulates tokens for the current sentence,
   │                             detects a pause (adaptive to signing speed) to
   │                             mark the sentence as ended
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

### Live server

```bash
python server.py
```

The server listens on port `8000` and serves a JSON payload at `/` with:
- `text`: the latest recognized or spoken text
- `audio_base64`: the latest synthesized audio as base64-encoded WAV data
- `audio_sample_rate`: the sample rate for that audio chunk

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

## Recent improvements

- **Sign-stability smoothing** (`ASLConfig.min_stable_frames`): a predicted
  sign must be the top prediction for several consecutive inference passes
  before it's accepted as a new gloss token. Since locked words can never
  be revised once spoken, a single flickering frame turning into a wrong
  token was the most expensive kind of error in this pipeline — this cuts
  false positives at the source, at the cost of a little acceptance latency.
- **Word-locking stability** (`TTSConfig.min_word_stability`): a candidate
  tail word must survive this many consecutive re-translations unchanged
  before it's locked and spoken, rather than committing to whatever T5
  outputs first. During the final sentence drain (signer has paused and
  translation has caught up), remaining words are spoken immediately since
  nothing more is coming to change them — see `SharedSentenceState.
  get_stable_next_word`.
- **Translator/speaker catch-up race fixed**: `SharedSentenceState` now
  tracks `expected_token_count` (set on every accepted gloss token) vs.
  `translated_token_count` (set after each successful re-translation).
  `fully_spoken()` requires these to match before finalizing a sentence, so
  the speaker can no longer declare a sentence done based on a candidate
  that hasn't yet incorporated the very last sign.
- **Pipelined TTS**: `TinyTTSSpeaker` now runs a decision/synthesis thread
  and a separate playback thread. A word is locked and its audio
  synthesized as soon as it's stable, without waiting for the previous
  word to finish playing — so synthesis of word N+1 overlaps with playback
  of word N instead of leaving dead air between every word. Locking
  happens the moment a word is committed to the playback queue (guaranteed
  to be spoken next in order), not the exact instant its audio starts.
- **Resilience**: every background thread (gloss-buffer pause watcher,
  translator, TTS decision loop, TTS playback loop) now catches and logs
  exceptions per-iteration instead of dying silently on the first bad
  inference call or synthesis failure. `main.py` configures basic logging
  and guards each frame's processing the same way.

## Notes / things you might still want to change

- **TinyTTS latency**: each word is still synthesized via a temp WAV
  file + `soundfile`/`sounddevice` round trip. If TinyTTS's installed API
  exposes a method that returns a raw array directly (skipping the file),
  swap that into `TinyTTSSpeaker._synthesize` for lower per-word latency.
- The confidence threshold (`ASLConfig.confidence_threshold`) is what
  prevents transition/blur frames between signs from being treated as real
  tokens in the first place — tune it against your model's actual softmax
  behavior, in tandem with `min_stable_frames`.
- `GlossTranslator._retranslate()` uses beam search (`num_beams=4`); drop to
  `num_beams=1` (greedy) for lower per-token latency if re-translation is
  falling behind the signer's pace (watch for the translator's queue never
  catching up — `LatestSlot` will silently drop stale updates in that case,
  and `expected_token_count` vs. `translated_token_count` growing apart is
  a good signal this is happening).
