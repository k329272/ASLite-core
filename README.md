# ASL → Speech

A real-time demo that turns sign language into spoken words. Point a webcam at a signer, and the app will recognize signs, build a sentence, and speak it out loud.

## What this app does

- Uses your webcam to watch signing in real time
- Recognizes individual signs from an ASL model
- Builds a sentence as signs are detected
- Speaks the translated words aloud as the sentence develops

This makes it easier to use sign language in a live setting without needing to type or manually translate each phrase.

## Before you start

You will need:

- Python 3.8 or newer
- A working webcam
- An ASL recognition model in OpenVINO format
- A translation model for turning glosses into natural language
- The TinyTTS assets needed for speech output

## Quick start

1. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Place your models in the expected locations.
   - The ASL model should be available as the OpenVINO `.xml` and `.bin` files configured in the settings.
   - The translation model should be placed in the folder referenced by the configuration.
   - The TinyTTS checkpoint should be available at the path configured for speech output.

3. Review the settings in `config.py` if you want to change:
   - the model paths
   - the confidence threshold for recognition
   - the speech and sentence timing behavior

4. Start the webcam demo:

   ```bash
   python main.py
   ```

5. Press `q` to exit the program.

6. Or start the live server that exposes the latest transcript and synthesized audio over HTTP:

   ```bash
   python server.py
   ```

   Open http://localhost:8000/ to receive JSON containing the latest `text`, `audio_base64`, and `audio_sample_rate` values.

## Using the app

When the app is running, you should see the current recognized sign and the most recent spoken sentence on screen. The app is designed to work best when:

- the camera has a clear view of the signer
- the signer pauses briefly between sentences
- the model has been trained on similar signs and lighting conditions

## Adjusting the experience

If the app is too eager to start speaking, too slow to respond, or misses signs, try adjusting the settings in `config.py`:

- Raise or lower the recognition confidence threshold
- Change the pause timing used to detect the end of a sentence
- Increase stability so brief flickers do not become false signs

These options are meant to make the app more reliable for your setup and speaking style.

## Troubleshooting

- If nothing is recognized, check that the camera is working and the model files are correctly configured.
- If speech does not play, confirm that the TinyTTS files are present and that your environment can play audio.
- If translations feel delayed, try lowering the amount of waiting before the app commits the next word.
- If the app repeats signs too often, increase the stability threshold so it waits for a more confident prediction.

## Main files

- `config.py` — main settings for models, thresholds, and timing
- `main.py` — launch point for the webcam demo
- `asl_recognizer.py` — handles sign recognition
- `translator.py` — turns recognized signs into spoken text
- `tts_engine.py` — produces speech output
