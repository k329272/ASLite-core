import numpy as np
from types import SimpleNamespace

from tts_engine import KittenTTSSpeaker


class DummyEngine:
    def __init__(self, model_name=None, cache_dir=None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.calls = []

    def generate(self, text, voice, speed, clean_text):
        self.calls.append((text, voice, speed, clean_text))
        return np.asarray([0.1, -0.2, 0.3], dtype=np.float32)


def test_synthesize_uses_kitten_tts_generate():
    speaker = object.__new__(KittenTTSSpeaker)
    speaker.cfg = SimpleNamespace(
        kitttentts_model="KittenML/kitten-tts-nano-0.8",
        kitttentts_voice="Jasper",
        kitttentts_speed=1.2,
        min_word_stability=2,
    )
    speaker.engine = DummyEngine()

    audio, sample_rate = speaker._synthesize("hello")

    assert sample_rate == 24000
    assert np.array_equal(audio, np.asarray([0.1, -0.2, 0.3], dtype=np.float32))
    assert speaker.engine.calls == [("hello", "Jasper", 1.2, True)]


def test_synthesize_text_to_wav_uses_phrase_level_generation():
    speaker = object.__new__(KittenTTSSpeaker)
    speaker.cfg = SimpleNamespace(
        kitttentts_model="KittenML/kitten-tts-nano-0.8",
        kitttentts_voice="Jasper",
        kitttentts_speed=1.2,
        min_word_stability=2,
    )
    speaker.engine = DummyEngine()

    payload = speaker.synthesize_text_to_wav("hello world")

    assert payload.startswith(b"RIFF")
    assert speaker.engine.calls == [("hello world", "Jasper", 1.2, True)]
