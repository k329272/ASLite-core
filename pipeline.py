"""
Wires the stages together:

  camera frames
      -> ASLRecognizer (OpenVINO)              [per-frame gloss token / None]
      -> StreamingGlossBuffer                   [accumulates tokens, detects sentence pauses]
      -> GlossTranslator (your fine-tuned T5)   [continuously re-translates the whole buffer,
                                                  forced to keep already-spoken words fixed]
      -> TinyTTSSpeaker                         [speaks the next unspoken word, locking it]

The translator and speaker race against each other on purpose: T5 keeps
refining the *unspoken tail* of the sentence as new signs arrive, while TTS
keeps locking in whatever word is currently at the front of that tail. Once
a sentence's pause boundary hits and every word has been voiced, state is
cleared and the next sentence starts clean.
"""

from config import PipelineConfig
from asl_recognizer import ASLRecognizer
from streaming_state import LatestSlot, SharedSentenceState, StreamingGlossBuffer
from translator import GlossTranslator
from tts_engine import TinyTTSSpeaker


class ASLSpeechPipeline:
    """Wire the recognizer, queue, translator, and speaker together."""

    def __init__(self, cfg: PipelineConfig):
        """Initialize all pipeline stages with the provided configuration."""
        self.cfg = cfg

        self.retranslate_slot = LatestSlot()
        self.sentence_state = SharedSentenceState()

        self.recognizer = ASLRecognizer(cfg.asl)
        self.gloss_buffer = StreamingGlossBuffer(
            cfg.queue, self.retranslate_slot, self.sentence_state
        )
        self.translator = GlossTranslator(
            cfg.translator, self.retranslate_slot, self.sentence_state
        )
        self.speaker = TinyTTSSpeaker(cfg.tts, self.sentence_state, self.gloss_buffer)

    def start(self):
        """Start all background workers in the pipeline."""
        self.gloss_buffer.start()
        self.translator.start()
        self.speaker.start()

    def stop(self):
        """Stop all background workers in the pipeline."""
        self.gloss_buffer.stop()
        self.translator.stop()
        self.speaker.stop()

    @property
    def last_spoken_text(self) -> str:
        """Expose the most recently spoken sentence text for the UI."""
        return self.speaker.last_spoken_text

    @property
    def last_audio_payload(self) -> bytes:
        """Expose the most recently synthesized audio payload for streaming."""
        return self.speaker.last_audio_payload

    @property
    def last_audio_sample_rate(self) -> int:
        """Expose the sample rate for the most recently synthesized audio."""
        return self.speaker.last_audio_sample_rate

    def on_frame(self, frame_bgr):
        """Process a single camera frame and forward any recognized token."""
        prediction = self.recognizer.process_frame(frame_bgr)
        if prediction is not None and prediction.token is not None:
            self.gloss_buffer.push(prediction.token)
            return prediction
        return None
