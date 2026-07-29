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
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

        self.retranslate_slot = LatestSlot()
        self.sentence_state = SharedSentenceState()

        self.recognizer = ASLRecognizer(cfg.asl)
        self.gloss_buffer = StreamingGlossBuffer(cfg.queue, self.retranslate_slot, self.sentence_state)
        self.translator = GlossTranslator(cfg.translator, self.retranslate_slot, self.sentence_state)
        self.speaker = TinyTTSSpeaker(cfg.tts, self.sentence_state, self.gloss_buffer)

    def start(self):
        self.gloss_buffer.start()
        self.translator.start()
        self.speaker.start()

    def stop(self):
        self.gloss_buffer.stop()
        self.translator.stop()
        self.speaker.stop()

    @property
    def last_spoken_text(self) -> str:
        return self.speaker.last_spoken_text

    def on_frame(self, frame_bgr):
        """Call this once per captured video frame."""
        prediction = self.recognizer.process_frame(frame_bgr)
        if prediction is not None and prediction.token is not None:
            self.gloss_buffer.push(prediction.token)
            return prediction
        return None
