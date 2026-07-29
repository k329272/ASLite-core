"""
Central configuration for the ASL -> T5 -> TTS pipeline.
Edit these values to point at your actual model files.
"""

from dataclasses import dataclass, field


@dataclass
class ASLConfig:
    """Configuration for the ASL recognition stage."""

    # Path to your OpenVINO IR ASL sign-recognition model
    model_xml: str = "models/asl_recognizer.xml"
    device: str = "CPU"  # "CPU", "GPU", "AUTO", etc.

    # Path to a JSON list mapping class index -> gloss token, e.g. ["HELLO", "THANK-YOU", ...]
    labels_path: str = "models/gloss_labels.json"

    # "landmarks" = model expects a flattened MediaPipe hand-landmark vector
    # "frame"     = model expects a raw resized/normalized image frame
    input_mode: str = "landmarks"
    frame_size: tuple = (224, 224)

    # Minimum softmax confidence to accept a prediction as a real sign
    # (rejects the implicit "no sign" / transition frames)
    confidence_threshold: float = 0.65

    # A predicted sign must be the top prediction for this many consecutive
    # inference passes before it's accepted as a new gloss token. Since
    # locked words can never be revised once spoken, a single flickering
    # frame turning into a wrong gloss token is the most expensive kind of
    # error in this pipeline -- this trades a little latency for a big
    # reduction in false positives.
    min_stable_frames: int = 3

    # Run inference at most this many times per second (throttles CPU use)
    inference_fps: int = 15


@dataclass
class AdaptiveQueueConfig:
    """Configuration for the adaptive gloss buffering queue."""

    # A run of identical consecutive tokens collapses to one (dedup)
    dedup_repeats: bool = True

    # Hard ceiling on how many gloss tokens accumulate in one sentence before
    # it's forced to end, even if the signer never pauses
    max_buffer_size: int = 12

    # Absolute max time (seconds) one sentence can run before being forced to
    # end, even if the signer hasn't paused
    max_wait_seconds: float = 4.0

    # Minimum pause (seconds) with no new token that counts as the end of a sentence
    base_pause_seconds: float = 0.9

    # Adaptive multiplier: the pause threshold used to end a sentence is
    # base_pause_seconds * adaptive_factor, where adaptive_factor grows/shrinks
    # based on the signer's recent average inter-token gap (faster signer -> shorter
    # pause needed to count as a break; slower signer -> more patience)
    min_adaptive_factor: float = 0.6
    max_adaptive_factor: float = 1.8

    # How many recent inter-token gaps to average when adapting
    gap_history_len: int = 8


@dataclass
class TranslatorConfig:
    """Configuration for the translator model."""

    model_path: str = (
        "optimized_t5_model"  # the folder saved in your notebook's last cell
    )
    device: str = "cpu"  # quantized model was exported for CPU
    max_input_length: int = 64
    max_new_tokens: int = 64
    num_beams: int = 4


@dataclass
class TTSConfig:
    """Configuration for the text-to-speech synthesis stage."""

    tinytts_checkpoint: str = "models/G.pth"  # TinyTTS pretrained checkpoint
    tinytts_device: str = "cpu"
    tinytts_speaker: str = "MALE"
    tinytts_speed: float = 1.0

    # A candidate tail word must come back unchanged across this many
    # consecutive re-translations before it's locked & spoken. Guards
    # against committing to a word T5 would have phrased differently with
    # a bit more context (classic greedy-commit risk in any simultaneous
    # translation system). Ignored during final sentence drain -- once the
    # signer has paused and translation has caught up, remaining words are
    # spoken immediately since nothing more is coming to change them.
    min_word_stability: int = 2


@dataclass
class PipelineConfig:
    """Top-level configuration for the full sign-to-speech pipeline."""

    asl: ASLConfig = field(default_factory=ASLConfig)
    queue: AdaptiveQueueConfig = field(default_factory=AdaptiveQueueConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
