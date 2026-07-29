"""
Central configuration for the ASL -> T5 -> TTS pipeline.
Edit these values to point at your actual model files.
"""

from dataclasses import dataclass, field


@dataclass
class ASLConfig:
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

    # Run inference at most this many times per second (throttles CPU use)
    inference_fps: int = 15


@dataclass
class AdaptiveQueueConfig:
    # A run of identical consecutive tokens collapses to one (dedup)
    dedup_repeats: bool = True

    # Hard ceiling on how many gloss tokens accumulate before a forced flush
    max_buffer_size: int = 12

    # Absolute max time (seconds) a buffer can sit before a forced flush,
    # even if the signer hasn't paused
    max_wait_seconds: float = 4.0

    # Minimum pause (seconds) with no new token that counts as a "sentence boundary"
    base_pause_seconds: float = 0.9

    # Adaptive multiplier: the pause threshold used for flushing is
    # base_pause_seconds * adaptive_factor, where adaptive_factor grows/shrinks
    # based on the signer's recent average inter-token gap (faster signer -> shorter
    # pause needed to count as a break; slower signer -> more patience)
    min_adaptive_factor: float = 0.6
    max_adaptive_factor: float = 1.8

    # How many recent inter-token gaps to average when adapting
    gap_history_len: int = 8


@dataclass
class TranslatorConfig:
    model_path: str = "optimized_t5_model"  # the folder saved in your notebook's last cell
    device: str = "cpu"  # quantized model was exported for CPU
    max_input_length: int = 64
    max_new_tokens: int = 64
    num_beams: int = 4


@dataclass
class TTSConfig:
    rate: int = 175          # words per minute
    volume: float = 1.0
    voice_id: str = ""       # leave blank for system default; see tts_engine.list_voices()


@dataclass
class PipelineConfig:
    asl: ASLConfig = field(default_factory=ASLConfig)
    queue: AdaptiveQueueConfig = field(default_factory=AdaptiveQueueConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
