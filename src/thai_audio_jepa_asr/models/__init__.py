"""Speech recognition and self-supervised models."""

from thai_audio_jepa_asr.models.audio_jepa import AudioJEPAEncoder
from thai_audio_jepa_asr.models.temporal_adapter import TemporalAdapter, validate_ctc_feasibility
from thai_audio_jepa_asr.models.ctc_model import AudioJEPACTCAudioModel
from thai_audio_jepa_asr.models.jepa_ssl import AudioJEPASSLModule, VisionTransformerPredictor, norm_mse_loss
from thai_audio_jepa_asr.models.baselines import BaseSpeechEncoder, AudioJEPAAdapter, GenericWav2Vec2Encoder

__all__ = [
    "AudioJEPAEncoder",
    "TemporalAdapter",
    "validate_ctc_feasibility",
    "AudioJEPACTCAudioModel",
    "AudioJEPASSLModule",
    "VisionTransformerPredictor",
    "norm_mse_loss",
    "BaseSpeechEncoder",
    "AudioJEPAAdapter",
    "GenericWav2Vec2Encoder",
]
