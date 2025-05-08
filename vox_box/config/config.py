from enum import Enum
from typing import Optional


class Config:
    """A class used to define vox-box configuration.

    Attributes:
        debug: Enable debug mode.
        host: Host to bind the server to.
        port: Port to bind the server to.
        model: Model path.
    """

    # Common options
    debug: bool = False

    # Server options
    host: Optional[str] = "0.0.0.0"
    port: Optional[int] = None
    data_dir: Optional[str] = None
    cache_dir: Optional[str] = None

    # Model options
    model: Optional[str] = None
    device: Optional[str] = "cpu"
    huggingface_repo_id: Optional[str] = None
    model_scope_model_id: Optional[str] = None


class BackendEnum(str, Enum):
    BARK = "Bark"
    COSY_VOICE = "CosyVoice"
    DIA = "Dia"
    FASTER_WHISPER = "FasterWhisper"
    FUN_ASR = "FunASR"


class TaskTypeEnum(str, Enum):
    TTS = "tts"
    STT = "stt"
