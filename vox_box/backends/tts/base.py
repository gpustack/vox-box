from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional
from vox_box.config.config import Config
from vox_box.utils.log import log_method

logger = logging.getLogger(__name__)


class TTSBackend(ABC):
    def __init__(
        self,
        cfg: Config,
    ):
        pass

    @abstractmethod
    def load() -> bool:
        pass

    @abstractmethod
    def is_load() -> bool:
        pass

    @abstractmethod
    def model_info() -> Dict:
        pass

    @log_method
    def speech(
        self,
        input: str,
        voice: Optional[str],
        speed: float = 1,
        reponse_format: str = "mp3",
        **kwargs
    ):
        pass

    def is_stream_supported(self) -> bool:
        return False

    def speech_stream(
        self, input: str, voice: Optional[str], speed: float = 1, **kwargs
    ):
        raise NotImplementedError("Streaming is not supported for this backend")
