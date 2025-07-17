import logging
import os
import re
import time
from typing import Dict, Optional

from vox_box.config import Config

logger = logging.getLogger(__name__)


def create_model_dict(id: str, **kwargs) -> Dict:
    d = {
        "id": id,
        "object": "model",
        "created": int(time.time()),
        "owner": "vox-box",
        "backend": "vox-box",
    }

    for k, v in kwargs.items():
        if v is not None:
            d[k] = v

    return d


def preconfigure_faster_whisper_env(cfg: Config):
    """
    Due to faster-whisper's problematic handling of device parameters, CUDA_VISIBLE_DEVICES requires special configuration.
    There are 3 methods to specify the GPU device for faster-whisper models:
    1. Manually set `CUDA_VISIBLE_DEVICES={gpu_index}` and use "--device cuda:0" in CLI parameters.
    2. If env `IS_FASTER_WHISPER` is unset AND the model path/name contains "faster-whisper", CUDA_VISIBLE_DEVICES will be set based on the GPU selector's device choice.
    3. When `IS_FASTER_WHISPER=True`, CUDA_VISIBLE_DEVICES will be set based on the GPU selector's device choice.
    """
    is_faster_whisper: Optional[bool] = None
    if os.getenv("IS_FASTER_WHISPER"):
        is_faster_whisper = os.getenv("IS_FASTER_WHISPER") in ("true", "1", "yes", "y")

    if is_faster_whisper is False:
        return

    # If unset is_faster_whisper, check if the model name contains "faster-whisper"
    if is_faster_whisper is None and re.search(
        r"faster.*whisper", cfg.model, re.IGNORECASE
    ):
        is_faster_whisper = True

    if is_faster_whisper is True and cfg.device.startswith("cuda:"):
        device_index = cfg.device.split(":")[1]
        if device_index.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = device_index
            logger.info(f"Set CUDA_VISIBLE_DEVICES = {device_index}")
        else:
            raise ValueError(f"Invalid CUDA device index: {device_index}")
