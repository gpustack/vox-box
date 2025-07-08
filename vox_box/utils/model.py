import logging
import os
import time
from typing import Dict

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


def parse_and_set_cuda_visible_devices(cfg: Config):
    """
    Parse CUDA device in format cuda:1 and set CUDA_VISIBLE_DEVICES accordingly.
    """
    if cfg.device.startswith("cuda:"):
        device_index = cfg.device.split(":")[1]
        if device_index.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = device_index
            logger.info(f"Set CUDA_VISIBLE_DEVICES = {device_index}")
        else:
            raise ValueError(f"Invalid CUDA device index: {device_index}")
