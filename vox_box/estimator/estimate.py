import logging
from typing import Dict, List
from vox_box.config.config import Config
from vox_box.estimator.bark import Bark
from vox_box.estimator.base import Estimator
from vox_box.estimator.cosyvoice import CosyVoice
from vox_box.estimator.dia import Dia
from vox_box.estimator.faster_whisper import FasterWhisper
from vox_box.estimator.funasr import FunASR
from vox_box.utils.model import create_model_dict

logger = logging.getLogger(__name__)


def estimate_model(cfg: Config) -> Dict:
    estimators: List[Estimator] = [
        CosyVoice(cfg),
        FasterWhisper(cfg),
        FunASR(cfg),
        Bark(cfg),
        Dia(cfg),
    ]

    model = cfg.model or cfg.huggingface_repo_id or cfg.model_scope_model_id
    model_info = create_model_dict(
        model,
        supported=False,
    )

    for estimator in estimators:
        model_info = estimator.model_info()
        if model_info["supported"]:
            return model_info

    return model_info
