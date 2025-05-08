import logging
import os
import json
from typing import Dict
from vox_box.config.config import BackendEnum, Config, TaskTypeEnum
from vox_box.downloader.downloaders import download_file
from vox_box.estimator.base import Estimator
from vox_box.utils.model import create_model_dict

logger = logging.getLogger(__name__)


class Dia(Estimator):
    def __init__(
        self,
        cfg: Config,
    ):
        self._cfg = cfg
        self._config_file_name = "config.json"

    def model_info(self) -> Dict:
        model = (
            self._cfg.model
            or self._cfg.huggingface_repo_id
            or self._cfg.model_scope_model_id
        )
        supported = self._supported()
        return create_model_dict(
            model,
            supported=supported,
            task_type=TaskTypeEnum.TTS,
            backend_framework=BackendEnum.DIA,
        )

    def _supported(self) -> bool:
        if self._cfg.model is not None:
            return is_dia_config(os.path.join(self._cfg.model, self._config_file_name))
        elif (
            self._cfg.huggingface_repo_id is not None
            or self._cfg.model_scope_model_id is not None
        ):
            return self._check_remote_model()

    def _check_remote_model(self) -> bool:
        try:
            download_file_path = download_file(
                huggingface_repo_id=self._cfg.huggingface_repo_id,
                huggingface_filename=self._config_file_name,
                model_scope_model_id=self._cfg.model_scope_model_id,
                model_scope_file_path=self._config_file_name,
                cache_dir=self._cfg.cache_dir,
            )
        except Exception as e:
            logger.debug(f"Failed to download {self._config_file_name}: {e}")
            return False

        return is_dia_config(download_file_path)


def is_dia_config(filepath: str) -> bool:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)

        data = config.get("data", {})

        audio_keys = {
            "audio_bos_value",
            "audio_eos_value",
            "audio_pad_value",
            "audio_length",
            "delay_pattern",
        }
        if not audio_keys.issubset(data):
            return False

        return True
    except Exception as e:
        print(f"Error loading Dia config: {e}")
        return False
