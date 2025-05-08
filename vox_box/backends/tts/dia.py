from dataclasses import dataclass
import logging
import os
import time
import tempfile
import torch
from typing import Dict, Optional
import soundfile as sf
from vox_box.third_party.dia.dia.model import Dia as DiaModel

from vox_box.backends.tts.base import TTSBackend
from vox_box.utils.audio import convert
from vox_box.utils.log import log_method
from vox_box.config.config import BackendEnum, Config, TaskTypeEnum
from vox_box.utils.model import create_model_dict


logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    max_tokens: Optional[int] = (
        int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") is not None else None
    )
    cfg_scale: float = float(os.getenv("CFG_SCALE", 3.0))
    temperature: float = float(os.getenv("TEMPERATURE", 1.3))
    top_p: float = float(os.getenv("TOP_P", 0.95))
    use_torch_compile: bool = bool(os.getenv("USE_TORCH_COMPILE", True))
    cfg_filter_top_k: int = int(os.getenv("CFG_FILTER_TOP_K", 35))
    audio_prompt: Optional[str] = os.getenv("AUDIO_PROMPT", None)
    verbose: bool = bool(os.getenv("VERBOSE", True))


class Dia(TTSBackend):
    def __init__(
        self,
        cfg: Config,
    ):
        self.model_load = False
        self._cfg = cfg
        self._model = None
        self._model_dict = {}

    def load(self):
        if self.model_load:
            return self

        try:
            dtype = "float16"
            device = torch.device(self._cfg.device)
            if self._cfg.device == "cpu":
                dtype = "float32"  # for more compatibility
            self._model = DiaModel.from_pretrained(
                self._cfg.model, compute_dtype=dtype, device=device
            )
        except Exception as e:
            raise RuntimeError(f"Error loading Dia model: {e}")

        self._model_dict = create_model_dict(
            self._cfg.model,
            task_type=TaskTypeEnum.TTS,
            backend_framework=BackendEnum.DIA,
            voices=["English"],
        )

        self.model_load = True
        return self

    def is_load(self) -> bool:
        return self.model_load

    def model_info(self) -> Dict:
        return self._model_dict

    @log_method
    def speech(
        self,
        input: str,
        voice: Optional[str] = "English",
        speed: float = 1,
        reponse_format: str = "mp3",
        **kwargs,
    ) -> str:
        sample_rate = 44100

        start_time = time.time()

        generate_config = GenerateConfig()
        if self._cfg.device == "cpu":
            generate_config.use_torch_compile = False
        output_audio = self._model.generate(
            text=input,
            max_tokens=generate_config.max_tokens,
            cfg_scale=generate_config.cfg_scale,
            temperature=generate_config.temperature,
            top_p=generate_config.top_p,
            cfg_filter_top_k=generate_config.cfg_filter_top_k,
            use_torch_compile=generate_config.use_torch_compile,
            audio_prompt=generate_config.audio_prompt,
            verbose=generate_config.verbose,
        )

        end_time = time.time()
        logger.info(
            f"Audio generation completed in {end_time - start_time:.2f} seconds"
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            wav_file_path = temp_file.name
            sf.write(wav_file_path, output_audio, sample_rate)
            output_file_path = convert(wav_file_path, reponse_format, speed)
            return output_file_path
