import dataclasses
import json
import logging
import os
import platform
from typing import Dict, List, Optional
import tempfile
import io
from vox_box.backends.stt.base import STTBackend
from vox_box.config.config import BackendEnum, Config, TaskTypeEnum
from vox_box.utils.log import log_method
from vox_box.utils.model import create_model_dict
from faster_whisper.transcribe import WhisperModel

logger = logging.getLogger(__name__)


class FasterWhisper(STTBackend):
    def __init__(
        self,
        cfg: Config,
    ):
        self._cfg = cfg
        self.model_load = False
        self._cfg = cfg
        self._model = None
        self._model_dict = {}

        self._preprocessor_config_json = None
        preprocessor_config_path = os.path.join(
            self._cfg.model, "preprocessor_config.json"
        )
        if os.path.exists(preprocessor_config_path):
            with open(preprocessor_config_path, "r", encoding="utf-8") as f:
                self._preprocessor_config_json = json.load(f)

    def load(self):
        if self.model_load:
            return self

        cpu_threads = 0
        if self._cfg.device == "cpu":
            cpu_threads = 8

        compute_type = "default"
        if platform.system() == "Darwin":
            compute_type = "int8"

        device = self._cfg.device
        device_index = 0
        if self._cfg.device != "cpu":
            arr = device.split(":")
            device = arr[0]
            if len(arr) > 1:
                device_index = int(arr[1])

        self._model = WhisperModel(
            self._cfg.model,
            device=device,
            device_index=device_index,
            cpu_threads=cpu_threads,
            compute_type=compute_type,
        )

        self._model_dict = create_model_dict(
            self._cfg.model,
            task_type=TaskTypeEnum.STT,
            backend_framework=BackendEnum.FASTER_WHISPER,
        )
        self.model_load = True
        return self

    def is_load(self) -> bool:
        return self.model_load

    def model_info(self) -> Dict:
        return self._model_dict

    @log_method
    def transcribe(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = 0.2,
        timestamp_granularities: Optional[List[str]] = ["segment"],
        response_format: str = "json",
        **kwargs,
    ):

        if language == "auto":
            language = None
            # Accept:
            # af, am, ar, as, az, ba, be, bg, bn, bo, br, bs, ca, cs, cy, da, de,
            # el, en, es, et, eu, fa, fi, fo, fr, gl, gu, ha, haw, he, hi, hr, ht,
            # hu, hy, id, is, it, ja, jw, ka, kk, km, kn, ko, la, lb, ln, lo, lt,
            # lv, mg, mi, mk, ml, mn, mr, ms, mt, my, ne, nl, nn, no, oc, pa, pl,
            # ps, pt, ro, ru, sa, sd, si, sk, sl, sn, so, sq, sr, su, sv, sw, ta,
            # te, tg, th, tk, tl, tr, tt, uk, ur, uz, vi, yi, yo, zh, yue

        without_timestamps = True
        word_timestamps = False
        if response_format == "verbose_json" and timestamp_granularities is not None:
            without_timestamps = False
            if "word" in timestamp_granularities:
                word_timestamps = True

        with tempfile.NamedTemporaryFile(buffering=0) as f:
            f.write(audio)

            segs, info = self._model.transcribe(
                f.name,
                language=language,
                initial_prompt=prompt,
                temperature=temperature,
                without_timestamps=without_timestamps,
                word_timestamps=word_timestamps,
                **kwargs,
            )

            # The transcription will actually run here.
            timestamps = []
            text_buffer = io.StringIO()
            for seg in segs:
                text_buffer.write(seg.text)

                if not without_timestamps:

                    if word_timestamps:
                        for wd in seg.words:
                            timestamps.append(dataclasses.asdict(wd))
                    else:
                        timestamps.append(dataclasses.asdict(seg))

            text = text_buffer.getvalue()
            if without_timestamps:
                return text

            response = {
                "task": "transcribe",
                "language": info.language,
                "duration": info.duration,
                "text": text,
            }
            if word_timestamps:
                response["words"] = timestamps
            else:
                response["segments"] = timestamps

            return response
