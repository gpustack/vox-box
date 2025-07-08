import json
import logging
import os
import platform
from typing import Dict, List, Optional
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

        device = self._cfg.device
        if device.startswith("cuda:"):
            device = device.split(":")[0]

        compute_type = "default"
        if platform.system() == "Darwin":
            compute_type = "int8"

        self._model = WhisperModel(
            self._cfg.model,
            device=device,
            cpu_threads=cpu_threads,
            compute_type=compute_type,
        )

        self._languages = self._get_languages()

        self._model_dict = create_model_dict(
            self._cfg.model,
            task_type=TaskTypeEnum.STT,
            backend_framework=BackendEnum.FASTER_WHISPER,
            languages=self._languages,
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

        audio_data = io.BytesIO(audio)
        segs, info = self._model.transcribe(
            audio_data,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            without_timestamps=without_timestamps,
            word_timestamps=word_timestamps,
        )

        # The transcription will actually run here.
        timestamps = []
        text_buffer = io.StringIO()
        for seg in segs:
            text_buffer.write(seg.text)

            if not without_timestamps:
                if word_timestamps:
                    for wd in seg.words:
                        timestamps.append(wd._asdict())
                else:
                    timestamps.append(seg._asdict())

        text = text_buffer.getvalue().strip()
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

    def _get_languages(self) -> List[Dict]:
        return [
            {"auto": "auto"},
            {"en": "english"},
            {"zh": "chinese"},
            {"de": "german"},
            {"es": "spanish"},
            {"ru": "russian"},
            {"ko": "korean"},
            {"fr": "french"},
            {"ja": "japanese"},
            {"pt": "portuguese"},
            {"pl": "polish"},
            {"ca": "catalan"},
            {"nl": "dutch"},
            {"it": "italian"},
            {"th": "thai"},
            {"tr": "turkish"},
            {"ar": "arabic"},
            {"sv": "swedish"},
            {"id": "indonesian"},
            {"hi": "hindi"},
            {"fi": "finnish"},
            {"vi": "vietnamese"},
            {"he": "hebrew"},
            {"uk": "ukrainian"},
            {"el": "greek"},
            {"ms": "malay"},
            {"cs": "czech"},
            {"ro": "romanian"},
            {"da": "danish"},
            {"hu": "hungarian"},
            {"ta": "tamil"},
            {"no": "norwegian"},
            {"ur": "urdu"},
            {"hr": "croatian"},
            {"bg": "bulgarian"},
            {"lt": "lithuanian"},
            {"la": "latin"},
            {"mi": "maori"},
            {"ml": "malayalam"},
            {"cy": "welsh"},
            {"sk": "slovak"},
            {"te": "telugu"},
            {"fa": "persian"},
            {"lv": "latvian"},
            {"bn": "bengali"},
            {"sr": "serbian"},
            {"az": "azerbaijani"},
            {"sl": "slovenian"},
            {"kn": "kannada"},
            {"et": "estonian"},
            {"mk": "macedonian"},
            {"br": "breton"},
            {"eu": "basque"},
            {"is": "icelandic"},
            {"hy": "armenian"},
            {"ne": "nepali"},
            {"mn": "mongolian"},
            {"bs": "bosnian"},
            {"kk": "kazakh"},
            {"sq": "albanian"},
            {"sw": "swahili"},
            {"gl": "galician"},
            {"mr": "marathi"},
            {"pa": "punjabi"},
            {"si": "sinhala"},
            {"km": "khmer"},
            {"sn": "shona"},
            {"yo": "yoruba"},
            {"so": "somali"},
            {"af": "afrikaans"},
            {"oc": "occitan"},
            {"ka": "georgian"},
            {"be": "belarusian"},
            {"tg": "tajik"},
            {"sd": "sindhi"},
            {"gu": "gujarati"},
            {"am": "amharic"},
            {"yi": "yiddish"},
            {"lo": "lao"},
            {"uz": "uzbek"},
            {"fo": "faroese"},
            {"ht": "haitian creole"},
            {"ps": "pashto"},
            {"tk": "turkmen"},
            {"nn": "nynorsk"},
            {"mt": "maltese"},
            {"sa": "sanskrit"},
            {"lb": "luxembourgish"},
            {"my": "myanmar"},
            {"bo": "tibetan"},
            {"tl": "tagalog"},
            {"mg": "malagasy"},
            {"as": "assamese"},
            {"tt": "tatar"},
            {"haw": "hawaiian"},
            {"ln": "lingala"},
            {"ha": "hausa"},
            {"ba": "bashkir"},
            {"jw": "javanese"},
            {"su": "sundanese"},
            {"yue": "cantonese"},
        ]
