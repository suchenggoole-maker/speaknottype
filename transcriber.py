"""语音识别模块 - 支持 whisper.cpp / faster-whisper / Windows 原生语音识别"""

import os
import subprocess
import logging
import config

log = logging.getLogger("SpeakNotType")


class TranscriptionError(Exception):
    """语音识别异常"""
    pass


class WhisperCppEngine:
    """whisper.cpp 子进程调用引擎（推荐，质量最高）"""

    def __init__(self, model: str = "tiny", language: str = "auto"):
        self.model = model
        self.language = language
        self._cli_path = config.get_whisper_cli_path()
        self._model_path = os.path.join(
            config.get_models_dir(), f"ggml-{model}.bin"
        )
        self._validate()

    def _validate(self):
        if not os.path.exists(self._cli_path):
            raise TranscriptionError(
                f"whisper-cli.exe not found: {self._cli_path}\n"
                f"Run 'python installer.py' or download manually:\n"
                f"  https://github.com/ggml-org/whisper.cpp/releases"
            )
        if not os.path.exists(self._model_path):
            raise TranscriptionError(
                f"Model file not found: {self._model_path}\n"
                f"Run 'python installer.py' to download models"
            )

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file using whisper.cpp CLI"""
        import time as _time
        from perf_timer import PerfTimer
        _t = PerfTimer("WhisperCpp")
        log = logging.getLogger("SpeakNotType")

        audio_size_kb = os.path.getsize(audio_path) / 1024
        _t.mark("start")

        lang_map = {"auto": "auto", "zh": "zh", "en": "en"}
        lang = lang_map.get(self.language, "auto")

        cmd = [
            self._cli_path,
            "-m", self._model_path,
            "-f", audio_path,
            "-otxt",
            "-l", lang,
            "--no-timestamps",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )

        proc_ms = _t.mark("whisper_cli") * 1000

        # whisper.cpp writes result to .txt file alongside audio
        txt_path = audio_path + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            os.remove(txt_path)
        elif result.stdout and result.stdout.strip():
            text = result.stdout.strip()
        elif result.returncode != 0:
            raise TranscriptionError(
                f"whisper.cpp failed: {result.stderr[:200]}"
            )
        else:
            text = ""

        text_len = len(text)
        log.info(
            f"[PERF] whisper.cpp (large-v3-turbo): {proc_ms:.0f}ms | "
            f"音频: {audio_size_kb:.0f}KB | "
            f"输出: {text_len}字符 | "
            f"速度: {audio_size_kb/(proc_ms/1000):.0f}KB/s"
        )
        return text


class WhisperCppServerEngine:
    """whisper.cpp Server 引擎 — 模型常驻内存，HTTP 调用（推荐，速度+准确率最佳）"""

    def __init__(self, model: str = "large-v3-turbo", language: str = "auto",
                 server_manager=None):
        self.model = model
        self.language = language
        self._manager = server_manager

    def transcribe(self, audio_path: str) -> str:
        from perf_timer import PerfTimer
        _t = PerfTimer("WhisperServer")
        _t.mark("http_request")
        text = self._manager.transcribe(audio_path)
        elapsed = _t.mark("done") * 1000
        log.info(f"[PERF] whisper-server ({self.model}): {elapsed:.0f}ms | 输出: {len(text)}字符")
        return text


class FasterWhisperEngine:
    """faster-whisper 引擎（CTranslate2，支持 GPU int8/fp16 加速）"""

    def __init__(self, model: str = "medium", language: str = "auto",
                 device: str = "auto", compute_type: str = "auto"):
        self.model = model
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self._fw_model = None
        self._model_path = self._resolve_model_path()
        self._init_model()

    def _resolve_model_path(self) -> str:
        """Resolve model name to path"""
        import os
        # First check local ct2 models dir
        local_ct2 = os.path.join("whisper", "models-ct2", self.model)
        if os.path.isdir(local_ct2) and os.path.exists(os.path.join(local_ct2, "model.bin")):
            return local_ct2
        # Fallback: HuggingFace repo IDs
        hf_map = {
            "small": "Systran/faster-whisper-small",
            "medium": "Systran/faster-whisper-medium",
            "large-v3": "Systran/faster-whisper-large-v3",
            "large-v3-turbo": "Systran/faster-whisper-large-v3-turbo",
        }
        return hf_map.get(self.model, self.model)

    def _init_model(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise TranscriptionError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )
        # Determine device
        device = self.device
        compute_type = self.compute_type
        if device == "auto":
            try:
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
                    compute_type = "float16"
                else:
                    device = "cpu"
                    compute_type = "int8"
            except Exception:
                device = "cpu"
                compute_type = "int8"
        log.info(f"[FasterWhisper] Loading {self.model} on {device}/{compute_type}...")
        self._fw_model = WhisperModel(
            self._model_path,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio_path: str) -> str:
        from perf_timer import PerfTimer
        _t = PerfTimer("FasterWhisper")
        _t.mark("start")

        segments, info = self._fw_model.transcribe(
            audio_path,
            language=self.language if self.language != "auto" else None,
            beam_size=5,
        )
        text = " ".join(seg.text for seg in segments).strip()

        elapsed = _t.mark("done") * 1000
        log.info(f"[PERF] faster-whisper ({self.model}): {elapsed:.0f}ms | 输出: {len(text)}字符")
        return text


class WindowsSpeechEngine:
    """Windows 原生语音识别引擎（零配置）"""

    def __init__(self, model: str = "", language: str = "auto"):
        self.language = language

    def transcribe(self, audio_path: str) -> str:
        try:
            import speech_recognition as sr
        except ImportError:
            raise TranscriptionError(
                "speech_recognition not installed. Run: pip install speechrecognition"
            )

        recognizer = sr.Recognizer()

        # Map language for Google API
        lang_map = {
            "zh": "zh-CN",
            "en": "en-US",
        }
        lang = lang_map.get(self.language, "en-US")

        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)

        # Google's free speech API (requires internet, works for many languages)
        errors = []
        try:
            text = recognizer.recognize_google(audio, language=lang)
            if text:
                return text
        except Exception as e:
            errors.append(f"Google API: {e}")

        # Fallback: recognize_sphinx requires pocketsphinx install
        try:
            text = recognizer.recognize_sphinx(audio, language=lang.split("-")[0])
            if text:
                return text
        except Exception as e:
            errors.append(f"Sphinx: {e}")

        raise TranscriptionError(
            f"Speech recognition failed: {'; '.join(errors)}\n"
            f"Try running with whisper-cpp for better quality:\n"
            f"1. Download ggml-tiny.bin from:\n"
            f"   https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin\n"
            f"2. Place it in: speaknottype/whisper/models/\n"
            f"3. Or set engine.backend = 'whisper-cpp' in config.toml"
        )


def create_engine(backend: str = None, model: str = "tiny", language: str = "auto",
                  server_manager=None, device: str = "auto"):
    """Factory: create recognition engine with auto-fallback"""

    engines = {
        "whisper-cpp": WhisperCppEngine,
        "whisper-server": WhisperCppServerEngine,
        "faster-whisper": FasterWhisperEngine,
        "windows-speech": WindowsSpeechEngine,
    }

    # Auto-detect best available engine if not specified
    if backend is None:
        backend = _detect_best_engine(server_manager)

    if backend not in engines:
        raise ValueError(f"Unknown engine: {backend}. Available: {list(engines.keys())}")

    # Pass extra args based on engine type
    if backend == "whisper-server":
        return engines[backend](model=model, language=language, server_manager=server_manager)
    if backend == "faster-whisper":
        return engines[backend](model=model, language=language, device=device)

    return engines[backend](model=model, language=language)


def _detect_best_engine(server_manager=None) -> str:
    """Auto-detect the best available speech recognition engine"""
    # 0. Check for CUDA GPU → faster-whisper is fastest
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            # Check if we have a local model
            import os
            local_model = os.path.join(
                config.get_project_root(), "whisper", "models-ct2", "medium"
            )
            if os.path.exists(os.path.join(local_model, "model.bin")):
                return "faster-whisper"
    except Exception:
        pass

    # 1. Whisper server is the best if available
    if server_manager is not None:
        return "whisper-server"

    # 2. Check for whisper.cpp (good quality, but slow per-invocation)
    cli_path = config.get_whisper_cli_path()
    if os.path.exists(cli_path):
        return "whisper-cpp"

    # 3. Default to windows-speech (zero config)
    return "windows-speech"