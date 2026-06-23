#!/usr/bin/env python3
"""SpeakNotType - Global voice input tool: press hotkey, speak, text appears."""

import sys
import os
import threading
import logging

# 获取项目根目录（兼容 PyInstaller 打包模式）
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = os.path.dirname(sys.executable)
else:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Add CUDA DLL path for faster-whisper GPU mode
_cuda_dll_dir = os.path.join(PROJECT_ROOT, "whisper", "Release")
if os.path.isdir(_cuda_dll_dir):
    os.environ["PATH"] = _cuda_dll_dir + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_cuda_dll_dir)
    except (AttributeError, OSError):
        pass

import config
import recorder
import transcriber
import output as output_module

# Logging to file to avoid terminal encoding issues
log_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[SpeakNotType] %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "speaknottype.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("SpeakNotType")

# Globals
_recorder: recorder.Recorder | None = None
_engine = None
_cfg: dict = {}
_recording_in_progress = False
_tray_icon = None
_server_manager = None
_stream_pipeline = None
_use_streaming = True  # 启用流水线模式


def _shutdown_server():
    """安全关闭 whisper-server"""
    global _server_manager
    if _server_manager:
        try:
            _server_manager.stop()
        except Exception:
            pass
        _server_manager = None


def _get_audio_duration(path: str) -> float | None:
    """获取音频文件时长（秒）"""
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.duration
    except Exception:
        return None


def on_hotkey_press():
    """Hotkey pressed - start recording"""
    global _recording_in_progress, _recorder
    if _recording_in_progress:
        return
    _recording_in_progress = True
    log.info("[REC] Recording started...")
    update_tray_icon("recording")

    # 使用流式录音器（server 模式下）
    if _use_streaming and _server_manager and _stream_pipeline:
        from streaming_recorder import StreamingRecorder
        _recorder = StreamingRecorder(
            sample_rate=_cfg["recording"]["sample_rate"],
            temp_dir=_cfg["recording"]["temp_dir"],
            chunk_handler=_stream_pipeline.submit_chunk,
        )
    _recorder.start_recording()


def on_hotkey_release():
    """Hotkey released - stop recording + transcribe + paste"""
    global _recording_in_progress
    if not _recording_in_progress:
        return

    log.info("[TRANS] Recording stopped, transcribing...")
    update_tray_icon("transcribing")

    _perf_t = None
    _audio_path = None
    try:
        from perf_timer import PerfTimer
        _perf_t = PerfTimer("FullPipeline")
        _perf_t.mark("start")

        from streaming_recorder import StreamingRecorder
        if isinstance(_recorder, StreamingRecorder):
            audio_path = _on_release_streaming(_perf_t)
        else:
            audio_path = _on_release_classic(_perf_t)

    except Exception as e:
        log.error(f"[ERR] Transcription failed: {e}", exc_info=True)
    finally:
        _recording_in_progress = False
        update_tray_icon("idle")
        # NOTE: streaming mode handles its own cleanup via temp files in stop_recording
        # classic mode also handles it via _on_release_classic

    if audio_path:
        total = _perf_t.log_summary(threshold_ms=5.0, label="全链路耗时")
        audio_duration = _get_audio_duration(audio_path)
        if audio_duration:
            realtime_ratio = total / audio_duration if audio_duration > 0 else 0
            log.info(
                f"[PERF] 实时率: {realtime_ratio:.2f}x | "
                f"语音{audio_duration:.1f}s → 处理{total:.1f}s"
            )
        log.info("[OK] Done")
        try:
            os.remove(audio_path)
        except Exception:
            pass


def _on_release_streaming(_perf_t) -> str | None:
    """流式模式：等所有段落的 Future 完成，拼接结果"""
    chunk_futures, audio_path = _recorder.stop_recording()
    if not audio_path:
        return None

    _perf_t.mark("wait_chunks")
    log.info(f"[STREAM] 等待 {len(chunk_futures)} 段识别完成...")

    texts = []
    for i, fut in enumerate(chunk_futures):
        try:
            text = fut.result(timeout=120)
            if text:
                texts.append(text)
        except Exception as e:
            log.error(f"[STREAM] 段 #{i+1} 失败: {e}")

    _perf_t.mark("merge")
    full_text = " ".join(texts).strip()
    log.info(f"[TEXT] Result: {full_text[:100]}{'...' if len(full_text) > 100 else ''}")

    if full_text:
        output_module.output_text(full_text, auto_paste=_cfg["output"]["auto_paste"])

    _perf_t.mark("output")
    return audio_path


def _on_release_classic(_perf_t) -> str | None:
    """非流式模式：原始逻辑"""
    audio_path = _recorder.stop_recording()
    if not audio_path:
        return None

    _perf_t.mark("transcribe")
    text = _engine.transcribe(audio_path)
    _perf_t.mark("output")

    log.info(f"[TEXT] Result: {text[:100]}{'...' if len(text) > 100 else ''}")
    if text:
        output_module.output_text(text, auto_paste=_cfg["output"]["auto_paste"])

    return audio_path


def setup_hotkey():
    """Register global hotkey"""
    hotkey = _cfg["general"]["hotkey"]
    mode = _cfg["general"]["mode"]

    import keyboard as kb

    # Parse hotkey: "ctrl+grave" -> key="grave", modifiers=["ctrl"]
    parts = hotkey.lower().split("+")
    main_key = parts[-1]
    modifiers = parts[:-1]

    def all_modifiers_pressed() -> bool:
        return all(kb.is_pressed(mod) for mod in modifiers) if modifiers else True

    if mode == "push-to-talk":
        kb.on_press_key(main_key, lambda e: on_hotkey_press() if all_modifiers_pressed() else None, suppress=True)
        kb.on_release_key(main_key, lambda e: on_hotkey_release(), suppress=True)
        log.info(f"[HOTKEY] Registered: {hotkey} (push-to-talk)")
    elif mode == "toggle":
        _toggle_state = [False]

        def on_toggle(e):
            if not all_modifiers_pressed():
                return
            if _toggle_state[0]:
                _toggle_state[0] = False
                on_hotkey_release()
            else:
                _toggle_state[0] = True
                on_hotkey_press()

        kb.on_press_key(main_key, on_toggle, suppress=True)
        log.info(f"[HOTKEY] Registered: {hotkey} (toggle)")


# Tray icon ----------------------------------------------------------------

def create_tray_icon():
    """Create system tray icon"""
    try:
        from PIL import Image, ImageDraw
        import pystray
    except ImportError:
        log.warning("[WARN] pystray/Pillow not installed, skipping tray icon")
        return None

    def create_image(state: str = "idle"):
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if state == "recording":
            color = (220, 50, 50, 240)
        elif state == "transcribing":
            color = (240, 180, 50, 240)
        else:
            color = (50, 120, 220, 240)

        draw.rounded_rectangle((4, 4, 60, 60), radius=12, fill=color)

        # Microphone icon (simplified)
        draw.rectangle((24, 14, 40, 40), fill=(255, 255, 255, 220))
        draw.rectangle((28, 40, 36, 48), fill=(255, 255, 255, 220))
        draw.rectangle((18, 46, 46, 50), fill=(255, 255, 255, 220))

        if state == "recording":
            draw.ellipse((44, 4, 60, 20), fill=(255, 60, 60, 255))
        elif state == "transcribing":
            draw.ellipse((44, 4, 60, 20), fill=(255, 200, 50, 255))

        return img

    def on_exit(icon, item):
        icon.stop()
        _shutdown_server()
        os._exit(0)

    def on_open(icon, item):
        os.startfile(PROJECT_ROOT)

    hotkey_text = _cfg['general']['hotkey']
    model_text = _cfg['engine']['model']
    backend_text = _cfg['engine'].get('backend', 'auto')

    icon = pystray.Icon(
        "SpeakNotType",
        create_image("idle"),
        menu=pystray.Menu(
            pystray.MenuItem("SpeakNotType (running)", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(f"Hotkey: {hotkey_text}", None, enabled=False),
            pystray.MenuItem(f"Engine: {backend_text} ({model_text})", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open project folder", on_open),
            pystray.MenuItem("Exit", on_exit),
        ),
    )
    icon._image = create_image
    return icon


def update_tray_icon(state: str):
    global _tray_icon
    if _tray_icon is None:
        return
    try:
        titles = {
            "recording": "[REC] SpeakNotType - Recording...",
            "transcribing": "[TRANS] SpeakNotType - Transcribing...",
            "idle": "SpeakNotType Ready - press hotkey to speak",
        }
        _tray_icon.icon = _tray_icon._image(state)
        _tray_icon.title = titles.get(state, "SpeakNotType")
    except Exception:
        pass


# Entry point --------------------------------------------------------------

def main():
    global _recorder, _engine, _cfg, _tray_icon, _server_manager, _stream_pipeline

    _cfg = config.load_config()
    log.info("=" * 40)
    log.info("SpeakNotType v0.1.0")
    log.info("=" * 40)

    # Detect microphone
    try:
        mics = recorder.list_microphones()
        if not mics:
            log.warning("[WARN] No microphone detected!")
        else:
            log.info(f"[MIC] Detected {len(mics)} mic(s): {mics[0]['name']}")
    except Exception as e:
        log.warning(f"[WARN] Microphone detection failed: {e}")

    # Initialize recorder
    _recorder = recorder.Recorder(
        sample_rate=_cfg["recording"]["sample_rate"],
        temp_dir=_cfg["recording"]["temp_dir"],
    )

    # Determine backend
    backend = _cfg["engine"].get("backend", "auto")
    model_name = _cfg["engine"]["model"]
    engine_kwargs = {"model": model_name, "language": _cfg["general"]["language"]}

    if backend == "auto":
        # Auto-detect best engine: faster-whisper (GPU) > whisper-server (CPU) > whisper-cpp (CLI)
        detected = transcriber._detect_best_engine()

        if detected == "faster-whisper":
            engine_kwargs["backend"] = "faster-whisper"
            log.info("[ENGINE] GPU detected -> using faster-whisper")
        elif detected == "whisper-server":
            try:
                from whisper_server_manager import WhisperServerManager
                srv_cfg = _cfg.get("server", {})
                _server_manager = WhisperServerManager(
                    model_path=config.get_model_path(model_name),
                    cli_path=config.get_whisper_cli_path(),
                    language=_cfg["general"]["language"],
                    threads=srv_cfg.get("threads", 0) or None,
                    vad=srv_cfg.get("vad", False),
                )
                _server_manager.start(timeout=90)
                engine_kwargs["backend"] = "whisper-server"
                engine_kwargs["server_manager"] = _server_manager
                log.info("[ENGINE] Using whisper-server (CPU)")
            except Exception as e:
                log.warning(f"[ENGINE] Server failed, fallback: {e}")
                engine_kwargs["backend"] = None
        else:
            engine_kwargs["backend"] = detected
    else:
        engine_kwargs["backend"] = backend

    # Initialize recognition engine
    try:
        _engine = transcriber.create_engine(**engine_kwargs)
        engine_name = type(_engine).__name__.replace("Engine", "")
        log.info(f"[ENGINE] Using: {engine_name} ({model_name})")
    except transcriber.TranscriptionError as e:
        log.error(f"[ERR] {e}")
        sys.exit(1)

    # Register hotkey
    setup_hotkey()

    # Start system tray (blocking)
    _tray_icon = create_tray_icon()
    if _tray_icon:
        log.info("[TRAY] System tray icon active")
        _tray_icon.title = "SpeakNotType Ready - press hotkey to speak"
        _tray_icon.run()
    else:
        log.info("[INFO] No tray mode. Press Ctrl+C to exit.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("[EXIT] Goodbye!")
            _shutdown_server()
            sys.exit(0)


if __name__ == "__main__":
    main()