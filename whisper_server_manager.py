"""whisper-server 进程管理器 — 模型常驻内存，HTTP 调用"""

import os
import json
import time
import logging
import subprocess
import threading
import urllib.request
import urllib.error
from urllib.parse import urlencode

from typing import Final

log = logging.getLogger("SpeakNotType")

# 根据 CPU 核数自动选择线程数 (保留 4 核给系统)
_DEFAULT_THREADS: Final[int] = max(4, min(os.cpu_count() or 8, 24))


class WhisperServerError(Exception):
    """whisper-server 异常"""
    pass


class WhisperServerManager:
    """管理 whisper-server 子进程生命周期"""

    def __init__(
        self,
        model_path: str,
        cli_path: str,
        host: str = "127.0.0.1",
        port: int = 18080,
        language: str = "auto",
        threads: int | None = None,
        vad: bool = False,
    ):
        self.model_path = model_path
        self.cli_path = cli_path
        self.host = host
        self.port = port
        self.language = language
        self.threads = threads if threads is not None else _DEFAULT_THREADS
        self.vad = vad
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._ready = False
        self._base_url = f"http://{host}:{port}"

    def start(self, timeout: float = 60.0) -> bool:
        """启动 whisper-server，等待模型加载完成

        Args:
            timeout: 等待模型加载的超时秒数

        Returns:
            bool: 是否成功启动
        """
        if self._process:
            log.info("[SERVER] Server already running")
            return True

        # 查找 whisper-server.exe（可能在 whisper/ 或 whisper/Release/）
        search_dirs = [
            os.path.dirname(self.cli_path),                                         # whisper/
            os.path.join(os.path.dirname(self.cli_path), "Release"),                # whisper/Release/
        ]
        server_exe = None
        for d in search_dirs:
            candidate = os.path.join(d, "whisper-server.exe")
            if os.path.exists(candidate):
                server_exe = candidate
                server_dir = d
                break

        if not server_exe:
            raise WhisperServerError(
                f"whisper-server.exe not found in:\n"
                f"  {search_dirs[0]}\n"
                f"  {search_dirs[1]}\n"
                f"Download whisper.cpp from: https://github.com/ggml-org/whisper.cpp/releases"
            )

        if not os.path.exists(self.model_path):
            raise WhisperServerError(
                f"Model not found: {self.model_path}"
            )

        log.info("[SERVER] Starting whisper-server (large-v3-turbo)...")
        log.info(f"[SERVER]   Model:       {self.model_path}")
        log.info(f"[SERVER]   Threads:     {self.threads}")
        log.info(f"[SERVER]   Port:        {self.port}")
        vad_status = "enabled" if self.vad else "disabled (explicit)"
        log.info(f"[SERVER]   VAD:         {vad_status}")
        log.info(f"[SERVER]   Best of:     1 (fast decoding)")
        log.info(f"[SERVER]   Loading model (takes ~20s)...")

        cmd = [
            server_exe,
            "-m", self.model_path,
            "-l", self.language,
            "-t", str(self.threads),
            "--host", self.host,
            "--port", str(self.port),
            "--no-gpu",
            "-bo", "1",                    # 只保留最优候补，加速 15%
        ]
        if self.vad:
            cmd.append("--vad")            # 静音裁剪

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # Inherit DLL search path from the server directory
            cwd=server_dir,
        )

        # 轮询等待服务器就绪
        started = time.monotonic()
        poll_interval = 1.0
        last_log = 0

        while time.monotonic() - started < timeout:
            if self._process.poll() is not None:
                raise WhisperServerError(
                    f"whisper-server exited prematurely (code={self._process.returncode})"
                )

            elapsed = int(time.monotonic() - started)
            if elapsed > 0 and elapsed % 5 == 0 and elapsed != last_log:
                last_log = elapsed
                log.info(f"[SERVER]   Still loading... ({elapsed}s)")

            if self._check_health():
                self._ready = True
                load_time = time.monotonic() - started
                log.info(f"[SERVER] ✅ Model loaded in {load_time:.1f}s, ready for inference")
                return True

            time.sleep(poll_interval)

        # Timeout
        self.stop()
        raise WhisperServerError(
            f"whisper-server failed to start within {timeout:.0f}s"
        )

    def _check_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            req = urllib.request.Request(f"{self._base_url}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def transcribe(self, audio_path: str) -> str:
        """发送音频到 whisper-server 进行转录

        Args:
            audio_path: WAV 音频文件路径

        Returns:
            转录文本
        """
        if not self._ready or not self._process:
            raise WhisperServerError("Server not ready. Call start() first.")

        if not os.path.exists(audio_path):
            raise WhisperServerError(f"Audio file not found: {audio_path}")

        # 读取音频文件
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        # 构建 multipart/form-data 请求
        boundary = "----SpeakNotTypeFormBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + audio_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/inference",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise WhisperServerError(
                f"Server returned {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}"
            )
        except urllib.error.URLError as e:
            raise WhisperServerError(f"Connection to server failed: {e.reason}")
        except json.JSONDecodeError as e:
            raise WhisperServerError(f"Invalid JSON response: {e}")

        text = (result.get("text") or "").strip()
        return text

    def transcribe_bytes(self, wav_bytes: bytes) -> str:
        """直接从 WAV 字节流转录（避免临时文件）"""
        if not self._ready or not self._process:
            raise WhisperServerError("Server not ready. Call start() first.")

        boundary = "----SpeakNotTypeFormBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + wav_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/inference",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise WhisperServerError(
                f"Server returned {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}"
            )
        except urllib.error.URLError as e:
            raise WhisperServerError(f"Connection to server failed: {e.reason}")

        return (result.get("text") or "").strip()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def stop(self, timeout: float = 5.0):
        """停止 whisper-server 进程"""
        with self._lock:
            if self._process:
                log.info("[SERVER] Shutting down whisper-server...")
                self._process.terminate()
                try:
                    self._process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    log.warning("[SERVER] Force killing...")
                    self._process.kill()
                    self._process.wait()
                self._process = None
                self._ready = False
                log.info("[SERVER] Stopped")