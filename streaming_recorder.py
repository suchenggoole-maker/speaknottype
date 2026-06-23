"""流式录音器 - 基于静音检测分段，触发并行识别"""

import os
import time
import threading
import logging
import tempfile
import queue
from concurrent.futures import Future
from typing import Callable
import numpy as np
import sounddevice as sd
import soundfile as sf

log = logging.getLogger("SpeakNotType")


class StreamingRecorder:
    """流式录音器
    - 实时检测静音，按句子分段
    - 每检测到一个完整段落，立即触发回调（异步识别）
    - stop_recording 时返回最后一段
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        temp_dir: str = "",
        # 静音检测参数
        silence_threshold: float = 0.015,   # 能量阈值（小于此值视为静音）
        silence_duration_ms: int = 1200,    # 静音 1.2s 才算段落结束（避免过度切分）
        min_chunk_duration_ms: int = 5000,  # 段落最短 5s（小于此值不触发提前识别）
        max_chunk_duration_ms: int = 12000, # 段落最长 12s（强制切分）
        chunk_handler: Callable[[np.ndarray, bool], Future] = None,
    ):
        self.sample_rate = sample_rate
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.silence_threshold = silence_threshold
        self.silence_frames_needed = silence_duration_ms * sample_rate // 1000
        self.min_chunk_frames = min_chunk_duration_ms * sample_rate // 1000
        self.max_chunk_frames = max_chunk_duration_ms * sample_rate // 1000
        self.chunk_handler = chunk_handler

        self._recording = False
        self._all_audio: list[np.ndarray] = []      # 完整录音备份
        self._current_chunk: list[np.ndarray] = []  # 当前段落累积
        self._current_chunk_frames = 0
        self._silence_frames = 0
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._chunk_futures: list[Future] = []      # 已派发的识别任务
        self._chunk_count = 0

    def start_recording(self):
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._all_audio = []
            self._current_chunk = []
            self._current_chunk_frames = 0
            self._silence_frames = 0
            self._chunk_futures = []
            self._chunk_count = 0

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=self.sample_rate // 50,  # 20ms blocks
            )
            self._stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        if not self._recording:
            return

        # 保留完整录音
        chunk_data = indata.copy()
        self._all_audio.append(chunk_data)
        self._current_chunk.append(chunk_data)
        self._current_chunk_frames += frames

        # 检测能量
        energy = float(np.abs(chunk_data).max())

        if energy < self.silence_threshold:
            self._silence_frames += frames
        else:
            self._silence_frames = 0

        # 触发条件：静音够久 + 段落够长，或段落超长
        should_emit = (
            self._silence_frames >= self.silence_frames_needed
            and self._current_chunk_frames >= self.min_chunk_frames
        ) or self._current_chunk_frames >= self.max_chunk_frames

        if should_emit:
            self._emit_chunk(is_final=False)
            self._silence_frames = 0

    def _emit_chunk(self, is_final: bool):
        """发出当前段落给识别器"""
        if not self._current_chunk:
            return
        chunk_audio = np.concatenate(self._current_chunk, axis=0).flatten()

        # 跳过纯静音段
        if not is_final and float(np.abs(chunk_audio).max()) < self.silence_threshold:
            self._current_chunk = []
            self._current_chunk_frames = 0
            return

        if self.chunk_handler:
            self._chunk_count += 1
            label = f"#{self._chunk_count}" + ("(final)" if is_final else "")
            duration = len(chunk_audio) / self.sample_rate
            log.info(f"[STREAM] 段落 {label}: {duration:.1f}s, 派发识别")
            future = self.chunk_handler(chunk_audio, is_final)
            if future is not None:
                self._chunk_futures.append(future)

        self._current_chunk = []
        self._current_chunk_frames = 0

    def stop_recording(self) -> tuple[list[Future], str | None]:
        """停止录音
        Returns:
            (chunk_futures, full_audio_path):
                chunk_futures: 所有已派发的识别 Future 列表
                full_audio_path: 完整录音保存路径（备份用）
        """
        with self._lock:
            if not self._recording:
                return [], None
            self._recording = False
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

        if not self._all_audio:
            return [], None

        # 把最后一段发出去（无论长度）
        if self._current_chunk:
            chunk_audio = np.concatenate(self._current_chunk, axis=0).flatten()
            if len(chunk_audio) > self.sample_rate // 10:  # >100ms 才有意义
                if self.chunk_handler:
                    self._chunk_count += 1
                    duration = len(chunk_audio) / self.sample_rate
                    log.info(f"[STREAM] 段落 #{self._chunk_count}(final): {duration:.1f}s, 派发识别")
                    future = self.chunk_handler(chunk_audio, True)
                    if future is not None:
                        self._chunk_futures.append(future)
            self._current_chunk = []

        # 保存完整录音作为备份
        full_audio = np.concatenate(self._all_audio, axis=0)
        duration_sec = len(full_audio) / self.sample_rate
        timestamp = int(time.time() * 1000)
        file_path = os.path.join(self.temp_dir, f"snt_recording_{timestamp}.wav")
        sf.write(file_path, full_audio, self.sample_rate)
        log.info(f"[STREAM] 完整录音: {duration_sec:.1f}s, {self._chunk_count} 段")

        return list(self._chunk_futures), file_path

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def chunk_count(self) -> int:
        return self._chunk_count