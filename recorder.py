"""录音模块 - 使用 sounddevice 录制麦克风输入"""

import os
import tempfile
import time
import threading
import logging
import sounddevice as sd
import soundfile as sf
import numpy as np

log = logging.getLogger("SpeakNotType")


class Recorder:
    """录音器，支持按说松停的 push-to-talk 模式"""

    def __init__(self, sample_rate: int = 16000, temp_dir: str = ""):
        self.sample_rate = sample_rate
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self._recording = False
        self._audio_data: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def start_recording(self):
        """开始录音：只有按下热键后才打开麦克风流"""
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._audio_data = []

            def callback(indata, frames, time_info, status):
                if self._recording:
                    self._audio_data.append(indata.copy())

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=callback,
                blocksize=self.sample_rate // 50,  # 20ms，降低设备启动后的录音延迟
            )
            self._stream.start()

    @staticmethod
    def _trim_silence(audio: np.ndarray,
                       sample_rate: int = 16000,
                       threshold: float = 0.01,
                       min_silence_ms: int = 300,
                       padding_ms: int = 150) -> np.ndarray:
        """能量检测法去除首尾静音，保留中间人声"""
        if len(audio) < sample_rate // 4:  # 太短不裁
            return audio

        # 计算每帧能量（每帧 10ms）
        frame_len = sample_rate // 100  # 10ms
        n_frames = len(audio) // frame_len
        if n_frames < 2:
            return audio

        # 裁到整数帧
        audio = audio[:n_frames * frame_len]
        frames = audio.reshape(n_frames, frame_len)
        energy = np.abs(frames).max(axis=1)

        # 找到超过阈值的帧范围
        voice_frames = np.where(energy > threshold)[0]
        if len(voice_frames) == 0:
            return audio  # 没有检测到声音，保留原样

        start_frame = max(0, voice_frames[0] - padding_ms // 10)
        end_frame = min(n_frames, voice_frames[-1] + padding_ms // 10)

        trimmed = audio[start_frame * frame_len: end_frame * frame_len]
        return trimmed

    def stop_recording(self) -> str | None:
        """停止录音，返回保存的音频文件路径"""
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

        if not self._audio_data:
            return None

        # 计算录音实际时长
        audio = np.concatenate(self._audio_data, axis=0)
        duration_sec = len(audio) / self.sample_rate

        # 静音裁剪：去除首尾静音段，减少推理量
        trimmed = self._trim_silence(audio, self.sample_rate)
        trimmed_sec = len(trimmed) / self.sample_rate
        if trimmed_sec < duration_sec:
            log.info(
                f"[PERF] 静音裁剪: {duration_sec:.1f}s → {trimmed_sec:.1f}s "
                f"(节省 {duration_sec - trimmed_sec:.1f}s)"
            )
            audio = trimmed
            duration_sec = trimmed_sec

        from perf_timer import PerfTimer
        _t = PerfTimer("Recorder")
        _t.mark("recording_duration")

        # 保存到临时文件
        timestamp = int(time.time() * 1000)
        file_path = os.path.join(self.temp_dir, f"snt_recording_{timestamp}.wav")
        sf.write(file_path, audio, self.sample_rate)
        save_ms = _t.mark("save_wav") * 1000

        log.info(
            f"[PERF] 录音: {duration_sec:.1f}s | "
            f"音频保存: {save_ms:.0f}ms | "
            f"文件大小: {os.path.getsize(file_path)/1024:.0f}KB"
        )
        return file_path

    @property
    def is_recording(self) -> bool:
        return self._recording

    def list_devices(self):
        """列出可用的录音设备"""
        return sd.query_devices()


def list_microphones():
    """列出所有麦克风设备"""
    devices = sd.query_devices()
    mics = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            mics.append({"id": i, "name": dev["name"], "channels": dev["max_input_channels"]})
    return mics
