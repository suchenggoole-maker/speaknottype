"""流式识别管道 - 段落到识别 Future 的桥梁"""

import io
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
import soundfile as sf

log = logging.getLogger("SpeakNotType")


class StreamingPipeline:
    """协调录音段落和 server 推理的管道
    - 录音器吐出 numpy 段 → 转 WAV bytes → 提交到线程池
    - 线程池单线程串行（server 一次只能处理一个请求）
    """

    def __init__(self, server_manager, sample_rate: int = 16000):
        self.server = server_manager
        self.sample_rate = sample_rate
        # 单线程池：server 一次只能处理一个请求
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snt-infer")
        self._lock = threading.Lock()

    def submit_chunk(self, audio: np.ndarray, is_final: bool) -> Future:
        """提交一个音频段供识别，立即返回 Future"""
        # 把 numpy 编码为 WAV bytes (在主线程做，开销小)
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        future = self._executor.submit(self._do_transcribe, wav_bytes, is_final)
        return future

    def _do_transcribe(self, wav_bytes: bytes, is_final: bool) -> str:
        import time
        t0 = time.monotonic()
        text = self.server.transcribe_bytes(wav_bytes)
        elapsed = (time.monotonic() - t0) * 1000
        log.info(f"[STREAM] 识别完成 ({elapsed:.0f}ms): [{text}]")
        return text

    def shutdown(self):
        self._executor.shutdown(wait=False)