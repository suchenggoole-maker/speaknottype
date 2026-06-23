"""输出模块 - 剪贴板 + 模拟键盘粘贴"""

import time
import logging
import pyperclip
import keyboard
import postprocessor


def output_text(text: str, auto_paste: bool = True) -> bool:
    """
    将文本输出到当前光标位置

    Args:
        text: 要输出的文本
        auto_paste: 是否自动粘贴（否则只写入剪贴板）

    Returns:
        是否成功
    """
    if not text:
        return False

    import time as _time
    from perf_timer import PerfTimer
    _t = PerfTimer("Output")
    log = logging.getLogger("SpeakNotType")
    _t.mark("start")

    # 先清理文本
    cleaned = postprocessor.postprocess(text)
    clean_ms = _t.mark("postprocess") * 1000

    try:
        # 写入剪贴板
        pyperclip.copy(cleaned)
    except Exception as e:
        log.error(f"[ERR] 剪贴板写入失败: {e}")
        return False

    clip_ms = _t.mark("clipboard") * 1000

    if auto_paste:
        # 等待剪贴板生效
        _time.sleep(0.1)
        # 模拟 Ctrl+V 粘贴
        keyboard.press_and_release("ctrl+v")
        # 给系统时间处理粘贴
        _time.sleep(0.05)
        paste_ms = _t.mark("paste") * 1000
    else:
        paste_ms = 0

    log.info(
        f"[PERF] 输出: 后处理{clean_ms:.0f}ms | "
        f"剪贴板{clip_ms:.0f}ms | "
        f"粘贴{paste_ms:.0f}ms | "
        f"共{len(cleaned)}字符"
    )
    return True