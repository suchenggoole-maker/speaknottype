"""后处理模块 - 文本清理与格式化"""

import re


def clean_text(text: str) -> str:
    """基础文本清理"""
    if not text:
        return ""

    # 1. 移除首尾空白
    text = text.strip()

    # 2. 移除多余空白（多个空格合并为一个）
    text = re.sub(r"\s+", " ", text)

    # 3. 移除特定开头（whisper 可能输出的前缀）
    text = re.sub(r"^(Silence|Background noise|Music)", "", text, flags=re.IGNORECASE).strip()

    return text


def postprocess(text: str, language: str = "auto") -> str:
    """后处理入口（MVP 版本仅做基础清理，预留后续 AI 润色/翻译接口）"""
    return clean_text(text)