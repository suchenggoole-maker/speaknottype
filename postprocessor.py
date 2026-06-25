"""后处理模块 - 保守文本清理、专有名词修正与格式化"""

import re

# Whisper 常见幻觉/无效文本。只过滤非常高置信的固定模板，避免误删用户真实内容。
HALLUCINATION_PATTERNS = [
    r"^\s*(silence|background noise|music)\s*$",
    r"^\s*thank you for watching[.!。！]*\s*$",
    r"^\s*thanks for watching[.!。！]*\s*$",
    r"^\s*字幕由.*提供\s*$",
    r"^\s*请不吝点赞.*\s*$",
]

# 如果 Whisper 把 initial_prompt 当作转写内容吐出来，直接删掉这些片段。
PROMPT_LEAK_PATTERNS = [
    r"请保留自然的中文标点[，,和及、\s]*中英文标点[。.!！]*",
    r"请保留自然的中文标点[、，,\s]*英文大小写和专有名词格式[。.!！]*",
    r"以下是带有自然中文标点的普通话和中英文混合语音转写文本[。.!！]*",
    r"你好，这是一个带有自然中文标点的转写示例[。.!！]*",
    r"这里是\s*SpeakNotType[，,]?\s*正在使用\s*Claude\s*Code[。.!！]*",
]

# 专有名词/常见误识别修正。越具体越好，避免把普通词改坏。
TERM_REPLACEMENTS = [
    # SpeakNotType variations seen in real testing
    (r"\bSpeak\s*No\s*Type(?=\b|\d)", "SpeakNotType"),
    (r"\bSpeak\s*Not\s*Type(?=\b|\d)", "SpeakNotType"),
    (r"\bSpeak\s*Note\s*Type(?=\b|\d)", "SpeakNotType"),
    (r"\bSpeak\s*Node\s*Type(?=\b|\d)", "SpeakNotType"),
    (r"\bSpeaknoType\b", "SpeakNotType"),
    (r"\bSpeakNoType\b", "SpeakNotType"),
    (r"\bSpeakNodeType\b", "SpeakNotType"),
    (r"\bSpeakNotype\b", "SpeakNotType"),
    (r"\bspeak\s*no\s*type\b", "SpeakNotType"),
    (r"\bspeak\s*not\s*type\b", "SpeakNotType"),
    (r"\bspeak\s*note\s*type\b", "SpeakNotType"),
    (r"\bspeak\s*node\s*type\b", "SpeakNotType"),

    # Tools / model names
    (r"\bClaude\s*Code\b", "Claude Code"),
    (r"\bclaude\s*code\b", "Claude Code"),
    (r"\bfast\s*whisper\b", "faster-whisper"),
    (r"\bfaster\s*whisper\b", "faster-whisper"),
    (r"\bwhisper\s*cpp\b", "whisper.cpp"),
    (r"\bwhisper\.\s+cpp\b", "whisper.cpp"),
    (r"\bGithub\b", "GitHub"),
    (r"\bgithub\b", "GitHub"),
    (r"\bPython\b", "Python"),
    (r"\bCUDA\b", "CUDA"),
]


def _is_hallucination(text: str) -> bool:
    for pattern in HALLUCINATION_PATTERNS:
        if re.match(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def clean_text(text: str) -> str:
    """基础文本清理"""
    if not text:
        return ""

    text = text.strip()
    if not text:
        return ""

    if _is_hallucination(text):
        return ""

    # 多空白合并为一个空格；换行先视作空格，避免粘贴到命令行多行执行。
    text = re.sub(r"\s+", " ", text)

    # 移除 whisper 可能输出的开头说明。
    text = re.sub(r"^(Silence|Background noise|Music)\s*", "", text, flags=re.IGNORECASE).strip()

    return text


def remove_prompt_leaks(text: str) -> str:
    """删除 Whisper 偶尔泄露出来的 initial_prompt 片段。"""
    for pattern in PROMPT_LEAK_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_terms(text: str) -> str:
    """专有名词/常见误识别修正"""
    for pattern, replacement in TERM_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_spacing(text: str) -> str:
    """保守空格/标点格式化，不做语义改写"""
    if not text:
        return ""

    # 中英文/数字之间加空格：中文SpeakNotType22 → 中文 SpeakNotType 22
    text = re.sub(r"([一-鿿])([A-Za-z0-9])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z0-9])([一-鿿])", r"\1 \2", text)

    # 标点前去空格，标点后单空格（中文标点后不强制空格）
    text = re.sub(r"\s+([,.;:!?，。！？；：])", r"\1", text)
    text = re.sub(r"([,.;:!?])([^\s一-鿿])", r"\1 \2", text)

    # 清理多余空格
    text = re.sub(r" {2,}", " ", text).strip()
    return text


def postprocess(text: str, language: str = "auto") -> str:
    """后处理入口：保守处理，不做数字转换、不做 AI 润色。"""
    text = clean_text(text)
    if not text:
        return ""
    text = remove_prompt_leaks(text)
    if not text:
        return ""
    text = normalize_terms(text)
    text = normalize_spacing(text)
    # Spacing rules may split terms such as whisper.cpp -> whisper. cpp; normalize once more.
    text = normalize_terms(text)
    return text
