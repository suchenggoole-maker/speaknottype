"""whisper.cpp 自动下载模块"""

import os
import sys
import requests
import zipfile
import io
import config

# GitHub Releases 下载地址
WHISPER_CPP_RELEASE = "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.0"

# Windows 预编译包（标准 CPU 构建，约 5.4 MB）
WHISPER_CPP_WIN_ZIP = "whisper-bin-x64.zip"

# 模型下载地址 (HuggingFace)
MODEL_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

MODEL_FILES = {
    "tiny":   "ggml-tiny.bin",
    "base":   "ggml-base.bin",
    "small":  "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large":  "ggml-large-v3.bin",
}

MODEL_SIZES = {
    "tiny":   "~75 MB",
    "base":   "~150 MB",
    "small":  "~500 MB",
    "medium": "~1.5 GB",
    "large":  "~3 GB",
}


def progress_bar(description: str, current: int, total: int):
    """简单的进度条（ASCII only，兼容 Windows GBK）"""
    if total <= 0:
        print(f"\r{description}: {current // 1024 // 1024} MB downloaded...", end="")
        return
    pct = current / total * 100
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    mb_dl = current / 1024 / 1024
    mb_total = total / 1024 / 1024
    print(f"\r{description}: [{bar}] {mb_dl:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="")
    if current >= total:
        print()


def download_with_progress(url: str, dest_path: str, desc: str = "Downloading"):
    """带进度条的文件下载"""
    print(f"\n[SpeakNotType] {desc}...")
    print(f"  URL: {url}")
    print(f"  Save to: {dest_path}")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                progress_bar(desc, downloaded, total_size)

    print()
    return dest_path


def download_whisper_cli() -> str:
    """下载 whisper-cli.exe"""
    dest_dir = config.get_whisper_dir()
    zip_path = os.path.join(dest_dir, WHISPER_CPP_WIN_ZIP)

    # 下载 zip 包
    url = f"{WHISPER_CPP_RELEASE}/{WHISPER_CPP_WIN_ZIP}"
    download_with_progress(url, zip_path, "Downloading whisper.cpp (Windows)")

    # 解压
    print(f"\n[SpeakNotType] Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    # 删除 zip
    os.remove(zip_path)

    # 查找解压后的 exe（可能在 Release/ 子目录中）
    exe_path = config.get_whisper_cli_path()
    import shutil

    for root, dirs, files in os.walk(dest_dir):
        for f in files:
            if f == "whisper-cli.exe":
                found = os.path.join(root, f)
                # Copy the largest one (real binary, not stub)
                if not os.path.exists(exe_path) or os.path.getsize(found) > os.path.getsize(exe_path):
                    shutil.copy2(found, exe_path)

    if os.path.exists(exe_path):
        kb = os.path.getsize(exe_path) / 1024
        print(f"  OK whisper-cli.exe ready: {exe_path} ({kb:.0f} KB)")
        return exe_path
    else:
        print(f"  WARNING exe not found after extraction, check manually: {dest_dir}")
        return ""


def download_model(model_name: str = "tiny") -> str:
    """下载指定模型"""
    if model_name not in MODEL_FILES:
        print(f"  ⚠️  未知模型: {model_name}，可用: {list(MODEL_FILES.keys())}")
        return ""

    filename = MODEL_FILES[model_name]
    dest_path = os.path.join(config.get_models_dir(), filename)

    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / 1024 / 1024
        print(f"  OK model exists: {dest_path} ({size_mb:.1f} MB)")
        return dest_path

    url = f"{MODEL_BASE_URL}/{filename}"
    size = MODEL_SIZES.get(model_name, "unknown")
    download_with_progress(url, dest_path, f"Downloading {model_name} model ({size})")
    return dest_path


if __name__ == "__main__":
    print("=" * 50)
    print("SpeakNotType - whisper.cpp download tool")
    print("=" * 50)

    # 默认下载 tiny 模型和 whisper-cli
    download_whisper_cli()
    download_model("tiny")

    print("\nDone!")
    print("Run 'python main.py' to start SpeakNotType")