#!/usr/bin/env python3
"""SpeakNotType one-click installer"""

import os
import sys
import subprocess
import config


def print_step(n, total, msg):
    print(f"\n[{n}/{total}] {msg}")


def check_python():
    print(f"  Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("  ERROR: Python 3.10+ required")
        sys.exit(1)
    print("  OK Python version OK")


def install_dependencies():
    print_step(1, 3, "Installing Python dependencies...")
    req_path = os.path.join(config.get_project_root(), "requirements.txt")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req_path],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("  OK Dependencies installed")
    else:
        print("  WARNING Some dependencies failed:")
        for line in result.stderr.split("\n")[-3:]:
            if line.strip():
                print(f"     {line}")


def download_whisper():
    print_step(2, 3, "Downloading whisper.cpp (optional, for best quality)...")
    print("  (This step may fail if GitHub is unreachable in your region)")
    print("  If it fails, the app will still work using Windows built-in speech.")

    sys.path.insert(0, config.get_project_root())
    try:
        from whisper.download import download_whisper_cli, download_model
        cfg = config.load_config()
        model = cfg["engine"]["model"]
        download_whisper_cli()
        download_model(model)
    except Exception as e:
        print(f"  WARNING: Could not download whisper.cpp: {e}")
        print()
        print("  Manual download option:")
        print("  1. Visit: https://github.com/ggml-org/whisper.cpp/releases")
        print("  2. Download: whisper-bin-x64.zip (Windows)")
        print("  3. Extract whisper-cli.exe into: speaknottype/whisper/")
        print("  4. Download: ggml-tiny.bin from HuggingFace")
        print("     https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin")
        print("  5. Place it in: speaknottype/whisper/models/")
        print()
        print("  The app will auto-detect whisper-cli.exe when available.")


def print_next_steps():
    print("\n" + "=" * 50)
    print("SpeakNotType installation complete!")
    print("=" * 50)
    print()
    engine_info = "(auto-detect)"
    cfg = config.load_config()
    if cfg["engine"].get("backend") and cfg["engine"]["backend"] != "auto":
        engine_info = cfg["engine"]["backend"]
    print(f"  Speech engine: {engine_info}")
    print()
    print("  To start:    python main.py")
    print("  To configure: edit config.toml")
    print()
    print("  Usage:")
    print("    1. Press Ctrl + `  (backtick, top-left of keyboard)")
    print("    2. Speak into your microphone")
    print("    3. Release hotkey - text appears at cursor")
    print()
    print("  Exit: Right-click tray icon -> Exit")


def main():
    print("=" * 50)
    print("SpeakNotType - One-Click Installer")
    print("=" * 50)
    print()

    check_python()
    install_dependencies()
    download_whisper()
    print_next_steps()


if __name__ == "__main__":
    main()