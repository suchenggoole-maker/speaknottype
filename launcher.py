#!/usr/bin/env python3
"""SpeakNotType 启动器 — 环境检测 + 配置 + 一键启动"""

import os
import sys
import json
import platform
import subprocess
import threading

# 获取项目根目录（兼容 PyInstaller 打包模式）
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = os.path.dirname(sys.executable)
else:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

import config as cfg_module

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    print("[ERR] tkinter not available. Install python3-tk or use config.toml directly.")
    sys.exit(1)


# ── 环境检测 ────────────────────────────────────────────────────────────

def detect_env() -> dict:
    """检测系统环境"""
    info = {}

    # CPU
    info["cpu_count"] = os.cpu_count() or 0

    # whisper binary
    whisper_dir = os.path.join(PROJECT_ROOT, "whisper")
    cli_exe = os.path.join(whisper_dir, "whisper-cli.exe")
    cli_release = os.path.join(whisper_dir, "Release", "whisper-cli.exe")
    info["cli_ok"] = os.path.exists(cli_exe) or os.path.exists(cli_release)

    server_exe = os.path.join(whisper_dir, "Release", "whisper-server.exe")
    info["server_ok"] = os.path.exists(server_exe)

    # Available models
    models_dir = os.path.join(whisper_dir, "models")
    ct2_dir = os.path.join(whisper_dir, "models-ct2")
    models = []
    ct2_models = []

    # whisper.cpp models (ggml-*.bin)
    if os.path.isdir(models_dir):
        for f in sorted(os.listdir(models_dir)):
            if f.startswith("ggml-") and f.endswith(".bin"):
                name = f.replace("ggml-", "").replace(".bin", "")
                size = os.path.getsize(os.path.join(models_dir, f))
                models.append((name, size))

    # faster-whisper CTranslate2 models (subdirs containing model.bin)
    if os.path.isdir(ct2_dir):
        for d in sorted(os.listdir(ct2_dir)):
            mp = os.path.join(ct2_dir, d, "model.bin")
            if os.path.isfile(mp):
                ct2_models.append((d, os.path.getsize(mp)))

    info["models"] = models
    info["ct2_models"] = ct2_models

    # GPU detection
    info["has_cuda"] = False
    info["gpu_name"] = ""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            info["has_cuda"] = True
            info["gpu_name"] = f"CUDA ({ctranslate2.get_cuda_device_count()} 设备)"
    except Exception:
        pass

    # Python version
    info["python"] = sys.version.split()[0]

    # Current config
    info["config"] = cfg_module.load_config()

    # OS
    info["os"] = platform.system()

    return info


# ── 启动器 UI ───────────────────────────────────────────────────────────

def size_str(bytes_: int) -> str:
    mb = bytes_ / 1024 / 1024
    if mb > 1000:
        return f"{mb/1024:.1f}GB"
    return f"{mb:.0f}MB"


class Launcher:
    def __init__(self):
        self.env = detect_env()
        self.root = tk.Tk()
        self.root.title("SpeakNotType 启动器")
        # Allow resize for low DPI screens
        self.root.resizable(True, True)
        self.root.minsize(560, 500)
        # Initial size
        self.root.geometry("580x760")
        # Center on screen
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        # Limit height to 90% of screen
        max_h = int(sh * 0.9)
        actual_h = min(760, max_h)
        x = (sw - 580) // 2
        y = max(20, (sh - actual_h) // 2)
        self.root.geometry(f"580x{actual_h}+{x}+{y}")

        self._build_ui()

    def _build_ui(self):
        # Outer container: top scrollable area + bottom fixed buttons
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        # Bottom button bar (packed FIRST so it's always reserved space)
        bottom = ttk.Frame(outer, padding=(16, 8, 16, 12))
        bottom.pack(side="bottom", fill="x")

        # Separator above the buttons
        ttk.Separator(outer, orient="horizontal").pack(side="bottom", fill="x")

        # Status bar above buttons
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(bottom, textvariable=self.status_var,
                               font=("Segoe UI", 9), foreground="gray")
        status_bar.pack(side="bottom", fill="x", pady=(6, 0))

        # Buttons
        btn_row = ttk.Frame(bottom)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="保存并启动", command=self._on_launch,
                   width=18).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="退出", command=self._on_exit,
                   width=10).pack(side="left")

        # Scrollable content area
        canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        main = ttk.Frame(canvas, padding=16)
        canvas_window = canvas.create_window((0, 0), window=main, anchor="nw")

        def _on_resize(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", _on_resize)
        main.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── Header ──
        header = ttk.Frame(main)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="SpeakNotType",
                  font=("Segoe UI", 18, "bold")).pack(side="left")
        ttk.Label(header, text="v0.1.0",
                  font=("Segoe UI", 10)).pack(side="left", padx=(6, 0))

        # ── Environment section ──
        self._build_section(main, "环境检测", self._build_env)

        # ── Configuration section ──
        self._build_section(main, "参数配置", self._build_config)

    def _build_section(self, parent, title, builder):
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.pack(fill="x", pady=(0, 10))
        builder(frame)

    def _build_env(self, parent):
        env = self.env
        rows = [
            ("操作系统", f"{env['os']}"),
            ("Python", f"{env['python']}"),
            ("CPU 核心数", f"{env['cpu_count']} 核"),
            ("GPU (CUDA)", "✅ " + env["gpu_name"] if env["has_cuda"] else "❌ 不可用"),
            ("whisper-cli", "✅ 已就绪" if env["cli_ok"] else "❌ 未找到"),
            ("whisper-server", "✅ 已就绪" if env["server_ok"] else "❌ 未找到"),
        ]
        for i, (k, v) in enumerate(rows):
            ttk.Label(parent, text=k, width=14, anchor="e").grid(row=i, column=0, sticky="e", padx=(0, 6), pady=1)
            color = "green" if v.startswith("✅") else ("red" if v.startswith("❌") else "black")
            lbl = ttk.Label(parent, text=v, foreground=color)
            lbl.grid(row=i, column=1, sticky="w", pady=1)

        # Available models (both ggml and ct2)
        ttk.Label(parent, text="可用模型", width=14, anchor="e").grid(
            row=len(rows), column=0, sticky="ne", padx=(0, 6), pady=(4, 0))
        model_frame = ttk.Frame(parent)
        model_frame.grid(row=len(rows), column=1, sticky="w", pady=(4, 0))

        if env["models"]:
            ttk.Label(model_frame, text="whisper.cpp (CPU/Server):",
                      foreground="gray", font=("Segoe UI", 9)).pack(anchor="w")
            for name, size in env["models"]:
                ttk.Label(model_frame, text=f"  • {name}  ({size_str(size)})",
                          foreground="green").pack(anchor="w")
        if env["ct2_models"]:
            ttk.Label(model_frame, text="faster-whisper (GPU/CPU):",
                      foreground="gray", font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))
            for name, size in env["ct2_models"]:
                ttk.Label(model_frame, text=f"  • {name}  ({size_str(size)})",
                          foreground="blue").pack(anchor="w")
        if not env["models"] and not env["ct2_models"]:
            ttk.Label(model_frame, text="  无模型文件", foreground="red").pack(anchor="w")

    def _build_config(self, parent):
        cfg = self.env["config"]

        # Model selection - shows both ggml and ct2 models with clear labels
        row = 0
        ttk.Label(parent, text="模型", width=10, anchor="e").grid(row=row, column=0, sticky="e", padx=(0, 8), pady=2)
        models_names = []
        for m in self.env["models"]:
            models_names.append(f"{m[0]} [whisper.cpp]")
        for m in self.env["ct2_models"]:
            models_names.append(f"{m[0]} [faster-whisper-GPU]")
        if not models_names:
            models_names = ["large-v3-turbo"]

        # Default to GPU model if available
        if self.env["has_cuda"] and self.env["ct2_models"]:
            default_model = f"{self.env['ct2_models'][0][0]} [faster-whisper-GPU]"
        else:
            default_model = models_names[0]
        self.model_var = tk.StringVar(value=default_model)
        model_combo = ttk.Combobox(parent, textvariable=self.model_var, values=models_names,
                                   state="readonly", width=30)
        model_combo.grid(row=row, column=1, sticky="w", pady=2)

        # Threads
        row += 1
        ttk.Label(parent, text="线程数", width=10, anchor="e").grid(row=row, column=0, sticky="e", padx=(0, 8), pady=2)
        thread_frame = ttk.Frame(parent)
        thread_frame.grid(row=row, column=1, sticky="w", pady=2)
        max_cores = self.env["cpu_count"]
        self.thread_var = tk.IntVar(value=max(4, min(max_cores, 24)))
        thread_scale = ttk.Scale(thread_frame, from_=1, to=max_cores, variable=self.thread_var,
                                 orient="horizontal", length=180)
        thread_scale.pack(side="left")
        self.thread_label = ttk.Label(thread_frame, text=f"  {self.thread_var.get()}")
        self.thread_label.pack(side="left", padx=(4, 0))
        # Auto-detect label
        ttk.Label(thread_frame, text=f"(最大 {max_cores})", foreground="gray").pack(side="left", padx=(4, 0))
        def _update_thread_label(*_):
            self.thread_label.config(text=f"  {self.thread_var.get()}")
        self.thread_var.trace_add("write", _update_thread_label)

        # VAD
        row += 1
        ttk.Label(parent, text="静音裁剪", width=10, anchor="e").grid(row=row, column=0, sticky="e", padx=(0, 8), pady=2)
        self.vad_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="VAD 自动跳过静音段", variable=self.vad_var).grid(
            row=row, column=1, sticky="w", pady=2)

        # Language
        row += 1
        ttk.Label(parent, text="语言", width=10, anchor="e").grid(row=row, column=0, sticky="e", padx=(0, 8), pady=2)
        self.lang_var = tk.StringVar(value=cfg["general"].get("language", "auto"))
        lang_combo = ttk.Combobox(parent, textvariable=self.lang_var,
                                  values=["auto", "zh", "en"], state="readonly", width=24)
        lang_combo.grid(row=row, column=1, sticky="w", pady=2)

        # Mode
        row += 1
        ttk.Label(parent, text="模式", width=10, anchor="e").grid(row=row, column=0, sticky="e", padx=(0, 8), pady=2)
        self.mode_var = tk.StringVar(value=cfg["general"].get("mode", "push-to-talk"))
        mode_combo = ttk.Combobox(parent, textvariable=self.mode_var,
                                  values=["push-to-talk", "toggle"], state="readonly", width=24)
        mode_combo.grid(row=row, column=1, sticky="w", pady=2)

        # Hotkey
        row += 1
        ttk.Label(parent, text="热键", width=10, anchor="e").grid(row=row, column=0, sticky="e", padx=(0, 8), pady=2)
        self.hotkey_var = tk.StringVar(value=cfg["general"].get("hotkey", "ctrl+grave"))
        ttk.Entry(parent, textvariable=self.hotkey_var, width=26).grid(row=row, column=1, sticky="w", pady=2)

        # Auto paste
        row += 1
        ttk.Label(parent, text="输出", width=10, anchor="e").grid(row=row, column=0, sticky="e", padx=(0, 8), pady=2)
        self.paste_var = tk.BooleanVar(value=cfg["output"].get("auto_paste", True))
        ttk.Checkbutton(parent, text="自动粘贴到光标位置", variable=self.paste_var).grid(
            row=row, column=1, sticky="w", pady=2)

        # Backend mode (server / CLI)
        row += 1
        ttk.Label(parent, text="引擎模式", width=10, anchor="e").grid(row=row, column=0, sticky="e", padx=(0, 8), pady=2)
        # Check if CUDA is available for faster-whisper
        has_cuda = False
        try:
            import ctranslate2
            has_cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            pass
        default_backend = "auto (faster-whisper GPU)" if has_cuda else "auto (server 优先)"
        self.backend_var = tk.StringVar(value=default_backend)

        backend_options = []
        if has_cuda:
            backend_options.append("auto (faster-whisper GPU)")
        if self.env["server_ok"]:
            backend_options.append("auto (server 优先)")
        backend_options.append("whisper-cpp (CLI, 省内存)")

        backend_combo = ttk.Combobox(parent, textvariable=self.backend_var,
                                     values=backend_options,
                                     state="readonly", width=28)
        backend_combo.grid(row=row, column=1, sticky="w", pady=2)

    # ── Actions ──

    def _write_config(self):
        """将 UI 配置写入 config.toml"""
        backend_raw = self.backend_var.get()
        model_raw = self.model_var.get()

        # Parse model name + backend hint
        if "[faster-whisper-GPU]" in model_raw:
            backend = "faster-whisper"
            model_name = model_raw.replace(" [faster-whisper-GPU]", "")
        elif "[whisper.cpp]" in model_raw:
            model_name = model_raw.replace(" [whisper.cpp]", "")
            # Backend depends on dropdown
            if "faster-whisper" in backend_raw or "GPU" in backend_raw:
                backend = "faster-whisper"  # but user picked CPU model - may fail
                model_name = "medium"  # force GPU model
            elif "server" in backend_raw:
                backend = "auto"
            else:
                backend = "whisper-cpp"
        else:
            # Plain model name (legacy)
            model_name = model_raw
            if "faster-whisper" in backend_raw or "GPU" in backend_raw:
                backend = "faster-whisper"
            elif "server" in backend_raw:
                backend = "auto"
            else:
                backend = "whisper-cpp"

        # Build config dict
        cfg = {
            "general": {
                "hotkey": self.hotkey_var.get().strip(),
                "language": self.lang_var.get(),
                "mode": self.mode_var.get(),
            },
            "engine": {
                "backend": backend,
                "model": model_name,
            },
            "output": {
                "auto_paste": self.paste_var.get(),
            },
            "recording": {
                "sample_rate": 16000,
                "format": "wav",
                "temp_dir": "",
            },
            "server": {
                "threads": self.thread_var.get(),
                "vad": False,  # VAD 暂不稳定，默认关闭
            },
        }

        # Write to config.toml
        config_path = os.path.join(PROJECT_ROOT, "config.toml")
        lines = []
        for section, values in cfg.items():
            lines.append(f"[{section}]")
            for k, v in values.items():
                if isinstance(v, bool):
                    lines.append(f"{k} = {'true' if v else 'false'}")
                elif isinstance(v, int):
                    lines.append(f"{k} = {v}")
                elif isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
            lines.append("")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return cfg

    def _on_launch(self):
        self.status_var.set("正在保存配置...")
        self.root.update()

        try:
            cfg = self._write_config()
        except Exception as e:
            self.status_var.set(f"❌ 配置保存失败: {e}")
            return

        self.status_var.set("✅ 配置已保存，启动中...")
        self.root.update()

        # 完全分离子进程，避免 PyInstaller 清理临时目录冲突
        main_py = os.path.join(PROJECT_ROOT, "main.py")
        python_exe = os.path.join(sys.base_exec_prefix, "python.exe")
        if not os.path.exists(python_exe):
            python_exe = "python"
        try:
            if sys.platform == "win32":
                startup = subprocess.STARTUPINFO()
                startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                subprocess.Popen(
                    [python_exe, main_py],
                    cwd=PROJECT_ROOT,
                    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW,
                    startupinfo=startup,
                )
            else:
                subprocess.Popen(
                    [python_exe, main_py],
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            self.root.after(200, self.root.destroy)
        except Exception as e:
            self.status_var.set(f"❌ 启动失败: {e}")
            return

        # Show success, close after 2s
        self.root.after(2000, self._close_after_launch)

    def _close_after_launch(self):
        self.root.destroy()

    def _on_exit(self):
        self.root.destroy()
        sys.exit(0)

    def run(self):
        self.root.mainloop()


def main():
    launcher = Launcher()
    launcher.run()


if __name__ == "__main__":
    main()