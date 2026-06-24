# SpeakNotType 🎤

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)]()

**全局语音输入工具** — 在任何应用中，按住热键说话，松手自动转文字并粘贴到光标位置。

```
按 Ctrl+` → 说话 → 松手 → 文字自动出现在光标处
```

支持中英文混合识别，完全本地运行（无需联网，保护隐私）。

---

## ✨ 特性

- 🚀 **极速识别**：RTX 50 系列 GPU 上 12 秒语音 **3 秒**出结果
- 🎯 **高准确率**：基于 Whisper large-v3 / medium 模型
- 🔒 **完全离线**：无需联网，无数据上传
- 💻 **多后端支持**：faster-whisper (GPU) / whisper.cpp (CPU)
- 🌐 **中英文混合**：识别 SpeakNoType 等专业名词无压力
- 🖥️ **系统托盘**：后台运行，热键随时调用
- ⚙️ **图形化启动器**：环境检测、参数配置、一键启动

---

## 📊 性能参考

测试环境：RTX 5080 + i9 24 核

| 后端 | 模型 | 12 秒语音处理时间 |
|------|------|---------------|
| faster-whisper (GPU) | medium | **~3 秒** ⚡ |
| faster-whisper (GPU) | large-v3 | ~5 秒 |
| whisper-server (24 线程 CPU) | large-v3-turbo | ~5 秒 |
| whisper-cli (4 线程 CPU) | large-v3-turbo | ~22 秒 |

---

## 🚀 快速开始

### 方式 A：下载预编译版（推荐）

1. 到 [Releases](https://github.com/suchenggoole-maker/speaknottype/releases) 下载 `SpeakNotType.exe`
2. 下载模型文件（任选其一）：
   - **faster-whisper medium** (1.5GB，推荐):
     ```bash
     # 用 HF 镜像加速下载（国内推荐）
     HF_ENDPOINT=https://hf-mirror.com python -c "
     from huggingface_hub import snapshot_download
     snapshot_download('Systran/faster-whisper-medium',
                        local_dir='whisper/models-ct2/medium')
     "
     ```
3. 双击 `SpeakNotType.exe` 启动

### 方式 B：从源码运行

```bash
# 1. 克隆仓库
git clone https://github.com/suchenggoole-maker/speaknottype.git
cd speaknottype

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 下载模型（同上）

# 4. 复制配置
cp config.example.toml config.toml

# 5. 启动
python launcher.py     # 图形化启动器（推荐）
# 或
python main.py         # 直接运行（用 config.toml）
```

---

## 🛠️ 系统要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| 操作系统 | Windows 10/11 | Windows 11 |
| Python | 3.10+ | 3.12 |
| 内存 | 4 GB | 8 GB+ |
| GPU | 可选 | NVIDIA RTX 30/40/50 系列 |

### GPU 加速要求

- NVIDIA 驱动 ≥ 525
- CUDA 12.x 支持（faster-whisper 通过 CTranslate2 自动适配）

---

## 🎯 使用方法

1. 启动后系统托盘出现图标（🔵 蓝色 = 待命）
2. **按住 Ctrl + `**（反引号键，Esc 下方）
3. 对着麦克风说话（图标变 🔴 红色）
4. 松开按键 → 转录中（🟡 黄色）
5. 文字自动粘贴到光标处

**退出**：右键托盘图标 → Exit

---

## ⚙️ 配置说明

编辑 `config.toml`（首次启动会从 `config.example.toml` 创建）：

```toml
[general]
hotkey = "ctrl+`"        # 热键（keyboard 库格式）
language = "auto"        # auto / zh / en
mode = "push-to-talk"    # push-to-talk / toggle

[engine]
backend = "auto"         # auto / faster-whisper / whisper-cpp
model = "medium"         # 模型名称

[output]
auto_paste = true        # 自动粘贴到光标
```

---

## 🧩 项目架构

```
speaknottype/
├── launcher.py              # 图形化启动器
├── main.py                  # 主程序入口（托盘 + 热键）
├── config.py                # 配置加载
├── recorder.py              # 录音模块
├── streaming_recorder.py    # 流式录音（实验性）
├── transcriber.py           # 识别引擎（多后端）
├── whisper_server_manager.py # whisper-server 进程管理
├── streaming_pipeline.py    # 流式识别管道
├── output.py                # 剪贴板 + 模拟粘贴
├── postprocessor.py         # 文本后处理
├── perf_timer.py            # 性能计时
└── whisper/
    ├── models/              # ggml-*.bin 模型（whisper.cpp 用）
    ├── models-ct2/          # CTranslate2 模型（faster-whisper 用）
    └── Release/             # whisper.cpp 二进制（自行下载）
```

---

## 🤝 贡献

欢迎 Pull Request 和 Issue！

- 报告 bug：[Issues](https://github.com/suchenggoole-maker/speaknottype/issues)
- 功能建议：[Discussions](https://github.com/suchenggoole-maker/speaknottype/discussions)
- 提交代码：Fork → 改 → Pull Request

---

## 📦 依赖

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 后端的 Whisper 实现
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) — C++ 实现的 Whisper
- [sounddevice](https://python-sounddevice.readthedocs.io/) — 跨平台录音
- [keyboard](https://github.com/boppreh/keyboard) — 全局热键
- [pystray](https://github.com/moses-palmer/pystray) — 系统托盘

---

## 📄 License

[MIT](LICENSE) © 2026 suchenggoole-maker

---

## 🙏 致谢

- OpenAI Whisper 团队提供的语音识别模型
- ggml-org / SYSTRAN 团队的开源实现
- 测试反馈来自所有早期使用者
