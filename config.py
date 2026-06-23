"""配置加载模块"""

import os
try:
    import tomllib as tomli  # Python 3.11+
except ModuleNotFoundError:
    import tomli  # fallback

DEFAULT_CONFIG = {
    "general": {
        "hotkey": "ctrl+`",
        "language": "auto",
        "mode": "push-to-talk",
    },
    "engine": {
        "backend": "auto",
        "model": "tiny",
    },
    "output": {
        "auto_paste": True,
    },
    "recording": {
        "sample_rate": 16000,
        "format": "wav",
        "temp_dir": "",
    },
    "server": {
        "threads": 0,    # 0 = auto-detect
        "vad": False,    # VAD 可能在部分设备上导致 500 错误，默认关闭
    },
}


def get_project_root() -> str:
    """获取项目根目录（config.toml 所在目录）"""
    return os.path.dirname(os.path.abspath(__file__))


def get_whisper_dir() -> str:
    """获取 whisper 相关文件目录"""
    return os.path.join(get_project_root(), "whisper")


def get_models_dir() -> str:
    """获取模型文件目录"""
    return os.path.join(get_whisper_dir(), "models")


def get_whisper_cli_path() -> str:
    """获取 whisper-cli 可执行文件路径"""
    return os.path.join(get_whisper_dir(), "whisper-cli.exe")


def get_model_path(model_name: str) -> str:
    """获取指定模型文件的完整路径"""
    return os.path.join(get_models_dir(), f"ggml-{model_name}.bin")


def load_config() -> dict:
    """加载 config.toml，缺失字段使用默认值"""
    config_path = os.path.join(get_project_root(), "config.toml")

    config = {}
    config.update(DEFAULT_CONFIG)

    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            user_config = tomli.load(f)

        # 深层合并
        for section, values in user_config.items():
            if section in config and isinstance(config[section], dict):
                config[section].update(values)
            else:
                config[section] = values

    return config