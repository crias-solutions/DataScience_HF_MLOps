from pathlib import Path
from typing import Any
import yaml


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
CONFIGS_DIR = PROJECT_ROOT / "configs"


class Config:
    def __init__(self, config_path: str | Path | None = None):
        self._config: dict[str, Any] = {}
        if config_path:
            self.load(config_path)

    def load(self, config_path: str | Path) -> None:
        with open(config_path) as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    @property
    def all(self) -> dict[str, Any]:
        return self._config.copy()


def load_config(config_name: str) -> Config:
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    config = Config()
    if config_path.exists():
        config.load(config_path)
    return config
