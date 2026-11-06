import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int


@dataclass(frozen=True)
class SessionConfig:
    frames_per_second: int
    frame_budget_ms: float
    max_hours_per_second: float
    max_displayed_people: int
    max_frame_ms: float
    max_unacknowledged_frames: int
    viewer_stall_seconds: float
    max_unattended_hours: float
    kept_idle_hours: float

    @property
    def frame_seconds(self) -> float:
        return 1 / self.frames_per_second

    @property
    def max_frame_seconds(self) -> float:
        return self.max_frame_ms / 1000

    @property
    def frame_budget_seconds(self) -> float:
        return self.frame_budget_ms / 1000


@dataclass(frozen=True)
class LimitsConfig:
    max_total_people: int


@dataclass(frozen=True)
class UiConfig:
    reconnect_delay_ms: int
    preview_delay_ms: int
    toast_ms: int


@dataclass(frozen=True)
class AppConfig:
    """Constants of the application itself; loaded from `configs/app.toml`."""

    server: ServerConfig
    session: SessionConfig
    limits: LimitsConfig
    ui: UiConfig


def load_app_config(path: Path) -> AppConfig:
    """Load and validate the application config; every number must be positive."""
    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    sections = {
        "server": ServerConfig,
        "session": SessionConfig,
        "limits": LimitsConfig,
        "ui": UiConfig,
    }
    config = AppConfig(**{name: cls(**raw[name]) for name, cls in sections.items()})
    for name in sections:
        for key, value in vars(getattr(config, name)).items():
            if isinstance(value, int | float) and not isinstance(value, bool) and value <= 0:
                raise ValueError(f"{path}: [{name}] {key} must be positive")
    return config
