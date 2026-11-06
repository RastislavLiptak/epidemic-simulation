
import pytest

from epidemic_sandbox.config.app import load_app_config
from epidemic_sandbox.paths import CONFIG_DIRECTORY

DEMO = CONFIG_DIRECTORY / "app.toml"


def test_demo_app_config_loads():
    config = load_app_config(DEMO)
    assert config.server.port == 8001 and config.session.frame_seconds == pytest.approx(1 / 30)
    assert config.session.frame_budget_seconds == pytest.approx(0.025)
    assert config.limits.max_total_people >= 1 and config.session.max_displayed_people >= 1


def test_non_positive_values_are_rejected(tmp_path):
    broken = tmp_path / "app.toml"
    original = DEMO.read_text()
    broken.write_text(original.replace("max_total_people = 2000000", "max_total_people = 0"))
    with pytest.raises(ValueError, match="max_total_people must be positive"):
        load_app_config(broken)


def test_missing_section_is_rejected(tmp_path):
    broken = tmp_path / "app.toml"
    broken.write_text(DEMO.read_text().replace("[limits]", "[other]"))
    with pytest.raises(KeyError):
        load_app_config(broken)
