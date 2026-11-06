"""A copy of the town config that a test can break on purpose."""

from pathlib import Path

from config_factory import CONFIG_DIRECTORY


def write_town(tmp_path, old=None, new=None, age_table=None) -> Path:
    """A copy of the town config in a temporary tree with the reference data it points to."""
    reference = tmp_path / "data" / "reference"
    reference.mkdir(parents=True)
    table = (
        CONFIG_DIRECTORY.parent / "data" / "reference" / "czechia_population_2019.csv"
    ).read_text()
    (reference / "czechia_population_2019.csv").write_text(age_table or table)
    text = (CONFIG_DIRECTORY / "town.toml").read_text()
    if old is not None:
        assert old in text
        text = text.replace(old, new, 1)
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "town.toml"
    path.write_text(text)
    return path
