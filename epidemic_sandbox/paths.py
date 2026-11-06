"""Where the files that are not code live: the configs, the reference data and the web UI."""

from pathlib import Path

PACKAGE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = PACKAGE_DIRECTORY.parent
CONFIG_DIRECTORY = PROJECT_DIRECTORY / "configs"
REFERENCE_DIRECTORY = PROJECT_DIRECTORY / "data" / "reference"
WEB_DIRECTORY = PACKAGE_DIRECTORY / "web"
