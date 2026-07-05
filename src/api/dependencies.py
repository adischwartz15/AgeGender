"""Application state and FastAPI dependency wiring.

Holds a single in-process ``AppState`` with the loaded config, artifacts,
and predictor, so ``/admin/reload-models`` can reload everything without
restarting the process.
"""

from __future__ import annotations

import logging

from src.inference.artifacts import load_all_artifacts
from src.inference.predictor import Predictor
from src.utils.config import CONFIG_DIR, load_config, resolve_device

logger = logging.getLogger(__name__)


class AppState:
    def __init__(self) -> None:
        self.config: dict = {}
        self.device: str = "cpu"
        self.predictor: Predictor | None = None

    def load(self) -> None:
        self.config = load_config(CONFIG_DIR / "api.yaml")
        api_config = self.config["api"]
        self.device = resolve_device(self.config.get("device", "auto"))

        artifacts = load_all_artifacts(api_config, self.device)
        self.predictor = Predictor(artifacts, api_config, self.device)
        if artifacts.warnings:
            for warning in artifacts.warnings:
                logger.warning(warning)


app_state = AppState()


def get_predictor() -> Predictor:
    if app_state.predictor is None:
        app_state.load()
    return app_state.predictor


def get_app_state() -> AppState:
    if app_state.predictor is None:
        app_state.load()
    return app_state
