from pathlib import Path

import streamlit as st

from fcscs.engines.raster_tools import resolve_output_dir


def get_project_root():
    return Path(__file__).resolve().parents[3]


def get_output_directory(config=None):
    if config is None:
        try:
            from fcscs.ui.app_state import get_config

            config = get_config()
        except Exception:
            config = None

    if config is not None and hasattr(config, "output_dir"):
        output_dir = resolve_output_dir(config.output_dir)
    else:
        output_dir = resolve_output_dir("../ForestCarbonScenarioSystem_outputs")

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def render_empty_message(message):
    st.info(message)
