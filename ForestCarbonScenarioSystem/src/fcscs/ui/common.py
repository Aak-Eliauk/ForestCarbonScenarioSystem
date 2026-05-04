from pathlib import Path

import streamlit as st

from fcscs.engines.raster_tools import resolve_output_dir
from fcscs.config.defaults import sanitize_scenario_name


def get_project_root():
    return Path(__file__).resolve().parents[3]


def get_output_directory(config=None, create=True):
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

    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_batch_output_directory(config=None, create=True):
    output_dir = get_output_directory(config, create=create)
    batch_name = "运行批次"
    if config is not None and hasattr(config, "batch_name"):
        batch_name = config.batch_name
    batch_dir = output_dir / sanitize_scenario_name(batch_name, default="运行批次")
    if create:
        batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def render_empty_message(message):
    st.info(message)
