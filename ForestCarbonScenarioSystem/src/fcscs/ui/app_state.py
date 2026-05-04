import streamlit as st

from fcscs.config.defaults import ScenarioConfig


STATE_KEY = "fcscs_state"


def init_state():
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = {
            "config": ScenarioConfig(),
            "events": [],
            "logging_patch_library": None,
            "simulation_bundle": None,
            "report_bundle": None,
            "simulation_history": [],
        }


def get_config():
    init_state()
    config = st.session_state[STATE_KEY]["config"]
    if not hasattr(config, "use_history_training"):
        config = ScenarioConfig(**config.to_dict())
        st.session_state[STATE_KEY]["config"] = config
    return config


def set_config(config):
    init_state()
    st.session_state[STATE_KEY]["config"] = config


def set_events(events):
    init_state()
    st.session_state[STATE_KEY]["events"] = events


def get_events():
    init_state()
    return st.session_state[STATE_KEY]["events"]


def set_logging_patch_library(patch_library):
    init_state()
    st.session_state[STATE_KEY]["logging_patch_library"] = patch_library


def get_logging_patch_library():
    init_state()
    return st.session_state[STATE_KEY]["logging_patch_library"]


def set_simulation_bundle(bundle):
    init_state()
    st.session_state[STATE_KEY]["simulation_bundle"] = bundle


def get_simulation_bundle():
    init_state()
    return st.session_state[STATE_KEY]["simulation_bundle"]


def set_report_bundle(bundle):
    init_state()
    st.session_state[STATE_KEY]["report_bundle"] = bundle


def get_report_bundle():
    init_state()
    return st.session_state[STATE_KEY]["report_bundle"]


def add_simulation_history(row):
    init_state()
    history = st.session_state[STATE_KEY]["simulation_history"]
    history.append(row)
    st.session_state[STATE_KEY]["simulation_history"] = history


def get_simulation_history():
    init_state()
    return st.session_state[STATE_KEY]["simulation_history"]


def clear_simulation_history():
    init_state()
    st.session_state[STATE_KEY]["simulation_history"] = []


def config_to_frame():
    init_state()
    return get_config().to_dict()
