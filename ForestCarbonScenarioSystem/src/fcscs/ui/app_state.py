import streamlit as st

from fcscs.config.defaults import ScenarioConfig


STATE_KEY = "fcscs_state"


def init_state():
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = {
            "config": ScenarioConfig(),
        }


def get_config():
    init_state()
    config = st.session_state[STATE_KEY]["config"]
    return config


def set_config(config):
    init_state()
    st.session_state[STATE_KEY]["config"] = config
