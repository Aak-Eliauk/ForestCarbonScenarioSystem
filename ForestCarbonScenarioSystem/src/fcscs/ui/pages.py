import streamlit as st

from fcscs.ui.app_state import init_state
from fcscs.ui.page_workbench import render_run_log_page, render_workbench_page, render_workbench_step_sidebar
from fcscs.ui.styles import apply_base_style


SIDEBAR_PANEL_KEY = "sidebar_panel"
SOFTWARE_FULL_NAME = "森林损失情景控制与蒙特卡洛碳储量模拟系统 V1.0"
SOFTWARE_SHORT_NAME = "森林损失碳模拟"


def run_app():
    st.set_page_config(
        page_title=SOFTWARE_FULL_NAME,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": SOFTWARE_FULL_NAME + "（简称：" + SOFTWARE_SHORT_NAME + "）",
        },
    )
    init_state()
    apply_base_style()
    render_sidebar()


def render_sidebar():
    if SIDEBAR_PANEL_KEY not in st.session_state:
        st.session_state[SIDEBAR_PANEL_KEY] = "workflow"

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-lockup">
                <div class="brand-mark">FC</div>
                <div class="brand-copy">
                    <div class="brand-kicker">森林损失情景控制 · 蒙特卡洛模拟</div>
                    <h1>森林损失<br>碳模拟</h1>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_workbench_step_sidebar()

    st.sidebar.divider()
    st.sidebar.markdown('<div class="sidebar-section-label">运行记录</div>', unsafe_allow_html=True)
    log_button_type = "primary" if st.session_state.get(SIDEBAR_PANEL_KEY) == "logs" else "secondary"
    if st.sidebar.button("查看运行日志", type=log_button_type, use_container_width=True):
        st.session_state[SIDEBAR_PANEL_KEY] = "logs"
        st.rerun()

    if st.session_state.get(SIDEBAR_PANEL_KEY) == "logs":
        render_run_log_page()
    else:
        render_workbench_page()
