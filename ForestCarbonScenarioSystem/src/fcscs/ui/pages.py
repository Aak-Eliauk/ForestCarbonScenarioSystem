import streamlit as st

from fcscs.ui.app_state import init_state
from fcscs.ui.page_results import render_results_page
from fcscs.ui.page_workbench import (
    render_run_log_page,
    render_workbench_page,
    render_workbench_step_sidebar,
)
from fcscs.ui.styles import apply_base_style


SIDEBAR_PANEL_KEY = "sidebar_panel"
SOFTWARE_FULL_NAME = "森林损失情景控制与蒙特卡洛碳储量模拟系统"
SOFTWARE_SHORT_NAME = "城镇扩张限制与采伐管控碳模拟系统"


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
                <div class="brand-mark">Forest<br>Carbon</div>
                <div class="brand-copy">
                    <div class="brand-kicker">城镇扩张限制 · 采伐管控</div>
                    <h1>森林损失情景控制与蒙特卡洛碳储量模拟系统</h1>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    active_panel = st.session_state.get(SIDEBAR_PANEL_KEY, "workflow")
    render_workbench_step_sidebar(active_panel)

    st.sidebar.divider()
    st.sidebar.markdown('<div class="sidebar-section-label">结果中心</div>', unsafe_allow_html=True)
    if active_panel == "results":
        st.sidebar.markdown('<div class="sidebar-nav-item active">结果查看</div>', unsafe_allow_html=True)
    else:
        if st.sidebar.button("结果查看", type="secondary", use_container_width=True):
            st.session_state[SIDEBAR_PANEL_KEY] = "results"
            st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown('<div class="sidebar-section-label">运行记录</div>', unsafe_allow_html=True)
    if active_panel == "logs":
        st.sidebar.markdown('<div class="sidebar-nav-item active">查看运行日志</div>', unsafe_allow_html=True)
    else:
        if st.sidebar.button("查看运行日志", type="secondary", use_container_width=True):
            st.session_state[SIDEBAR_PANEL_KEY] = "logs"
            st.rerun()

    if st.session_state.get(SIDEBAR_PANEL_KEY) == "logs":
        render_run_log_page()
    elif st.session_state.get(SIDEBAR_PANEL_KEY) == "results":
        render_results_page()
    else:
        render_workbench_page()
