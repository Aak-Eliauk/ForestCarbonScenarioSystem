import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from fcscs.config.defaults import ScenarioConfig, build_default_batch_name, build_preset_config, list_preset_names, sanitize_scenario_name
from fcscs.engines.raster_tools import parse_env_raster_paths, parse_year_raster_paths, path_exists, resolve_input_path
from fcscs.services.background_run_service import find_recent_jobs, read_status, start_background_run, terminate_background_run
from fcscs.ui.app_state import (
    get_config,
    set_config,
)
from fcscs.ui.common import get_batch_output_directory, get_output_directory
from fcscs.ui.styles import render_page_banner


WIZARD_STEP_KEY = "workbench_wizard_step"
RUN_PROGRESS_KEY = "workbench_run_progress"
RUN_STAGE_KEY = "workbench_run_stage"
RUN_MESSAGE_KEY = "workbench_run_message"
RUN_LOG_KEY = "workbench_run_log"
RUN_LOG_PATH_KEY = "workbench_run_log_path"
BACKGROUND_STATUS_PATH_KEY = "workbench_background_status_path"
RUN_STARTING_KEY = "workbench_background_starting"
LAST_RENDER_STEP_KEY = "workbench_last_render_step"
RUN_STAGES = ["等待开始", "检查输入", "准备快速测试", "生成事件", "经验强度采样", "训练模型", "蒙特卡洛模拟", "汇总结果", "完成"]
WIZARD_STEPS = [
    ("data", "数据准备"),
    ("scenario", "情景方案"),
    ("run", "启动运行"),
    ("status", "运行状态"),
]

STEP_DESCRIPTIONS = {
    "data": "选择基础栅格、扰动约束、环境因子和历史训练数据。",
    "scenario": "设置森林损失控制情景、蒙特卡洛次数和高级模型参数。",
    "run": "检查数据与情景配置，设置批次名并启动后台计算。",
    "status": "查看后台计算进度、切换运行批次，并在需要时终止任务。",
}


def render_workbench_page():
    _ensure_wizard_state()

    config = get_config()
    step_index = int(st.session_state[WIZARD_STEP_KEY])
    _refresh_run_batch_name_when_opening_step(config, step_index)
    step_key, step_label = WIZARD_STEPS[step_index]
    render_page_banner(str(step_index + 1) + " " + step_label, STEP_DESCRIPTIONS.get(step_key, ""))
    if step_index == 0:
        _render_data_step(config)
    elif step_index == 1:
        _render_scenario_step(config)
    elif step_index == 2:
        _render_run_step(config)
    else:
        _render_run_status_content(config)


def _ensure_wizard_state():
    if WIZARD_STEP_KEY not in st.session_state:
        st.session_state[WIZARD_STEP_KEY] = 0

    current = int(st.session_state[WIZARD_STEP_KEY])
    if current < 0 or current >= len(WIZARD_STEPS):
        current = 0
    st.session_state[WIZARD_STEP_KEY] = current


def _refresh_run_batch_name_when_opening_step(config, step_index):
    previous_step = st.session_state.get(LAST_RENDER_STEP_KEY)
    if step_index == 2 and previous_step != 2:
        st.session_state["wizard_batch_name"] = build_default_batch_name(config.scenario_name)
    st.session_state[LAST_RENDER_STEP_KEY] = step_index


def render_workbench_step_sidebar(active_panel="workflow"):
    _ensure_wizard_state()
    config = get_config()
    labels = _build_step_labels(config)
    current_step = int(st.session_state[WIZARD_STEP_KEY])
    st.sidebar.markdown('<div class="sidebar-section-label">流程步骤</div>', unsafe_allow_html=True)
    for index, label in enumerate(labels):
        is_current_step = active_panel == "workflow" and index == current_step
        if is_current_step:
            st.sidebar.markdown('<div class="sidebar-nav-item active">' + label + "</div>", unsafe_allow_html=True)
            continue

        if st.sidebar.button(label, key="sidebar_step_jump_" + str(index), type="secondary", use_container_width=True):
            st.session_state[WIZARD_STEP_KEY] = index
            st.session_state["sidebar_panel"] = "workflow"
            st.rerun()


def render_run_log_page():
    render_page_banner("运行日志", "集中查看本次运行记录、异常提示和历史日志文件。")

    config = get_config()
    log_rows = st.session_state.get(RUN_LOG_KEY, [])
    log_path = st.session_state.get(RUN_LOG_PATH_KEY)
    running_jobs = _running_background_jobs(config)
    recent_log_files = _get_recent_log_files()

    c1, c2, c3 = st.columns(3)
    c1.metric("本次记录", len(log_rows) + len(running_jobs))
    c2.metric("日志文件", len(recent_log_files))
    if running_jobs:
        current_state = "运行中 " + str(len(running_jobs)) + " 个"
    else:
        current_state = st.session_state.get(RUN_STAGE_KEY, "等待开始")
    c3.metric("当前状态", current_state)

    c_back, c_path = st.columns([1, 2])
    with c_back:
        if st.button("返回当前流程步骤", type="primary", use_container_width=True, key="log_page_back_to_workflow"):
            st.session_state["sidebar_panel"] = "workflow"
            st.rerun()
    with c_path:
        if log_path:
            st.caption("当前日志文件：" + str(log_path))
        elif running_jobs:
            st.caption("正在读取后台运行批次的日志文件。")
        else:
            st.caption("当前还没有新的运行日志。开始运行后会自动记录。")

    _render_current_run_records(log_rows, running_jobs)
    _render_recent_log_files(recent_log_files)


def _render_current_run_records(log_rows, running_jobs):
    st.subheader("本次运行记录")
    has_content = False

    if running_jobs:
        st.markdown("**正在运行的后台任务**")
        rows = []
        labels = []
        for index, job in enumerate(running_jobs):
            batch_name = str(job.get("batch_name", ""))
            state_label = _format_job_state(job.get("state", ""))
            labels.append(batch_name + " / " + state_label)
            rows.append(
                {
                    "批次": batch_name,
                    "状态": state_label,
                    "进度": str(job.get("percent", 0)) + "%",
                    "阶段": job.get("stage", ""),
                    "更新时间": job.get("updated_at", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        selected_label = st.selectbox("查看正在运行的批次日志", labels, key="running_log_file_select")
        selected_index = labels.index(selected_label)
        selected_job = running_jobs[selected_index]
        file_rows = _read_run_event_rows(selected_job.get("run_log_path", ""), limit=20)
        if file_rows:
            st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)
        else:
            st.info("该批次已经启动，日志文件正在生成。")
        has_content = True

    if log_rows:
        st.markdown("**本次会话记录**")
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
        has_content = True

    if not has_content:
        st.info("本次会话暂无运行日志。")


def _render_run_status_content(config):
    c_back, c_result = st.columns([1, 1])
    with c_back:
        if st.button("返回启动运行", use_container_width=True, key="status_back_to_run_step"):
            st.session_state["sidebar_panel"] = "workflow"
            st.session_state[WIZARD_STEP_KEY] = 2
            st.rerun()
    with c_result:
        if st.button("查看结果", type="primary", use_container_width=True, key="status_go_to_results"):
            st.session_state["sidebar_panel"] = "results"
            st.rerun()

    _render_run_status_live_area()


@st.fragment(run_every="5s")
def _render_run_status_live_area():
    _render_run_status_live_content()


def _render_run_status_live_content():
    config = get_config()
    active_status = _get_active_background_status(config)
    _render_progress_area(active_status)
    _render_background_job_panel(config, active_status)

    if not active_status:
        st.info("当前没有选中的后台批次。启动运行后，系统会自动跳转到这里显示进度。")

    if _background_finished(active_status):
        _render_run_complete_actions()


def _build_step_labels(config):
    data_ok = _data_ready(config)
    scenario_ok = _scenario_ready(config)
    current_step = int(st.session_state[WIZARD_STEP_KEY])
    active_status = _get_active_background_status(config)
    run_ok = bool(active_status)
    status_ok = _background_finished(active_status)
    completed = [data_ok, scenario_ok, run_ok, status_ok]

    labels = []
    for index, (_, label) in enumerate(WIZARD_STEPS):
        if index == current_step:
            status = "当前"
        elif completed[index]:
            status = "完成"
        else:
            status = "待处理"
        labels.append(f"{index + 1:02d}  {label}  /  {status}")
    return labels


def _render_data_step(current):
    output_dir = _folder_path_input("输出文件夹", current.output_dir, "wizard_output_dir")
    project_rasters = _find_project_rasters()

    _render_section_title("基础栅格数据", "用于确定基准年森林碳储量和目标年土地利用变化。")
    agbd_raster_path = _raster_path_input("森林地上生物量密度（AGBD）基准栅格", current.agbd_raster_path, project_rasters, "wizard_agbd")
    tcc_raster_path = _raster_path_input("树冠覆盖度基准栅格", current.tcc_raster_path, project_rasters, "wizard_tcc")
    lulc_base_raster_path = _raster_path_input("基准年土地利用栅格", current.lulc_base_raster_path, project_rasters, "wizard_lulc_base")
    lulc_target_raster_path = _raster_path_input("目标年预测土地利用栅格", current.lulc_target_raster_path, project_rasters, "wizard_lulc_target")

    _render_section_title("扰动与约束数据", "森林损失驱动因素用于识别采伐发生位置，自然保护区用于约束采伐区域。")
    drivers_raster_path = _raster_path_input("森林损失驱动因素栅格", current.drivers_raster_path, project_rasters, "wizard_drivers")
    reserve_raster_path = _raster_path_input("自然保护区栅格", current.reserve_raster_path, project_rasters, "wizard_reserve")

    _render_section_title("环境因子栅格", "用于机器学习训练和预测，可表达地形、气候水分与人为活动影响。")
    terrain_path = _get_env_path_by_role(current.env_raster_paths, "terrain")
    climate_path = _get_env_path_by_role(current.env_raster_paths, "climate")
    human_path = _get_env_path_by_role(current.env_raster_paths, "human")
    extra_env_items = _get_extra_env_items(current.env_raster_paths)

    terrain_path = _raster_path_input("地形因子栅格", terrain_path, project_rasters, "wizard_env_terrain")
    st.caption("地形因子可使用坡度、高程、地形起伏度等数据，默认推荐坡度。")
    climate_path = _raster_path_input("气候水分因子栅格", climate_path, project_rasters, "wizard_env_climate")
    st.caption("气候水分因子可使用年降水量、平均降水量、实际蒸散量、湿度等数据。")
    human_path = _raster_path_input("人为活动因子栅格", human_path, project_rasters, "wizard_env_human")
    st.caption("人为活动因子可使用道路距离、居民点距离、城镇距离、人口密度、夜光等数据。")

    with st.expander("其他环境因子"):
        extra_env_items = _render_extra_env_factor_form(extra_env_items, project_rasters)

    _render_section_title("历史训练数据", "连续年份树冠覆盖度用于量化森林损失强度，并用森林损失驱动因素数据定位采伐发生位置。")
    with st.expander("历史训练数据输入", expanded=True):
        st.caption("至少需要两组相同年份的历史森林地上生物量密度、树冠覆盖度和土地利用栅格，系统会按相邻年份构造训练样本。")
        history_row_count = _history_row_count_input(current)
        history_agbd_paths = _render_year_raster_form(
            "历史森林地上生物量密度（AGBD）",
            current.history_agbd_paths,
            project_rasters,
            "wizard_hist_agbd",
            history_row_count,
        )
        history_tcc_paths = _render_year_raster_form("历史树冠覆盖度", current.history_tcc_paths, project_rasters, "wizard_hist_tcc", history_row_count)
        history_lulc_paths = _render_year_raster_form("历史土地利用", current.history_lulc_paths, project_rasters, "wizard_hist_lulc", history_row_count)

    data_missing = _find_data_missing(
        agbd_raster_path,
        tcc_raster_path,
        lulc_base_raster_path,
        lulc_target_raster_path,
        drivers_raster_path,
        reserve_raster_path,
        terrain_path,
        climate_path,
        human_path,
        history_agbd_paths,
        history_tcc_paths,
        history_lulc_paths,
    )
    _render_required_path_status(data_missing)

    with st.expander("高级数据设置"):
        c7, c8, c9, c10 = st.columns(4)
        with c7:
            forest_lulc_codes = st.text_input("森林类型编码", value=current.forest_lulc_codes, key="wizard_forest_codes")
        with c8:
            urban_lulc_codes = st.text_input("城镇类型编码", value=current.urban_lulc_codes, key="wizard_urban_codes")
        with c9:
            logging_driver_value = st.number_input("采伐扰动编码", value=current.logging_driver_value, step=1, key="wizard_logging_value")
        with c10:
            reserve_value = st.number_input("自然保护区编码", value=current.reserve_value, step=1, key="wizard_reserve_value")
        write_raster_outputs = st.checkbox("输出 GeoTIFF 栅格结果", value=current.write_raster_outputs, key="wizard_write_tif")

    c_back, c_save = st.columns([1, 2])
    with c_back:
        if st.button("重置为当前配置", use_container_width=True, key="wizard_data_reset"):
            _clear_data_widget_state()
            st.rerun()
    with c_save:
        if st.button("保存数据并继续", type="primary", use_container_width=True, key="wizard_save_data"):
            env_raster_paths = _build_env_raster_text(terrain_path, climate_path, human_path, extra_env_items)
            missing = _find_data_missing(
                agbd_raster_path,
                tcc_raster_path,
                lulc_base_raster_path,
                lulc_target_raster_path,
                drivers_raster_path,
                reserve_raster_path,
                terrain_path,
                climate_path,
                human_path,
                history_agbd_paths,
                history_tcc_paths,
                history_lulc_paths,
            )
            if missing:
                st.error("必填数据还不完整：" + "、".join(missing))
                return

            new_config = current.copy()
            new_config.output_dir = output_dir
            new_config.agbd_raster_path = agbd_raster_path
            new_config.tcc_raster_path = tcc_raster_path
            new_config.lulc_base_raster_path = lulc_base_raster_path
            new_config.lulc_target_raster_path = lulc_target_raster_path
            new_config.drivers_raster_path = drivers_raster_path
            new_config.reserve_raster_path = reserve_raster_path
            new_config.env_raster_paths = env_raster_paths
            new_config.history_agbd_paths = history_agbd_paths
            new_config.history_tcc_paths = history_tcc_paths
            new_config.history_lulc_paths = history_lulc_paths
            new_config.forest_lulc_codes = forest_lulc_codes
            new_config.urban_lulc_codes = urban_lulc_codes
            new_config.logging_driver_value = int(logging_driver_value)
            new_config.reserve_value = int(reserve_value)
            new_config.write_raster_outputs = bool(write_raster_outputs)

            _save_and_clear(new_config)
            _go_to_step(1)


def _render_scenario_step(current):
    preset_names = list_preset_names()
    if st.session_state.get("wizard_preset") not in (None, *preset_names):
        del st.session_state["wizard_preset"]
    preset_index = _find_preset_index(current.scenario_name, preset_names)
    preset_name = st.selectbox(
        "预设方案",
        preset_names,
        index=preset_index,
        key="wizard_preset",
        on_change=_apply_selected_preset,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        scenario_name = st.text_input("情景名称", value=current.scenario_name, key="wizard_scenario_name")
    with c2:
        base_year = st.number_input("基准年", value=current.base_year, step=1, key="wizard_base_year")
    with c3:
        target_year = st.number_input("目标年", value=current.target_year, step=1, key="wizard_target_year")

    c4, c5, c6 = st.columns(3)
    with c4:
        logging_area_reduction = st.slider("采伐面积减少", 0.0, 0.9, float(current.logging_area_reduction), 0.05, key="wizard_log_area")
    with c5:
        urban_area_reduction = st.slider("城镇扩张减少", 0.0, 0.9, float(current.urban_area_reduction), 0.05, key="wizard_urban_area")
    with c6:
        mc_n_simulations = st.number_input("蒙特卡洛次数", min_value=1, max_value=5000, value=current.mc_n_simulations, step=10, key="wizard_mc")

    logging_severity_reduction = current.logging_severity_reduction
    urban_severity_reduction = current.urban_severity_reduction
    urban_speed_shift = current.urban_speed_shift
    logging_patch_min_size = current.logging_patch_min_size
    logging_patch_max_size = current.logging_patch_max_size
    logging_library_patch_count = current.logging_library_patch_count
    logging_library_years = current.logging_library_years
    ml_sample_count = current.ml_sample_count
    ml_n_estimators = current.ml_n_estimators
    ml_max_depth = current.ml_max_depth
    agbd_to_agc_factor = current.agbd_to_agc_factor
    severity_method = current.severity_method
    base_seed = current.base_seed
    use_driver_sample_weight = current.use_driver_sample_weight
    logging_probability_band = current.logging_probability_band
    urban_probability_band = current.urban_probability_band
    driver_probability_scale = current.driver_probability_scale
    severity_sample_count = current.severity_sample_count
    logging_cap = current.logging_severity_cap_quantile
    if logging_cap is None:
        logging_cap = 1.0
    pixel_area_ha = current.pixel_area_ha

    with st.expander("高级模型参数"):
        a1, a2, a3 = st.columns(3)
        with a1:
            logging_severity_reduction = st.slider("采伐强度降低", 0.0, 0.9, float(current.logging_severity_reduction), 0.05, key="wizard_log_sev")
            logging_patch_min_size = st.number_input("最小斑块像元数", min_value=1, max_value=1000, value=current.logging_patch_min_size, step=1, key="wizard_patch_min")
            ml_sample_count = st.number_input("训练样本上限", min_value=100, max_value=200000, value=current.ml_sample_count, step=100, key="wizard_sample")
        with a2:
            urban_severity_reduction = st.slider("城镇扰动强度降低", 0.0, 0.9, float(current.urban_severity_reduction), 0.05, key="wizard_urban_sev")
            logging_patch_max_size = st.number_input("最大斑块像元数", min_value=1, max_value=5000, value=current.logging_patch_max_size, step=1, key="wizard_patch_max")
            ml_n_estimators = st.number_input("随机森林树数量", min_value=10, max_value=500, value=current.ml_n_estimators, step=10, key="wizard_tree")
        with a3:
            urban_speed_shift = st.slider("扩张时间偏移", -1.0, 1.0, float(current.urban_speed_shift), 0.05, key="wizard_urban_speed")
            logging_library_patch_count = st.number_input("斑块模板数量", min_value=10, max_value=5000, value=current.logging_library_patch_count, step=10, key="wizard_patch_count")
            ml_max_depth = st.number_input("随机森林最大深度", min_value=3, max_value=40, value=current.ml_max_depth, step=1, key="wizard_depth")

        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            logging_library_years = st.number_input("历史年份数量", min_value=1, max_value=30, value=current.logging_library_years, step=1, key="wizard_lib_years")
        with b2:
            agbd_to_agc_factor = st.number_input("AGBD 转 AGC 系数", min_value=0.10, max_value=1.00, value=float(current.agbd_to_agc_factor), step=0.01, format="%.2f", key="wizard_agc")
        with b3:
            method_labels = ["S1 经验分层抽样", "S2 环境校正抽样"]
            method_index = 0 if current.severity_method == "S1" else 1
            method_label = st.selectbox("扰动强度方法", method_labels, index=method_index, key="wizard_sev_method")
            severity_method = method_label.split(" ")[0]
        with b4:
            base_seed = st.number_input("随机种子", value=current.base_seed, step=1, key="wizard_seed")
        with b5:
            pixel_area_ha = st.number_input("像元面积 ha", min_value=0.0001, max_value=100.0, value=float(current.pixel_area_ha), step=0.10, format="%.4f", key="wizard_pixel_area")

        c_weight, c_log_band, c_urban_band, c_scale = st.columns(4)
        with c_weight:
            use_driver_sample_weight = st.checkbox("使用 Drivers 概率作为训练权重", value=current.use_driver_sample_weight, key="wizard_driver_weight")
        with c_log_band:
            logging_probability_band = st.number_input("采伐概率波段", min_value=1, max_value=20, value=current.logging_probability_band, step=1, key="wizard_log_prob_band")
        with c_urban_band:
            urban_probability_band = st.number_input("城镇概率波段", min_value=1, max_value=20, value=current.urban_probability_band, step=1, key="wizard_urban_prob_band")
        with c_scale:
            driver_probability_scale = st.number_input("概率缩放值", min_value=1.0, max_value=10000.0, value=float(current.driver_probability_scale), step=10.0, key="wizard_driver_scale")

        severity_sample_count = st.number_input(
            "经验强度样本数",
            min_value=100,
            max_value=200000,
            value=current.severity_sample_count,
            step=100,
            key="wizard_sev_sample_count",
        )

    c_back, c_save = st.columns([1, 2])
    with c_back:
        if st.button("返回数据准备", use_container_width=True, key="wizard_back_to_data"):
            _go_to_step(0)
    with c_save:
        if st.button("保存情景并继续", type="primary", use_container_width=True, key="wizard_save_scenario"):
            if int(target_year) <= int(base_year):
                st.error("目标年必须大于基准年。")
                return
            if int(logging_patch_max_size) < int(logging_patch_min_size):
                st.error("最大斑块像元数不能小于最小斑块像元数。")
                return

            new_config = current.copy()
            new_config.scenario_name = sanitize_scenario_name(scenario_name)
            new_config.base_year = int(base_year)
            new_config.target_year = int(target_year)
            new_config.future_years = ScenarioConfig.build_future_years(base_year, target_year)
            new_config.logging_area_reduction = float(logging_area_reduction)
            new_config.logging_severity_reduction = float(logging_severity_reduction)
            new_config.logging_severity_cap_quantile = float(logging_cap)
            new_config.urban_area_reduction = float(urban_area_reduction)
            new_config.urban_speed_shift = float(urban_speed_shift)
            new_config.urban_severity_reduction = float(urban_severity_reduction)
            new_config.logging_patch_min_size = int(logging_patch_min_size)
            new_config.logging_patch_max_size = int(logging_patch_max_size)
            new_config.logging_library_years = int(logging_library_years)
            new_config.logging_library_patch_count = int(logging_library_patch_count)
            new_config.mc_n_simulations = int(mc_n_simulations)
            new_config.ml_sample_count = int(ml_sample_count)
            new_config.ml_n_estimators = int(ml_n_estimators)
            new_config.ml_max_depth = int(ml_max_depth)
            new_config.agbd_to_agc_factor = float(agbd_to_agc_factor)
            new_config.pixel_area_ha = float(pixel_area_ha)
            new_config.severity_method = severity_method
            new_config.base_seed = int(base_seed)
            new_config.use_driver_sample_weight = bool(use_driver_sample_weight)
            new_config.logging_probability_band = int(logging_probability_band)
            new_config.urban_probability_band = int(urban_probability_band)
            new_config.driver_probability_scale = float(driver_probability_scale)
            new_config.severity_sample_count = int(severity_sample_count)

            _save_and_clear(new_config)
            _go_to_step(2)


def _find_preset_index(scenario_name, preset_names):
    name = str(scenario_name).strip()
    if name == "BAU":
        name = "基准情景"
    for index, preset_name in enumerate(preset_names):
        if preset_name == name:
            return index
    return 0


def _apply_selected_preset():
    preset_name = st.session_state.get("wizard_preset", "基准情景")
    current = get_config()
    preset_config = build_preset_config(preset_name)
    preset_config.agbd_raster_path = current.agbd_raster_path
    preset_config.tcc_raster_path = current.tcc_raster_path
    preset_config.lulc_base_raster_path = current.lulc_base_raster_path
    preset_config.lulc_target_raster_path = current.lulc_target_raster_path
    preset_config.drivers_raster_path = current.drivers_raster_path
    preset_config.reserve_raster_path = current.reserve_raster_path
    preset_config.env_raster_paths = current.env_raster_paths
    preset_config.history_agbd_paths = current.history_agbd_paths
    preset_config.history_tcc_paths = current.history_tcc_paths
    preset_config.history_lulc_paths = current.history_lulc_paths
    preset_config.forest_lulc_codes = current.forest_lulc_codes
    preset_config.urban_lulc_codes = current.urban_lulc_codes
    preset_config.logging_driver_value = current.logging_driver_value
    preset_config.reserve_value = current.reserve_value
    preset_config.write_raster_outputs = current.write_raster_outputs
    preset_config.output_dir = current.output_dir
    preset_config.batch_name = current.batch_name
    preset_config.grid_rows = current.grid_rows
    preset_config.grid_cols = current.grid_cols
    _save_and_clear(preset_config)


def _render_run_step(config):
    starting_job = bool(st.session_state.get(RUN_STARTING_KEY, False))
    running_jobs = _running_background_jobs(config)
    running_count = len(running_jobs)

    if "wizard_batch_name" not in st.session_state:
        st.session_state["wizard_batch_name"] = build_default_batch_name(config.scenario_name)
    batch_name = st.text_input("运行批次名", key="wizard_batch_name")
    batch_preview_config = config.copy()
    batch_preview_config.batch_name = sanitize_scenario_name(batch_name, default=build_default_batch_name(config.scenario_name))
    batch_dir = get_batch_output_directory(batch_preview_config, create=False)
    st.caption("本次结果将保存到：" + str(batch_dir))
    if batch_dir.exists():
        st.warning("该批次目录已经存在。开始运行时系统会自动追加时间后缀，避免覆盖已有结果。")

    checks = _build_preflight_rows(config)
    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)
    can_run = all(row["状态"] == "通过" for row in checks)

    if not can_run:
        st.warning("请先处理未通过的检查项。")
    if starting_job:
        st.info("后台进程正在启动，请稍候。进程完全启动前不能重复点击开始运行。")

    run_mode = st.radio(
        "运行方式",
        ["快速测试", "完整运行"],
        horizontal=True,
        key="wizard_run_mode",
    )
    quick_size = 520
    if run_mode == "快速测试":
        quick_size = st.number_input("测试窗口大小", min_value=128, max_value=1200, value=520, step=64, key="wizard_quick_size")
    if run_mode == "快速测试":
        effective_mc = min(int(config.mc_n_simulations), 3)
        st.caption(f"快速测试会临时使用较小计算量：本次实际蒙特卡洛次数为 {effective_mc} 次；完整运行将使用情景设置的 {int(config.mc_n_simulations)} 次。")
    else:
        st.caption(f"完整运行将使用情景设置的蒙特卡洛次数：{int(config.mc_n_simulations)} 次。")

    max_running_jobs = _suggest_background_job_limit(config, run_mode)
    too_many_jobs = running_count >= max_running_jobs
    if running_count >= 2 and not too_many_jobs:
        st.warning(
            "当前已有 "
            + str(running_count)
            + " 个后台批次正在运行，继续多开可能造成系统卡顿。建议确认机器空闲后再启动。"
        )
    if too_many_jobs:
        st.error(
            "当前已有 "
            + str(running_count)
            + " 个后台批次正在运行，已达到建议上限 "
            + str(max_running_jobs)
            + " 个。请先等待任务完成，或到运行状态页终止部分批次。"
        )

    c_back, c_run = st.columns([1, 2])
    with c_back:
        if st.button("返回情景方案", use_container_width=True, key="wizard_back_to_scenario"):
            _go_to_step(1)
    with c_run:
        run_clicked = st.button(
            "正在启动..." if starting_job else "开始运行",
            type="primary",
            use_container_width=True,
            disabled=(not can_run) or starting_job or too_many_jobs,
            key="wizard_start_run",
        )

    if run_clicked:
        st.session_state[RUN_STARTING_KEY] = True
        _clear_results_only()
        run_config = _prepare_run_config(config, batch_name)
        try:
            with st.spinner("正在启动后台计算进程，请稍候..."):
                latest_running_count = len(_running_background_jobs(config))
                latest_limit = _suggest_background_job_limit(config, run_mode)
                if latest_running_count >= latest_limit:
                    raise RuntimeError("后台运行批次数已达到建议上限，请稍后再启动新任务。")
                job = start_background_run(run_config, run_mode, int(quick_size))
            st.session_state[BACKGROUND_STATUS_PATH_KEY] = str(job["status_path"])
            st.session_state[RUN_LOG_PATH_KEY] = str(job.get("run_log_path", ""))
            st.session_state[RUN_STARTING_KEY] = False
            st.session_state["sidebar_panel"] = "workflow"
            st.session_state[WIZARD_STEP_KEY] = 3
            st.success("后台任务已启动。计算会在独立 Python 进程中继续运行，刷新页面不会中断。")
            st.rerun()
        except Exception as error:
            st.session_state[RUN_STARTING_KEY] = False
            st.error("后台任务启动失败：" + str(error))


def _render_run_complete_actions():
    st.success("运行已完成。结果已经自动保存，可以进入结果查看。")
    c_keep, c_result = st.columns([1, 2])
    with c_keep:
        st.caption("需要重新运行时，可以调整上方运行方式后再次点击开始运行。")
    with c_result:
        if st.button("查看结果", type="primary", use_container_width=True, key="wizard_go_to_results_after_run"):
            st.session_state["sidebar_panel"] = "results"
            st.rerun()


def _prepare_run_config(config, batch_name):
    run_config = config.copy()
    safe_batch_name = sanitize_scenario_name(batch_name, default=build_default_batch_name(config.scenario_name))
    run_config.batch_name = safe_batch_name

    batch_dir = get_batch_output_directory(run_config, create=False)
    if batch_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_config.batch_name = sanitize_scenario_name(safe_batch_name + "_" + timestamp)

    return run_config


def _render_progress_area(active_status=None):
    progress_value = int(st.session_state.get(RUN_PROGRESS_KEY, 0))
    stage = str(st.session_state.get(RUN_STAGE_KEY, "等待开始"))
    message = str(st.session_state.get(RUN_MESSAGE_KEY, "点击开始运行后，这里会显示实时进度。"))

    if active_status:
        progress_value = int(active_status.get("percent", progress_value))
        stage = str(active_status.get("stage", stage))
        message = str(active_status.get("message", message))

    st.markdown("**运行进度**")
    progress_bar = st.progress(progress_value, text=f"{progress_value}% {stage}")
    status_table = st.empty()
    status_table.dataframe(_build_run_stage_frame(stage), use_container_width=True, hide_index=True)
    message_box = st.empty()
    message_box.info(message)
    log_table = st.empty()
    _render_log_table(log_table, active_status)
    return progress_bar, status_table, message_box, log_table


def _get_active_background_status(config):
    status_path = st.session_state.get(BACKGROUND_STATUS_PATH_KEY)
    if status_path:
        status = read_status(status_path)
        if status and _should_show_status_job(status):
            status["status_path"] = str(status_path)
            return status
        if BACKGROUND_STATUS_PATH_KEY in st.session_state:
            del st.session_state[BACKGROUND_STATUS_PATH_KEY]

    output_dir = get_output_directory(config, create=False)
    recent_jobs = _visible_recent_jobs(output_dir, limit=1)
    if recent_jobs:
        st.session_state[BACKGROUND_STATUS_PATH_KEY] = recent_jobs[0].get("status_path", "")
        return recent_jobs[0]
    return {}


def _render_background_job_panel(config, active_status):
    output_dir = get_output_directory(config, create=False)
    recent_jobs = _visible_recent_jobs(output_dir, limit=8)
    if not active_status and not recent_jobs:
        return

    st.markdown("**后台任务**")
    if active_status:
        state_label = _format_job_state(active_status.get("state", ""))
        st.caption(
            "当前批次："
            + str(active_status.get("batch_name", ""))
            + "；状态："
            + state_label
            + "；更新时间："
            + str(active_status.get("updated_at", ""))
        )
        if active_status.get("state") == "failed" and active_status.get("error"):
            with st.expander("后台错误详情"):
                st.code(str(active_status.get("error")), language="text")
        if _background_running(active_status):
            if st.button("终止当前运行", type="secondary", use_container_width=True, key="wizard_stop_active_background_job"):
                ok, message = terminate_background_run(active_status.get("status_path", ""))
                if ok:
                    st.warning(message)
                else:
                    st.error(message)
                st.rerun()

    if recent_jobs:
        rows = []
        for job in recent_jobs:
            rows.append(
                {
                    "批次": job.get("batch_name", ""),
                    "状态": _format_job_state(job.get("state", "")),
                    "进度": str(job.get("percent", 0)) + "%",
                    "阶段": job.get("stage", ""),
                    "更新时间": job.get("updated_at", ""),
                    "目录": job.get("batch_dir", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        selected_batch = st.selectbox("切换查看后台批次", [str(row["批次"]) for row in rows], key="wizard_background_job_select")
        for job in recent_jobs:
            if str(job.get("batch_name", "")) == selected_batch:
                if st.button("查看该批次进度", use_container_width=True, key="wizard_use_selected_background_job"):
                    st.session_state[BACKGROUND_STATUS_PATH_KEY] = job.get("status_path", "")
                    st.rerun()
                break


def _visible_recent_jobs(output_dir, limit=8):
    all_jobs = find_recent_jobs(output_dir, limit=50)
    visible_jobs = []
    for job in all_jobs:
        if _should_show_status_job(job):
            visible_jobs.append(job)
    return visible_jobs[:limit]


def _running_background_jobs(config):
    output_dir = get_output_directory(config, create=False)
    all_jobs = find_recent_jobs(output_dir, limit=50)
    running_jobs = []
    for job in all_jobs:
        state = str(job.get("state", ""))
        if state in {"starting", "running"}:
            running_jobs.append(job)
    return running_jobs


def _suggest_background_job_limit(config, run_mode):
    limit = 4
    if run_mode == "完整运行":
        limit = 3
    if int(config.mc_n_simulations) >= 300:
        limit = min(limit, 3)

    available_memory = _available_memory_gb()
    if available_memory is None:
        return limit
    if available_memory < 4:
        return 2
    if available_memory < 8:
        return min(limit, 3)
    return limit


def _available_memory_gb():
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        if not ok:
            return None
        return float(status.ullAvailPhys) / 1024 / 1024 / 1024
    except Exception:
        return None


def _should_show_status_job(job):
    state = str(job.get("state", ""))
    if state in {"starting", "running"}:
        return True

    updated_at = _parse_status_time(job.get("updated_at", ""))
    if updated_at is None:
        return False

    seconds = (datetime.now() - updated_at).total_seconds()
    return seconds <= 600


def _parse_status_time(text):
    try:
        return datetime.strptime(str(text), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _format_job_state(state):
    state = str(state)
    if state == "running":
        return "运行中"
    if state == "starting":
        return "启动中"
    if state == "finished":
        return "完成"
    if state == "failed":
        return "失败"
    if state == "stopped":
        return "已终止"
    return state or "未知"


def _background_finished(active_status):
    return bool(active_status) and active_status.get("state") == "finished"


def _background_running(active_status):
    if not active_status:
        return False
    return active_status.get("state") in {"starting", "running"}


def _build_run_stage_frame(current_stage):
    rows = []
    if current_stage in RUN_STAGES:
        current_index = RUN_STAGES.index(current_stage)
    else:
        current_index = 0

    for index, stage in enumerate(RUN_STAGES):
        if stage == "等待开始" and current_stage != "等待开始":
            status = "完成"
        elif index < current_index:
            status = "完成"
        elif index == current_index:
            status = "进行中" if current_stage != "完成" else "完成"
        else:
            status = "等待"
        rows.append({"步骤": stage, "状态": status})
    return pd.DataFrame(rows)


def _render_log_table(log_table, active_status=None):
    if active_status:
        rows = _read_run_event_rows(active_status.get("run_log_path", ""))
        if not rows:
            rows = [
                {
                    "时间": active_status.get("updated_at", ""),
                    "阶段": active_status.get("stage", ""),
                    "消息": active_status.get("message", ""),
                }
            ]
        log_table.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        status_path = active_status.get("status_path")
        if status_path:
            st.caption("状态文件：" + str(status_path))
        run_log_path = active_status.get("run_log_path")
        if run_log_path:
            st.caption("日志文件：" + str(run_log_path))
        return

    log_rows = st.session_state.get(RUN_LOG_KEY, [])
    if not log_rows:
        log_table.caption("运行日志将在开始运行后显示。")
        return
    log_table.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
    log_path = st.session_state.get(RUN_LOG_PATH_KEY)
    if log_path:
        st.caption("日志文件：" + str(log_path))


def _read_run_event_rows(log_path_text, limit=10):
    log_path = Path(str(log_path_text or ""))
    if not log_path.exists():
        return []

    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    rows = []
    for line in lines[-limit:]:
        parts = line.split(" | ", 4)
        if len(parts) == 5:
            rows.append({"时间": parts[0], "状态": parts[1], "进度": parts[2], "阶段": parts[3], "消息": parts[4]})
        else:
            rows.append({"时间": "", "状态": "", "进度": "", "阶段": "日志", "消息": line})
    return rows


def _get_recent_log_files(limit=12):
    config = get_config()
    output_dir = get_output_directory(config, create=False)
    log_files = []
    if output_dir.exists():
        for path in output_dir.glob("*/run_logs/run_events.log"):
            status_path = path.parent.parent / "run_status.json"
            status = read_status(status_path)
            state = str(status.get("state", ""))
            if state in {"starting", "running"}:
                continue
            if path.is_file():
                log_files.append(path)
    return sorted(log_files, key=lambda item: item.stat().st_mtime, reverse=True)[:limit]


def _render_recent_log_files(log_files):
    st.subheader("日志文件")

    if not log_files:
        st.caption("还没有生成日志文件。")
        return

    rows = []
    for path in log_files:
        stat = path.stat()
        batch_name = path.parent.parent.name
        rows.append(
            {
                "批次": batch_name,
                "文件名": path.name,
                "修改时间": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "大小 KB": round(stat.st_size / 1024, 2),
                "路径": str(path),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    labels = []
    label_map = {}
    for path in log_files:
        label = path.parent.parent.name + " / " + path.name
        labels.append(label)
        label_map[label] = path

    selected_name = st.selectbox("查看日志内容", labels, key="recent_log_file_select")
    selected_path = label_map[selected_name]
    try:
        text = selected_path.read_text(encoding="utf-8")
    except Exception as error:
        st.warning("读取日志失败：" + str(error))
        return

    st.text_area("日志内容", text, height=360, key="recent_log_file_content")


def _build_preflight_rows(config):
    rows = []
    rows.append({"检查项": "数据配置", "状态": "通过" if _data_ready(config) else "未通过", "说明": "必填数据已配置"})
    rows.append({"检查项": "情景年份", "状态": "通过" if _scenario_ready(config) else "未通过", "说明": f"{config.base_year} 到 {config.target_year}"})
    rows.append({"检查项": "输出目录", "状态": "通过" if _output_dir_ready(config) else "未通过", "说明": config.output_dir})
    history_ready, history_detail = _history_training_ready_detail(config)
    rows.append({"检查项": "历史训练数据", "状态": "通过" if history_ready else "未通过", "说明": history_detail})
    return rows


def _render_section_title(title, subtitle=""):
    if subtitle:
        html_text = f'<div class="data-section-title">{title}<span>{subtitle}</span></div>'
    else:
        html_text = f'<div class="data-section-title">{title}</div>'
    st.markdown(html_text, unsafe_allow_html=True)


def _render_required_path_status(missing_items):
    if not missing_items:
        st.markdown(
            """
            <div class="data-check-card ok">
                <strong>数据检查通过</strong>
                <p>基础栅格、扰动约束数据和历史训练数据均已填写并可读取。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    missing_text = "、".join(missing_items)
    st.markdown(
        f"""
        <div class="data-check-card warn">
            <strong>还有 {len(missing_items)} 项需要补充</strong>
            <p>{missing_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _raster_path_input(label, current_value, project_rasters, key, compact=False):
    select_key = key + "_path_combo"
    pending_key = key + "_pending_path"
    pending_value = ""

    if pending_key in st.session_state:
        pending_value = str(st.session_state[pending_key]).strip()
        del st.session_state[pending_key]
        st.session_state[select_key] = pending_value

    session_value = st.session_state.get(select_key, "")
    default_value = pending_value or session_value or current_value
    options = _build_path_options(default_value, project_rasters, session_value)

    if compact:
        selected_value = _render_path_combo(label, options, select_key, default_value)
        if st.button("浏览...", key=key + "_browse"):
            picked_path = _open_windows_file_dialog(label)
            if picked_path:
                st.session_state[pending_key] = picked_path
                _rerun_page()
    else:
        label_col, path_col, button_col = st.columns([2.2, 6.3, 0.9])
        with label_col:
            st.markdown(f"**{label}**")
        with path_col:
            selected_value = _render_path_combo("路径", options, select_key, default_value)
        with button_col:
            if st.button("浏览", key=key + "_browse", use_container_width=True):
                picked_path = _open_windows_file_dialog(label)
                if picked_path:
                    st.session_state[pending_key] = picked_path
                    _rerun_page()

    return str(selected_value).strip()


def _render_path_combo(label, options, select_key, default_value):
    if select_key in st.session_state:
        return st.selectbox(
            label,
            options,
            key=select_key,
            label_visibility="collapsed",
            accept_new_options=True,
            format_func=_format_path_option,
        )

    return st.selectbox(
        label,
        options,
        index=_find_option_index(options, default_value),
        key=select_key,
        label_visibility="collapsed",
        accept_new_options=True,
        format_func=_format_path_option,
    )


def _build_path_options(current_value, project_rasters, session_value=""):
    options = []
    current_text = str(current_value).strip()
    if current_text:
        options.append(current_text)

    session_text = str(session_value).strip()
    if session_text and session_text not in options:
        options.append(session_text)

    for path_text in project_rasters:
        clean_path = str(path_text).strip()
        if clean_path and clean_path not in options:
            options.append(clean_path)

    if not options:
        options.append("")
    return options


def _format_path_option(value):
    text = str(value).strip()
    if not text:
        return "输入路径或从 data 文件夹选择"
    text = text.replace("\\", "/")
    if text.startswith("../data/"):
        return text
    if "/data/" in text:
        return "..." + text[text.rfind("/data/") :]
    return text


def _find_option_index(options, value):
    value = str(value).strip()
    for index, option in enumerate(options):
        if str(option).strip() == value:
            return index
    return 0


def _folder_path_input(label, current_value, key):
    text_key = key + "_path_text"
    if text_key not in st.session_state:
        st.session_state[text_key] = current_value

    c1, c2 = st.columns([4, 1])
    with c2:
        st.write("")
        if st.button("选择...", key=key + "_browse"):
            picked_path = _open_windows_folder_dialog(label)
            if picked_path:
                st.session_state[text_key] = picked_path
                _rerun_page()
    with c1:
        st.text_input(label, key=text_key)

    return str(st.session_state.get(text_key, "")).strip()


def _open_windows_file_dialog(title):
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="选择" + title,
            filetypes=[
                ("GeoTIFF 栅格文件", "*.tif *.tiff"),
                ("所有文件", "*.*"),
            ],
        )
        root.destroy()
        return str(file_path)
    except Exception as error:
        st.error("无法打开系统文件选择窗口：" + str(error))
        return ""


def _open_windows_folder_dialog(title):
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder_path = filedialog.askdirectory(title="选择" + title)
        root.destroy()
        return str(folder_path)
    except Exception as error:
        st.error("无法打开系统文件夹选择窗口：" + str(error))
        return ""


def _rerun_page():
    st.rerun()


@st.cache_data(show_spinner=False)
def _find_project_rasters():
    project_root = Path(__file__).resolve().parents[3]
    scan_roots = [project_root]

    outer_data_dir = project_root.parent / "data"
    if outer_data_dir.exists():
        scan_roots.append(outer_data_dir)

    paths = []
    for scan_root in scan_roots:
        patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
        for pattern in patterns:
            for path in scan_root.rglob(pattern):
                if _should_skip_path(path):
                    continue
                if scan_root == project_root:
                    relative_path = path.relative_to(project_root)
                    paths.append(str(relative_path).replace("\\", "/"))
                else:
                    relative_path = path.relative_to(project_root.parent)
                    paths.append("../" + str(relative_path).replace("\\", "/"))

    paths = sorted(set(paths))
    if len(paths) > 500:
        return paths[:500]
    return paths


def _should_skip_path(path):
    parts = set(path.parts)
    if "__pycache__" in parts:
        return True
    if ".git" in parts:
        return True
    if "outputs" in parts:
        return True
    return False


def _get_env_path_by_role(env_text, role):
    env_items = parse_env_raster_paths(env_text)
    candidate_words = _env_candidate_words(role)

    for name, path_text in env_items:
        text = (str(name) + " " + str(path_text)).lower()
        for word in candidate_words:
            if word.lower() in text:
                return str(path_text)

    return ""


def _get_extra_env_items(env_text):
    items = []
    env_items = parse_env_raster_paths(env_text)
    fixed_paths = {
        _get_env_path_by_role(env_text, "terrain"),
        _get_env_path_by_role(env_text, "climate"),
        _get_env_path_by_role(env_text, "human"),
    }
    fixed_names = {"slope", "moisture", "accessibility"}

    for name, path_text in env_items:
        clean_name = str(name).strip()
        clean_path = str(path_text).strip()
        if clean_name.lower() in fixed_names:
            continue
        if clean_path in fixed_paths:
            continue
        items.append({"name": clean_name, "path": clean_path})
    return items


def _render_extra_env_factor_form(items, project_rasters):
    count_key = "wizard_extra_env_count"
    if count_key not in st.session_state:
        st.session_state[count_key] = max(3, len(items) + 1)
    if st.button("增加一个环境因子", key="wizard_extra_env_add"):
        st.session_state[count_key] = int(st.session_state[count_key]) + 1
        st.rerun()

    row_count = max(int(st.session_state[count_key]), len(items) + 1)
    rows = _prepare_fixed_rows(items, row_count, {"name": "", "path": ""})
    result = []
    for index in range(len(rows)):
        row = rows[index]
        name_value, path_value = _render_extra_env_factor_row(row, project_rasters, index)
        if str(name_value).strip() and str(path_value).strip():
            result.append({"name": str(name_value).strip(), "path": str(path_value).strip()})
    return result


def _render_extra_env_factor_row(row, project_rasters, index):
    name_key = f"wizard_extra_env_name_{index}"
    path_key = f"wizard_extra_env_path_{index}"
    select_key = path_key + "_path_combo"
    pending_key = path_key + "_pending_path"
    clear_key = f"wizard_extra_env_clear_{index}"
    pending_value = ""

    if clear_key in st.session_state:
        if name_key in st.session_state:
            st.session_state[name_key] = ""
        st.session_state[select_key] = ""
        del st.session_state[clear_key]
        row = {"name": "", "path": ""}
    if pending_key in st.session_state:
        pending_value = str(st.session_state[pending_key]).strip()
        del st.session_state[pending_key]
        st.session_state[select_key] = pending_value

    session_value = st.session_state.get(select_key, "")
    default_value = pending_value or session_value or row.get("path", "")
    options = _build_path_options(default_value, project_rasters, session_value)

    label_col, name_col, path_col, browse_col, delete_col = st.columns([1.2, 1.8, 5.2, 0.8, 0.8])
    with label_col:
        st.markdown("**扩展因子**")
    with delete_col:
        if st.button("删除", key=f"wizard_extra_env_delete_{index}", use_container_width=True):
            st.session_state[clear_key] = True
            st.rerun()
    with browse_col:
        if st.button("浏览", key=f"wizard_extra_env_browse_{index}", use_container_width=True):
            picked_path = _open_windows_file_dialog("扩展环境因子")
            if picked_path:
                st.session_state[pending_key] = picked_path
                st.rerun()
    with name_col:
        name_value = st.text_input("因子名称", value=row.get("name", ""), key=name_key, label_visibility="collapsed", placeholder="如土壤、夜光")
    with path_col:
        path_value = _render_path_combo("路径", options, select_key, default_value)

    return name_value, path_value


def _prepare_fixed_rows(items, count, empty_row):
    rows = []
    for item in items:
        rows.append(dict(item))
    while len(rows) < count:
        rows.append(dict(empty_row))
    return rows[:count]


def _history_row_count_input(config):
    existing_count = max(
        len(_parse_year_path_items(config.history_agbd_paths)),
        len(_parse_year_path_items(config.history_tcc_paths)),
        len(_parse_year_path_items(config.history_lulc_paths)),
        int(config.logging_library_years),
        2,
    )
    if "wizard_history_row_count" not in st.session_state:
        st.session_state["wizard_history_row_count"] = existing_count
    return int(
        st.number_input(
            "历史年份行数",
            min_value=2,
            max_value=30,
            step=1,
            key="wizard_history_row_count",
        )
    )


def _render_year_raster_form(label, text_value, project_rasters, key_prefix, row_count):
    st.markdown(f'<div class="history-group-title">{label}</div>', unsafe_allow_html=True)
    year_head, path_head, browse_head = st.columns([1.45, 6.7, 0.85])
    with year_head:
        st.caption("年份")
    with path_head:
        st.caption("栅格文件")
    with browse_head:
        st.caption("操作")

    items = _parse_year_path_items(text_value)
    rows = _prepare_fixed_rows(items, int(row_count), {"year": "", "path": ""})
    result = []
    for index in range(len(rows)):
        row = rows[index]
        year_value, path_value = _render_year_raster_row(row, project_rasters, key_prefix, index)
        if str(path_value).strip():
            result.append({"year": str(year_value).strip(), "path": str(path_value).strip()})
    return _build_year_path_text(result)


def _render_year_raster_row(row, project_rasters, key_prefix, index):
    year_key = f"{key_prefix}_year_{index}"
    path_key = f"{key_prefix}_path_{index}"
    select_key = path_key + "_path_combo"
    pending_key = path_key + "_pending_path"
    path_watch_key = path_key + "_year_source_path"
    pending_value = ""

    if pending_key in st.session_state:
        pending_value = str(st.session_state[pending_key]).strip()
        del st.session_state[pending_key]
        st.session_state[select_key] = pending_value
    session_value = st.session_state.get(select_key, "")
    default_value = pending_value or session_value or row.get("path", "")
    options = _build_path_options(default_value, project_rasters, session_value)

    path_for_year = str(default_value).strip()
    extracted_year = _extract_year_from_path(path_for_year)
    default_year = str(row.get("year", "")).strip()
    if not default_year:
        default_year = extracted_year

    current_year = str(st.session_state.get(year_key, default_year)).strip()
    last_path = str(st.session_state.get(path_watch_key, "")).strip()
    if extracted_year and path_for_year and path_for_year != last_path:
        st.session_state[year_key] = extracted_year
        st.session_state[path_watch_key] = path_for_year
    elif extracted_year and not _valid_year_text(current_year):
        st.session_state[year_key] = extracted_year

    year_col, path_col, browse_col = st.columns([1.45, 6.7, 0.85])
    with year_col:
        if year_key in st.session_state:
            year_value = st.text_input("年份", key=year_key, label_visibility="collapsed", placeholder="2020")
        else:
            year_value = st.text_input("年份", value=default_year, key=year_key, label_visibility="collapsed", placeholder="2020")
    with path_col:
        path_value = _render_path_combo("路径", options, select_key, default_value)
        if not str(year_value).strip():
            year_value = _extract_year_from_path(path_value)
    with browse_col:
        if st.button("浏览", key=f"{path_key}_browse", use_container_width=True):
            picked_path = _open_windows_file_dialog("历史栅格")
            if picked_path:
                st.session_state[pending_key] = picked_path
                st.rerun()

    return year_value, path_value


def _parse_year_path_items(text_value):
    items = []
    env_items = parse_env_raster_paths(text_value)
    for year_text, path_text in env_items:
        items.append({"year": str(year_text).strip(), "path": str(path_text).strip()})
    return items


def _build_year_path_text(items):
    lines = []
    missing_index = 1
    for item in items:
        year_text = str(item.get("year", "")).strip()
        path_text = str(item.get("path", "")).strip()
        if not path_text:
            continue
        if not year_text:
            lines.append("缺少年份" + str(missing_index) + "=" + path_text)
            missing_index += 1
            continue
        lines.append(year_text + "=" + path_text)
    return "\n".join(lines)


def _extract_year_from_path(path_text):
    text = str(path_text).replace("\\", "/")
    file_name = text.split("/")[-1]
    matches = re.findall(r"(19\d{2}|20\d{2}|21\d{2})", file_name)
    if not matches:
        return ""
    return matches[-1]


def _valid_year_text(text):
    return re.fullmatch(r"(19\d{2}|20\d{2}|21\d{2})", str(text).strip()) is not None


def _build_env_raster_text(terrain_path, climate_path, human_path, extra_items):
    lines = []

    terrain_path = str(terrain_path).strip()
    climate_path = str(climate_path).strip()
    human_path = str(human_path).strip()

    if terrain_path:
        lines.append("slope=" + terrain_path)
    if climate_path:
        lines.append("moisture=" + climate_path)
    if human_path:
        lines.append("accessibility=" + human_path)

    existing_names = {"slope", "moisture", "accessibility"}
    existing_paths = {terrain_path, climate_path, human_path}
    for item in extra_items:
        clean_name = str(item.get("name", "")).strip()
        clean_path = str(item.get("path", "")).strip()
        if not clean_name or not clean_path:
            continue
        if clean_name.lower() in existing_names:
            continue
        if clean_path in existing_paths:
            continue
        lines.append(clean_name + "=" + clean_path)
        existing_paths.add(clean_path)

    return "\n".join(lines)


def _env_candidate_words(role):
    if role == "terrain":
        return ["slope", "terrain", "elevation", "dem", "relief", "地形", "坡度", "高程", "起伏"]
    if role == "climate":
        return ["moisture", "map", "aet", "precip", "rain", "humidity", "water", "气候", "水分", "降水", "湿度", "蒸散"]
    if role == "human":
        return [
            "accessibility",
            "distroadnet",
            "distance",
            "road",
            "urban",
            "settlement",
            "population",
            "human",
            "可达",
            "距离",
            "道路",
            "居民",
            "城镇",
            "人口",
            "人为",
        ]
    return []


def _find_data_missing(
    agbd_path,
    tcc_path,
    lulc_base_path,
    lulc_target_path,
    drivers_path,
    reserve_path,
    terrain_path,
    climate_path,
    human_path,
    history_agbd_paths,
    history_tcc_paths,
    history_lulc_paths,
):
    items = [
        ("森林地上生物量密度（AGBD）基准栅格", agbd_path),
        ("树冠覆盖度基准栅格", tcc_path),
        ("基准年土地利用栅格", lulc_base_path),
        ("目标年预测土地利用栅格", lulc_target_path),
        ("森林损失驱动因素栅格", drivers_path),
        ("自然保护区栅格", reserve_path),
        ("地形因子栅格", terrain_path),
        ("气候水分因子栅格", climate_path),
        ("人为活动因子栅格", human_path),
    ]
    missing = []
    for name, path_text in items:
        problem = _raster_path_problem(name, path_text)
        if problem:
            missing.append(problem)

    history_missing = _find_history_data_missing(history_agbd_paths, history_tcc_paths, history_lulc_paths)
    missing.extend(history_missing)
    return missing


def _find_history_data_missing(history_agbd_paths, history_tcc_paths, history_lulc_paths):
    agbd_paths, agbd_errors = _validate_year_raster_paths("历史森林地上生物量密度（AGBD）", history_agbd_paths)
    tcc_paths, tcc_errors = _validate_year_raster_paths("历史树冠覆盖度", history_tcc_paths)
    lulc_paths, lulc_errors = _validate_year_raster_paths("历史土地利用", history_lulc_paths)

    missing = []
    missing.extend(agbd_errors)
    missing.extend(tcc_errors)
    missing.extend(lulc_errors)
    if missing:
        return missing

    agbd_years = set(agbd_paths.keys())
    tcc_years = set(tcc_paths.keys())
    lulc_years = set(lulc_paths.keys())
    common_years = sorted(agbd_years & tcc_years & lulc_years)
    all_years = sorted(agbd_years | tcc_years | lulc_years)
    if all_years and set(common_years) != set(all_years):
        missing.append("历史AGBD、树冠覆盖度和土地利用年份不一致，请使用相同年份")
        return missing

    valid_years = []
    for year in common_years:
        valid_years.append(year)

    if len(valid_years) < 2:
        missing.append("历史训练数据至少需要两个共同年份的AGBD、树冠覆盖度和土地利用栅格")
        return missing

    has_adjacent_years = False
    for index in range(len(valid_years) - 1):
        if int(valid_years[index + 1]) - int(valid_years[index]) == 1:
            has_adjacent_years = True
            break
    if not has_adjacent_years:
        missing.append("历史训练数据需要至少一组连续年份")

    return missing


def _validate_year_raster_paths(label, text_value):
    paths = {}
    errors = []
    raw_text = str(text_value or "").strip()
    if raw_text == "":
        return paths, [label + "未填写"]
    pieces = raw_text.replace("\r", "\n").replace(";", "\n").split("\n")
    for piece in pieces:
        clean_piece = piece.strip()
        if clean_piece == "" or "=" not in clean_piece:
            continue
        year_text, path_text = clean_piece.split("=", 1)
        year_text = year_text.strip()
        path_text = path_text.strip()
        if path_text == "":
            continue
        if not _valid_year_text(year_text):
            errors.append(label + "存在未填写或格式不正确的年份")
            continue
        problem = _raster_path_problem(label + " " + year_text, path_text)
        if problem:
            errors.append(problem)
            continue
        year = int(year_text)
        if year in paths:
            errors.append(label + "存在重复年份：" + str(year))
            continue
        paths[year] = path_text
    if not paths and not errors:
        errors.append(label + "未填写")
    return paths, errors


def _raster_path_problem(label, path_text):
    text = str(path_text or "").strip()
    if text == "":
        return label + "未填写"
    suffix = resolve_input_path(text).suffix.lower()
    if suffix not in {".tif", ".tiff"}:
        return label + "格式不正确，需要 GeoTIFF（.tif 或 .tiff）"
    if not path_exists(text):
        return label + "文件不存在或路径无法读取"
    return ""


def _history_training_ready_detail(config):
    errors = _find_history_data_missing(config.history_agbd_paths, config.history_tcc_paths, config.history_lulc_paths)
    if errors:
        return False, errors[0]

    agbd_paths = parse_year_raster_paths(config.history_agbd_paths)
    years = sorted(agbd_paths.keys())
    if not years:
        return False, "历史训练数据未填写"
    return True, "共同年份：" + "、".join(str(year) for year in years)


def _save_and_clear(config):
    set_config(config)
    _reset_run_progress()


def _clear_results_only():
    _reset_run_progress()


def _reset_run_progress():
    st.session_state[RUN_PROGRESS_KEY] = 0
    st.session_state[RUN_STAGE_KEY] = "等待开始"
    st.session_state[RUN_MESSAGE_KEY] = "点击开始运行后，这里会显示实时进度。"


def _clear_data_widget_state():
    for key in list(st.session_state.keys()):
        if key.startswith("wizard_") and (
            key.endswith("_path_text")
            or key.endswith("_path_combo")
            or key.endswith("_pending_path")
            or key.startswith("wizard_extra_env_")
            or key == "wizard_extra_env_count"
            or key.startswith("wizard_hist_")
            or key == "wizard_output_dir_path_text"
        ):
            del st.session_state[key]


def _go_to_step(step_index):
    st.session_state[WIZARD_STEP_KEY] = step_index
    st.rerun()


def _data_ready(config):
    missing = _find_data_missing(
        config.agbd_raster_path,
        config.tcc_raster_path,
        config.lulc_base_raster_path,
        config.lulc_target_raster_path,
        config.drivers_raster_path,
        config.reserve_raster_path,
        _get_env_path_by_role(config.env_raster_paths, "terrain"),
        _get_env_path_by_role(config.env_raster_paths, "climate"),
        _get_env_path_by_role(config.env_raster_paths, "human"),
        config.history_agbd_paths,
        config.history_tcc_paths,
        config.history_lulc_paths,
    )
    return len(missing) == 0


def _scenario_ready(config):
    return int(config.target_year) > int(config.base_year)


def _output_dir_ready(config):
    try:
        get_output_directory(config, create=False)
        return True
    except Exception:
        return False
