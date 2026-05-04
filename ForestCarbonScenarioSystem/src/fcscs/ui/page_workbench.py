from datetime import datetime
from pathlib import Path
import time
import traceback

import pandas as pd
import streamlit as st

from fcscs.config.defaults import ScenarioConfig, build_preset_config, list_preset_names, sanitize_scenario_name
from fcscs.engines.raster_tools import parse_env_raster_paths, parse_year_raster_paths, path_exists
from fcscs.services.quick_run_service import build_quick_config
from fcscs.services.workflow_service import run_simulation_workflow
from fcscs.ui.app_state import (
    add_simulation_history,
    get_config,
    get_report_bundle,
    get_simulation_bundle,
    set_config,
    set_events,
    set_logging_patch_library,
    set_report_bundle,
    set_simulation_bundle,
)
from fcscs.ui.common import get_output_directory
from fcscs.ui.result_views import export_report
from fcscs.ui.styles import render_page_banner


WIZARD_STEP_KEY = "workbench_wizard_step"
RUN_PROGRESS_KEY = "workbench_run_progress"
RUN_STAGE_KEY = "workbench_run_stage"
RUN_MESSAGE_KEY = "workbench_run_message"
RUN_LOG_KEY = "workbench_run_log"
RUN_LOG_PATH_KEY = "workbench_run_log_path"
RUN_STAGES = ["等待开始", "检查输入", "准备快速测试", "生成事件", "经验强度采样", "训练模型", "蒙特卡洛模拟", "汇总结果", "完成"]
WIZARD_STEPS = [
    ("data", "数据准备"),
    ("scenario", "情景方案"),
    ("run", "运行检查"),
]


def render_workbench_page():
    render_page_banner("工作台", "")
    _ensure_wizard_state()

    config = get_config()
    step_index = int(st.session_state[WIZARD_STEP_KEY])
    if step_index == 0:
        _render_data_step(config)
    elif step_index == 1:
        _render_scenario_step(config)
    else:
        _render_run_step(config)


def _ensure_wizard_state():
    if WIZARD_STEP_KEY not in st.session_state:
        st.session_state[WIZARD_STEP_KEY] = 0

    try:
        current = int(st.session_state[WIZARD_STEP_KEY])
    except Exception:
        current = 0

    if current < 0 or current >= len(WIZARD_STEPS):
        current = 0
    st.session_state[WIZARD_STEP_KEY] = current


def render_workbench_step_sidebar():
    _ensure_wizard_state()
    config = get_config()
    labels = _build_step_labels(config)
    current_step = int(st.session_state[WIZARD_STEP_KEY])
    selected = st.sidebar.radio("流程步骤", labels, index=current_step)
    selected_index = labels.index(selected)
    if selected_index != current_step:
        st.session_state[WIZARD_STEP_KEY] = selected_index
        st.session_state["sidebar_panel"] = "workflow"
        st.rerun()


def render_run_log_page():
    render_page_banner("运行日志", "集中查看本次运行记录、异常提示和历史日志文件。")

    log_rows = st.session_state.get(RUN_LOG_KEY, [])
    log_path = st.session_state.get(RUN_LOG_PATH_KEY)
    recent_log_files = _get_recent_log_files()

    c1, c2, c3 = st.columns(3)
    c1.metric("本次日志", len(log_rows))
    c2.metric("历史文件", len(recent_log_files))
    c3.metric("当前状态", st.session_state.get(RUN_STAGE_KEY, "等待开始"))

    c_back, c_path = st.columns([1, 2])
    with c_back:
        if st.button("返回当前流程步骤", type="primary", use_container_width=True, key="log_page_back_to_workflow"):
            st.session_state["sidebar_panel"] = "workflow"
            st.rerun()
    with c_path:
        if log_path:
            st.caption("当前日志文件：" + str(log_path))
        else:
            st.caption("当前还没有新的运行日志。开始运行后会自动记录。")

    st.subheader("本次运行记录")
    if log_rows:
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
    else:
        st.info("本次会话暂无运行日志。")

    _render_recent_log_files(recent_log_files)


def _build_step_labels(config):
    data_ok = _data_ready(config)
    scenario_ok = _scenario_ready(config)
    current_step = int(st.session_state[WIZARD_STEP_KEY])
    run_ok = get_report_bundle() is not None
    completed = [data_ok, scenario_ok, run_ok]

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
    st.subheader("1 数据准备")

    use_raster_data = True
    output_dir = _folder_path_input("输出文件夹", current.output_dir, "wizard_output_dir")
    project_rasters = _find_project_rasters()

    _render_section_title("基础栅格数据", "用于确定基准年森林碳储量和目标年土地利用变化。")
    agbd_raster_path = _raster_path_input("森林地上生物量密度（AGBD）基准栅格", current.agbd_raster_path, project_rasters, "wizard_agbd")
    tcc_raster_path = _raster_path_input("树冠覆盖度基准栅格", current.tcc_raster_path, project_rasters, "wizard_tcc")
    lulc_base_raster_path = _raster_path_input("基准年土地利用栅格", current.lulc_base_raster_path, project_rasters, "wizard_lulc_base")
    lulc_target_raster_path = _raster_path_input("目标年预测土地利用栅格", current.lulc_target_raster_path, project_rasters, "wizard_lulc_target")

    _render_section_title("扰动与约束数据", "森林损失驱动因素用于识别采伐发生位置，自然保护区用于避让约束。")
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

    _render_section_title("历史训练数据", "必填。连续年份树冠覆盖度用于量化森林损失强度，Drivers 用于定位采伐发生位置。")
    with st.expander("历史训练数据输入", expanded=True):
        use_history_training = True
        st.caption("至少需要两组相同年份的历史森林地上生物量密度、树冠覆盖度和土地利用栅格，系统会按相邻年份构造训练样本。")
        history_agbd_paths = _render_year_raster_form("历史森林地上生物量密度（AGBD）", current.history_agbd_paths, project_rasters, "wizard_hist_agbd")
        history_tcc_paths = _render_year_raster_form("历史树冠覆盖度", current.history_tcc_paths, project_rasters, "wizard_hist_tcc")
        history_lulc_paths = _render_year_raster_form("历史土地利用", current.history_lulc_paths, project_rasters, "wizard_hist_lulc")

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

    grid_rows = current.grid_rows
    grid_cols = current.grid_cols

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
            new_config.use_raster_data = True
            new_config.agbd_raster_path = agbd_raster_path
            new_config.tcc_raster_path = tcc_raster_path
            new_config.lulc_base_raster_path = lulc_base_raster_path
            new_config.lulc_target_raster_path = lulc_target_raster_path
            new_config.drivers_raster_path = drivers_raster_path
            new_config.reserve_raster_path = reserve_raster_path
            new_config.env_raster_paths = env_raster_paths
            new_config.use_history_training = bool(use_history_training)
            new_config.history_agbd_paths = history_agbd_paths
            new_config.history_tcc_paths = history_tcc_paths
            new_config.history_lulc_paths = history_lulc_paths
            new_config.forest_lulc_codes = forest_lulc_codes
            new_config.urban_lulc_codes = urban_lulc_codes
            new_config.logging_driver_value = int(logging_driver_value)
            new_config.reserve_value = int(reserve_value)
            new_config.write_raster_outputs = bool(write_raster_outputs)
            new_config.output_dir = output_dir
            new_config.grid_rows = int(grid_rows)
            new_config.grid_cols = int(grid_cols)

            _save_and_clear(new_config)
            _go_to_step(1)


def _render_scenario_step(current):
    st.subheader("2 情景方案")

    c0, c_load = st.columns([3, 1])
    with c0:
        preset_name = st.selectbox("预设方案", list_preset_names(), key="wizard_preset")
    with c_load:
        st.write("")
        if st.button("载入预设", use_container_width=True, key="wizard_load_preset"):
            preset_config = build_preset_config(preset_name)
            _copy_data_settings(current, preset_config)
            _save_and_clear(preset_config)
            st.rerun()

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
            severity_method = st.selectbox("扰动强度方法", ["S1", "S2"], index=0 if current.severity_method == "S1" else 1, key="wizard_sev_method")
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

        c_sev, c_note = st.columns([1, 2])
        with c_sev:
            severity_sample_count = st.number_input("经验强度样本数", min_value=100, max_value=200000, value=current.severity_sample_count, step=100, key="wizard_sev_sample_count")
        with c_note:
            st.caption("历史训练数据为必填，系统会按树冠覆盖度、地形、气候水分和人为活动因子分层抽取扰动强度。")

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


def _render_run_step(config):
    st.subheader("3 运行检查")

    checks = _build_preflight_rows(config)
    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)
    can_run = all(row["状态"] == "通过" for row in checks)

    if not can_run:
        st.warning("请先处理未通过的检查项。")

    run_mode = st.radio(
        "运行方式",
        ["快速测试", "完整运行"],
        horizontal=True,
        key="wizard_run_mode",
    )
    quick_size = 520
    if run_mode == "快速测试" and config.use_raster_data:
        quick_size = st.number_input("测试窗口大小", min_value=128, max_value=1200, value=520, step=64, key="wizard_quick_size")
    if run_mode == "快速测试":
        effective_mc = min(int(config.mc_n_simulations), 3)
        st.caption(f"快速测试会临时使用较小计算量：本次实际蒙特卡洛次数为 {effective_mc} 次；完整运行将使用情景设置的 {int(config.mc_n_simulations)} 次。")
    else:
        st.caption(f"完整运行将使用情景设置的蒙特卡洛次数：{int(config.mc_n_simulations)} 次。")

    progress_bar, status_table, message_box, log_table = _render_progress_area()

    c_back, c_run = st.columns([1, 2])
    with c_back:
        if st.button("返回情景方案", use_container_width=True, key="wizard_back_to_scenario"):
            _go_to_step(1)
    with c_run:
        run_clicked = st.button(
            "开始运行",
            type="primary",
            use_container_width=True,
            disabled=not can_run,
            key="wizard_start_run",
        )

    if run_clicked:
        _clear_results_only()
        _run_with_progress(config, run_mode, int(quick_size), progress_bar, status_table, message_box, log_table)

    if get_report_bundle() is not None:
        _render_run_complete_actions()


def _render_run_complete_actions():
    st.success("运行已完成。结果已经自动保存，可以进入结果查看。")
    c_keep, c_result = st.columns([1, 2])
    with c_keep:
        st.caption("需要重新运行时，可以调整上方运行方式后再次点击开始运行。")
    with c_result:
        if st.button("查看结果", type="primary", use_container_width=True, key="wizard_go_to_results_after_run"):
            st.session_state["sidebar_panel"] = "results"
            st.rerun()


def _render_progress_area():
    progress_value = int(st.session_state.get(RUN_PROGRESS_KEY, 0))
    stage = str(st.session_state.get(RUN_STAGE_KEY, "等待开始"))
    message = str(st.session_state.get(RUN_MESSAGE_KEY, "点击开始运行后，这里会显示实时进度。"))

    if get_report_bundle() is not None and progress_value < 100:
        progress_value = 100
        stage = "完成"
        message = "运行已完成。"

    st.markdown("**运行进度**")
    progress_bar = st.progress(progress_value, text=f"{progress_value}% {stage}")
    status_table = st.empty()
    status_table.dataframe(_build_run_stage_frame(stage), use_container_width=True, hide_index=True)
    message_box = st.empty()
    message_box.info(message)
    log_table = st.empty()
    _render_log_table(log_table)
    return progress_bar, status_table, message_box, log_table


def _run_with_progress(config, run_mode, quick_size, progress_bar, status_table, message_box, log_table):
    run_label = run_mode
    run_config = config
    log_path = _start_run_log(config)

    try:
        _update_progress(progress_bar, status_table, message_box, log_table, log_path, 1, "检查输入", "正在读取已保存的配置。")

        if run_mode == "快速测试":
            _update_progress(progress_bar, status_table, message_box, log_table, log_path, 8, "准备快速测试", "正在准备快速测试输入。")
            run_config = build_quick_config(config, quick_size)
            run_label = "快速测试"

        result = run_simulation_workflow(
            run_config,
            progress_callback=lambda percent, stage, message: _update_progress(
                progress_bar,
                status_table,
                message_box,
                log_table,
                log_path,
                percent,
                stage,
                message,
            ),
        )
        set_events(result.events)
        set_logging_patch_library(result.patch_library)
        set_simulation_bundle(result.simulation_bundle)
        set_report_bundle(result.report_bundle)
        export_dir = get_output_directory(config) / "report_exports" / sanitize_scenario_name(result.report_bundle.scenario_name)
        export_report(result.report_bundle, export_dir)

        _add_history_from_bundle(result.simulation_bundle, run_label)
        _update_progress(progress_bar, status_table, message_box, log_table, log_path, 100, "完成", "运行完成，结果已保存，可以查看结果。")
    except ValueError as error:
        _update_progress(progress_bar, status_table, message_box, log_table, log_path, 0, "等待开始", "运行前检查失败。")
        _record_error(log_path, error)
        st.error(str(error))
    except Exception as error:
        _update_progress(progress_bar, status_table, message_box, log_table, log_path, 0, "等待开始", "运行失败。")
        _record_error(log_path, error)
        st.error("运行失败：" + str(error))
        with st.expander("错误详情"):
            st.code(traceback.format_exc(), language="text")


def _update_progress(progress_bar, status_table, message_box, log_table, log_path, percent, stage, message):
    st.session_state[RUN_PROGRESS_KEY] = int(percent)
    st.session_state[RUN_STAGE_KEY] = stage
    st.session_state[RUN_MESSAGE_KEY] = message
    _append_run_log(log_path, stage, message)
    progress_bar.progress(int(percent), text=f"{int(percent)}% {stage}")
    status_table.dataframe(_build_run_stage_frame(stage), use_container_width=True, hide_index=True)
    message_box.info(message)
    _render_log_table(log_table)
    time.sleep(0.15)


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


def _add_history_from_bundle(bundle, run_label):
    row = {
        "运行方式": run_label,
        "情景名称": bundle.scenario_name,
        "模拟次数": bundle.summary["n_simulations"],
        "平均AGBD": round(bundle.summary["mean_agbd_per_ha"], 4),
        "平均AGC": round(bundle.summary["mean_agc_per_ha"], 4),
        "模型R2": round(bundle.summary["mean_model_r2"], 4),
        "模型MAE": round(bundle.summary["mean_model_mae"], 4),
    }
    add_simulation_history(row)


def _start_run_log(config):
    st.session_state[RUN_LOG_KEY] = []
    log_dir = get_output_directory(config) / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / (timestamp + "_" + sanitize_scenario_name(config.scenario_name) + ".log")
    st.session_state[RUN_LOG_PATH_KEY] = str(log_path)
    _append_run_log(log_path, "开始", "运行日志已创建：" + str(log_path))
    return log_path


def _append_run_log(log_path, stage, message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    row = {"时间": timestamp, "阶段": stage, "消息": message}
    log_rows = list(st.session_state.get(RUN_LOG_KEY, []))
    log_rows.append(row)
    st.session_state[RUN_LOG_KEY] = log_rows
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(f"{timestamp} [{stage}] {message}\n")


def _record_error(log_path, error):
    error_text = traceback.format_exc()
    _append_run_log(log_path, "异常", str(error))
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(error_text)


def _render_log_table(log_table):
    log_rows = st.session_state.get(RUN_LOG_KEY, [])
    if not log_rows:
        log_table.caption("运行日志将在开始运行后显示。")
        return
    log_table.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
    log_path = st.session_state.get(RUN_LOG_PATH_KEY)
    if log_path:
        st.caption("日志文件：" + str(log_path))


def _get_recent_log_files(limit=12):
    config = get_config()
    log_dir = get_output_directory(config) / "run_logs"
    if not log_dir.exists():
        return []
    return sorted(log_dir.glob("*.log"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]


def _render_recent_log_files(log_files):
    st.subheader("历史日志文件")

    if not log_files:
        st.caption("还没有生成历史日志文件。")
        return

    rows = []
    for path in log_files:
        stat = path.stat()
        rows.append(
            {
                "文件名": path.name,
                "修改时间": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "大小 KB": round(stat.st_size / 1024, 2),
                "路径": str(path),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    selected_name = st.selectbox("查看日志内容", [path.name for path in log_files], key="recent_log_file_select")
    selected_path = next(path for path in log_files if path.name == selected_name)
    try:
        text = selected_path.read_text(encoding="utf-8")
    except Exception as error:
        st.warning("读取日志失败：" + str(error))
        return

    st.text_area("日志内容", text, height=360, key="recent_log_file_content")


def _build_preflight_rows(config):
    rows = []
    rows.append(_check_row("数据配置", _data_ready(config), "必填数据已配置"))
    rows.append(_check_row("情景年份", _scenario_ready(config), f"{config.base_year} 到 {config.target_year}"))
    rows.append(_check_row("输出目录", _output_dir_ready(config), getattr(config, "output_dir", "../ForestCarbonScenarioSystem_outputs")))
    rows.append(_check_row("运行模式", True, "真实栅格数据"))
    return rows


def _check_row(item, passed, detail):
    return {"检查项": item, "状态": "通过" if passed else "未通过", "说明": detail}


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
        if select_key in st.session_state:
            del st.session_state[select_key]

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


def _initial_path_value(current_value, options):
    current_text = str(current_value).strip()
    if current_text:
        return current_text
    if options:
        return str(options[0]).strip()
    return ""


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
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


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
            del st.session_state[name_key]
        if select_key in st.session_state:
            del st.session_state[select_key]
        del st.session_state[clear_key]
        row = {"name": "", "path": ""}
    if pending_key in st.session_state:
        pending_value = str(st.session_state[pending_key]).strip()
        del st.session_state[pending_key]
        if select_key in st.session_state:
            del st.session_state[select_key]

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


def _render_year_raster_form(label, text_value, project_rasters, key_prefix):
    st.markdown(f'<div class="history-group-title">{label}</div>', unsafe_allow_html=True)
    items = _parse_year_path_items(text_value)
    rows = _prepare_fixed_rows(items, 6, {"year": "", "path": ""})
    result = []
    for index in range(len(rows)):
        row = rows[index]
        year_value, path_value = _render_year_raster_row(row, project_rasters, key_prefix, index)
        if str(year_value).strip() and str(path_value).strip():
            result.append({"year": str(year_value).strip(), "path": str(path_value).strip()})
    return _build_year_path_text(result)


def _render_year_raster_row(row, project_rasters, key_prefix, index):
    year_key = f"{key_prefix}_year_{index}"
    path_key = f"{key_prefix}_path_{index}"
    select_key = path_key + "_path_combo"
    pending_key = path_key + "_pending_path"
    pending_value = ""

    if pending_key in st.session_state:
        pending_value = str(st.session_state[pending_key]).strip()
        del st.session_state[pending_key]
        if select_key in st.session_state:
            del st.session_state[select_key]
    session_value = st.session_state.get(select_key, "")
    default_value = pending_value or session_value or row.get("path", "")
    options = _build_path_options(default_value, project_rasters, session_value)

    row_label_col, year_col, path_col, browse_col = st.columns([1.2, 0.85, 6.1, 0.85])
    with row_label_col:
        st.caption("年份数据")
    with year_col:
        year_value = st.text_input("年份", value=str(row.get("year", "")), key=year_key, label_visibility="collapsed", placeholder="2020")
    with path_col:
        path_value = _render_path_combo("路径", options, select_key, default_value)
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
    for item in items:
        year_text = str(item.get("year", "")).strip()
        path_text = str(item.get("path", "")).strip()
        if not year_text or not path_text:
            continue
        lines.append(year_text + "=" + path_text)
    return "\n".join(lines)


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
        if not path_exists(path_text):
            missing.append(name)

    history_missing = _find_history_data_missing(history_agbd_paths, history_tcc_paths, history_lulc_paths)
    missing.extend(history_missing)
    return missing


def _find_history_data_missing(history_agbd_paths, history_tcc_paths, history_lulc_paths):
    agbd_paths = parse_year_raster_paths(history_agbd_paths)
    tcc_paths = parse_year_raster_paths(history_tcc_paths)
    lulc_paths = parse_year_raster_paths(history_lulc_paths)

    missing = []
    common_years = sorted(set(agbd_paths.keys()) & set(tcc_paths.keys()) & set(lulc_paths.keys()))
    valid_years = []
    for year in common_years:
        if not path_exists(agbd_paths[year]):
            continue
        if not path_exists(tcc_paths[year]):
            continue
        if not path_exists(lulc_paths[year]):
            continue
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


def _copy_data_settings(source, target):
    target.use_raster_data = source.use_raster_data
    target.agbd_raster_path = source.agbd_raster_path
    target.tcc_raster_path = source.tcc_raster_path
    target.lulc_base_raster_path = source.lulc_base_raster_path
    target.lulc_target_raster_path = source.lulc_target_raster_path
    target.drivers_raster_path = source.drivers_raster_path
    target.reserve_raster_path = source.reserve_raster_path
    target.env_raster_paths = source.env_raster_paths
    target.use_history_training = source.use_history_training
    target.history_agbd_paths = source.history_agbd_paths
    target.history_tcc_paths = source.history_tcc_paths
    target.history_lulc_paths = source.history_lulc_paths
    target.forest_lulc_codes = source.forest_lulc_codes
    target.urban_lulc_codes = source.urban_lulc_codes
    target.logging_driver_value = source.logging_driver_value
    target.reserve_value = source.reserve_value
    target.write_raster_outputs = source.write_raster_outputs
    target.output_dir = source.output_dir
    target.grid_rows = source.grid_rows
    target.grid_cols = source.grid_cols


def _save_and_clear(config):
    set_config(config)
    set_events([])
    set_logging_patch_library(None)
    set_simulation_bundle(None)
    set_report_bundle(None)
    _reset_run_progress()


def _clear_results_only():
    set_events([])
    set_logging_patch_library(None)
    set_simulation_bundle(None)
    set_report_bundle(None)
    _reset_run_progress()


def _reset_run_progress():
    st.session_state[RUN_PROGRESS_KEY] = 0
    st.session_state[RUN_STAGE_KEY] = "等待开始"
    st.session_state[RUN_MESSAGE_KEY] = "点击开始运行后，这里会显示实时进度。"


def _clear_data_widget_state():
    for key in list(st.session_state.keys()):
        if key.startswith("wizard_") and (
            key.endswith("_path_text")
            or key.endswith("_project_select")
            or key.endswith("_path_combo")
            or key.endswith("_pending_path")
            or key.startswith("wizard_extra_env_")
            or key == "wizard_extra_env_count"
            or key.startswith("wizard_hist_")
            or key in {"wizard_data_mode", "wizard_output_dir_path_text", "wizard_use_history_training"}
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
        get_output_directory(config)
        return True
    except Exception:
        return False


def _status_text(value):
    return "完成" if value else "待处理"
