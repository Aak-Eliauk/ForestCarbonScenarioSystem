from datetime import datetime
from pathlib import Path
import time
import traceback

import pandas as pd
import streamlit as st

from fcscs.config.defaults import ScenarioConfig, build_preset_config, list_preset_names, sanitize_scenario_name
from fcscs.engines.raster_tools import parse_env_raster_paths, path_exists
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
RUN_STAGES = ["等待开始", "检查输入", "准备快速测试", "生成事件", "计算强度", "模型预测", "汇总结果", "完成"]
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

    mode_index = 1 if current.use_raster_data else 0
    data_mode = st.radio(
        "数据来源",
        ["演示网格", "真实栅格"],
        index=mode_index,
        horizontal=True,
        key="wizard_data_mode",
    )
    use_raster_data = data_mode == "真实栅格"
    output_dir = _folder_path_input("输出文件夹", current.output_dir, "wizard_output_dir")

    project_rasters = _find_project_rasters()

    if use_raster_data:
        st.markdown("**必填栅格**")
        c1, c2 = st.columns(2)
        with c1:
            agbd_raster_path = _raster_path_input("AGBD 基准栅格", current.agbd_raster_path, project_rasters, "wizard_agbd")
            lulc_base_raster_path = _raster_path_input("基准年 LULC", current.lulc_base_raster_path, project_rasters, "wizard_lulc_base")
        with c2:
            tcc_raster_path = _raster_path_input("TCC 树冠覆盖", current.tcc_raster_path, project_rasters, "wizard_tcc")
            lulc_target_raster_path = _raster_path_input("目标年 LULC", current.lulc_target_raster_path, project_rasters, "wizard_lulc_target")

        st.markdown("**可选栅格**")
        c3, c4 = st.columns(2)
        with c3:
            drivers_raster_path = _raster_path_input("Drivers 扰动来源", current.drivers_raster_path, project_rasters, "wizard_drivers")
        with c4:
            reserve_raster_path = _raster_path_input("保护区", current.reserve_raster_path, project_rasters, "wizard_reserve")

        selected_env_paths = st.multiselect(
            "环境因子",
            project_rasters,
            default=_find_existing_env_defaults(current.env_raster_paths, project_rasters),
            key="wizard_env_select",
        )
        env_raster_paths = st.text_area(
            "环境因子路径",
            value=current.env_raster_paths,
            height=90,
            key="wizard_env_text",
        )

        _render_required_path_status(
            [
                ("AGBD 基准栅格", agbd_raster_path),
                ("TCC 树冠覆盖", tcc_raster_path),
                ("基准年 LULC", lulc_base_raster_path),
                ("目标年 LULC", lulc_target_raster_path),
            ]
        )

        with st.expander("高级数据设置"):
            c5, c6, c7, c8 = st.columns(4)
            with c5:
                forest_lulc_codes = st.text_input("森林 LULC 编码", value=current.forest_lulc_codes, key="wizard_forest_codes")
            with c6:
                urban_lulc_codes = st.text_input("城镇 LULC 编码", value=current.urban_lulc_codes, key="wizard_urban_codes")
            with c7:
                logging_driver_value = st.number_input("采伐 Drivers 编码", value=current.logging_driver_value, step=1, key="wizard_logging_value")
            with c8:
                reserve_value = st.number_input("保护区编码", value=current.reserve_value, step=1, key="wizard_reserve_value")
            write_raster_outputs = st.checkbox("输出 GeoTIFF", value=current.write_raster_outputs, key="wizard_write_tif")

        grid_rows = current.grid_rows
        grid_cols = current.grid_cols
    else:
        agbd_raster_path = current.agbd_raster_path
        tcc_raster_path = current.tcc_raster_path
        lulc_base_raster_path = current.lulc_base_raster_path
        lulc_target_raster_path = current.lulc_target_raster_path
        drivers_raster_path = current.drivers_raster_path
        reserve_raster_path = current.reserve_raster_path
        env_raster_paths = current.env_raster_paths
        selected_env_paths = []
        forest_lulc_codes = current.forest_lulc_codes
        urban_lulc_codes = current.urban_lulc_codes
        logging_driver_value = current.logging_driver_value
        reserve_value = current.reserve_value
        write_raster_outputs = current.write_raster_outputs

        c1, c2 = st.columns(2)
        with c1:
            grid_rows = st.number_input("演示网格行数", min_value=16, max_value=512, value=current.grid_rows, step=16, key="wizard_rows")
        with c2:
            grid_cols = st.number_input("演示网格列数", min_value=16, max_value=512, value=current.grid_cols, step=16, key="wizard_cols")

    c_back, c_save = st.columns([1, 2])
    with c_back:
        if st.button("重置为当前配置", use_container_width=True, key="wizard_data_reset"):
            _clear_data_widget_state()
            st.rerun()
    with c_save:
        if st.button("保存数据并继续", type="primary", use_container_width=True, key="wizard_save_data"):
            env_raster_paths = _merge_env_paths(env_raster_paths, selected_env_paths)
            missing = []
            if use_raster_data:
                missing = _find_required_missing(
                    agbd_raster_path,
                    tcc_raster_path,
                    lulc_base_raster_path,
                    lulc_target_raster_path,
                )
            if missing:
                st.error("必填栅格缺失：" + "、".join(missing))
                return

            new_config = current.copy()
            new_config.use_raster_data = bool(use_raster_data)
            new_config.agbd_raster_path = agbd_raster_path
            new_config.tcc_raster_path = tcc_raster_path
            new_config.lulc_base_raster_path = lulc_base_raster_path
            new_config.lulc_target_raster_path = lulc_target_raster_path
            new_config.drivers_raster_path = drivers_raster_path
            new_config.reserve_raster_path = reserve_raster_path
            new_config.env_raster_paths = env_raster_paths
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
    if config.use_raster_data:
        rows.append(_check_row("运行模式", True, "真实栅格建议先快速测试"))
    else:
        rows.append(_check_row("运行模式", True, f"演示网格 {config.grid_rows} x {config.grid_cols}"))
    return rows


def _check_row(item, passed, detail):
    return {"检查项": item, "状态": "通过" if passed else "未通过", "说明": detail}


def _render_required_path_status(items):
    rows = []
    for name, path_text in items:
        rows.append(
            {
                "数据项": name,
                "状态": "已找到" if path_exists(path_text) else "缺失",
                "路径": str(path_text),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _raster_path_input(label, current_value, project_rasters, key):
    text_key = key + "_path_text"
    select_key = key + "_project_select"
    default_option = "手动输入或浏览文件"

    if text_key not in st.session_state:
        st.session_state[text_key] = current_value

    options = [default_option]
    options.extend(project_rasters)

    selected = st.selectbox(label + " 项目内文件", options, index=0, key=select_key)
    if selected != default_option:
        st.session_state[text_key] = selected

    c1, c2 = st.columns([4, 1])
    with c2:
        st.write("")
        if st.button("浏览...", key=key + "_browse"):
            picked_path = _open_windows_file_dialog(label)
            if picked_path:
                st.session_state[text_key] = picked_path
                _rerun_page()
    with c1:
        st.text_input(label + " 路径", key=text_key)

    return str(st.session_state.get(text_key, "")).strip()


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
                    paths.append(str(path))

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


def _find_existing_env_defaults(env_text, project_rasters):
    defaults = []
    env_items = parse_env_raster_paths(env_text)
    for _, path_text in env_items:
        normalized = str(path_text).replace("\\", "/")
        if normalized in project_rasters:
            defaults.append(normalized)
    return defaults


def _merge_env_paths(text_value, selected_paths):
    lines = []

    for line in str(text_value).splitlines():
        clean_line = line.strip()
        if clean_line:
            lines.append(clean_line)

    existing_paths = set()
    for _, path_text in parse_env_raster_paths("\n".join(lines)):
        existing_paths.add(str(path_text).replace("\\", "/"))

    for path_text in selected_paths:
        normalized = str(path_text).replace("\\", "/")
        if normalized in existing_paths:
            continue
        name = Path(normalized).stem
        lines.append(name + "=" + normalized)
        existing_paths.add(normalized)

    return "\n".join(lines)


def _find_required_missing(agbd_path, tcc_path, lulc_base_path, lulc_target_path):
    items = [
        ("AGBD", agbd_path),
        ("TCC", tcc_path),
        ("基准年 LULC", lulc_base_path),
        ("目标年 LULC", lulc_target_path),
    ]
    missing = []
    for name, path_text in items:
        if not path_exists(path_text):
            missing.append(name)
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
            or key in {"wizard_data_mode", "wizard_output_dir_path_text", "wizard_env_select", "wizard_env_text"}
        ):
            del st.session_state[key]


def _go_to_step(step_index):
    st.session_state[WIZARD_STEP_KEY] = step_index
    st.rerun()


def _data_ready(config):
    if not config.use_raster_data:
        return config.grid_rows > 0 and config.grid_cols > 0
    required = [
        config.agbd_raster_path,
        config.tcc_raster_path,
        config.lulc_base_raster_path,
        config.lulc_target_raster_path,
    ]
    for item in required:
        if not path_exists(item):
            return False
    return True


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
