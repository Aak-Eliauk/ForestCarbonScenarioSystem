from pathlib import Path

import pandas as pd
import streamlit as st

from fcscs.config.defaults import sanitize_scenario_name
from fcscs.domain.models import ReportBundle
from fcscs.ui.app_state import get_config, get_report_bundle, get_simulation_bundle
from fcscs.ui.common import get_output_directory
from fcscs.ui.result_views import (
    export_report,
    render_detail_tables,
    render_distribution_charts,
    render_output_files,
    render_result_maps,
    render_result_overview,
)
from fcscs.ui.styles import render_page_banner


RASTER_FILE_KEYS = {
    "mean_AGBD.tif": "mean_AGBD_tif",
    "mean_AGC.tif": "mean_AGC_tif",
    "q05_AGBD.tif": "q05_AGBD_tif",
    "q95_AGBD.tif": "q95_AGBD_tif",
}


def render_results_page():
    render_page_banner("结果查看", "查看当前模拟结果，也可以加载输出目录中已经保存的历史结果。")

    current_report = get_report_bundle()
    current_bundle = get_simulation_bundle()
    history_items = _discover_history_results()

    options = []
    if current_report is not None:
        options.append(("current", "当前会话结果：" + current_report.scenario_name, None))
    for item in history_items:
        options.append(("history", item["label"], item))

    if not options:
        st.info("还没有可查看的结果。完成一次模拟后，系统会自动保存结果；已有栅格输出也会显示在这里。")
        if st.button("返回运行检查", type="primary", use_container_width=True, key="results_back_to_run"):
            st.session_state["sidebar_panel"] = "workflow"
            st.session_state["workbench_wizard_step"] = 2
            st.rerun()
        return

    option_labels = [option[1] for option in options]
    selected_label = st.selectbox("结果来源", option_labels, key="result_source_select")
    selected_index = option_labels.index(selected_label)
    source_type, _, item = options[selected_index]

    if source_type == "current":
        _render_current_result(current_report, current_bundle)
    else:
        _render_history_result(item)


def _render_current_result(report, bundle):
    st.caption("来源：当前会话内存结果")
    _render_result_tabs(report, bundle)

    st.info("运行成功后系统会自动保存结果。下方按钮用于手动重新保存或覆盖同名历史报告。")
    if st.button("重新保存当前结果", type="primary", use_container_width=True, key="save_current_report_history"):
        export_dir = get_output_directory(get_config()) / "report_exports" / sanitize_scenario_name(report.scenario_name)
        export_report(report, export_dir)
        st.success("结果已保存到：" + str(export_dir))


def _render_history_result(item):
    report = _load_history_report(item)
    if report is None:
        st.warning("历史结果读取失败。")
        return

    st.caption("来源：" + str(item["path"]))
    _render_result_tabs(report, None)


def _render_result_tabs(report, bundle):
    overview_tab, map_tab, distribution_tab, output_tab, detail_tab = st.tabs(
        ["概览", "地图预览", "不确定性分布", "输出文件", "详细表格"]
    )
    with overview_tab:
        render_result_overview(report, bundle)
    with map_tab:
        render_result_maps(report.output_files)
    with distribution_tab:
        render_distribution_charts(report)
    with output_tab:
        render_output_files(report.output_files)
    with detail_tab:
        render_detail_tables(report)


def _discover_history_results():
    output_dir = get_output_directory(get_config())
    items = []

    export_root = output_dir / "report_exports"
    if export_root.exists():
        for scenario_dir in sorted(export_root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
            if not scenario_dir.is_dir():
                continue
            if not (scenario_dir / "summary.csv").exists() and not (scenario_dir / "simulation_distribution.csv").exists():
                continue
            items.append(
                {
                    "kind": "report",
                    "scenario_name": scenario_dir.name,
                    "path": scenario_dir,
                    "label": "历史报告：" + scenario_dir.name,
                }
            )

    known_scenarios = {str(item["scenario_name"]) for item in items}
    raster_root = output_dir / "raster_predictions"
    if raster_root.exists():
        for scenario_dir in sorted(raster_root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
            if not scenario_dir.is_dir():
                continue
            if scenario_dir.name in known_scenarios:
                continue
            output_files = _find_raster_output_files(scenario_dir)
            if not output_files:
                continue
            items.append(
                {
                    "kind": "raster",
                    "scenario_name": scenario_dir.name,
                    "path": scenario_dir,
                    "label": "历史栅格：" + scenario_dir.name,
                }
            )

    return items


def _load_history_report(item):
    if item["kind"] == "report":
        return _load_report_export(item["path"], item["scenario_name"])
    return _load_raster_only_report(item["path"], item["scenario_name"])


def _load_report_export(export_dir, scenario_name):
    summary_df = _read_csv_if_exists(export_dir / "summary.csv")
    distribution_df = _read_csv_if_exists(export_dir / "simulation_distribution.csv")
    metrics_df = _read_csv_if_exists(export_dir / "metrics.csv")
    yearly_event_df = _read_csv_if_exists(export_dir / "yearly_events.csv")
    training_summary_df = _read_csv_if_exists(export_dir / "model_training_summary.csv")
    training_sample_df = _read_csv_if_exists(export_dir / "model_training_sample_preview.csv")
    output_files = _load_export_output_files(export_dir)

    if distribution_df is None:
        distribution_df = pd.DataFrame()
    if summary_df is None:
        summary_df = pd.DataFrame()

    metrics = {}
    if metrics_df is not None and not metrics_df.empty:
        for _, row in metrics_df.iterrows():
            metrics[str(row.get("指标", ""))] = row.get("数值", None)

    return ReportBundle(
        scenario_name,
        summary_df,
        distribution_df,
        metrics,
        yearly_event_df,
        training_summary_df,
        training_sample_df,
        output_files,
    )


def _load_raster_only_report(raster_dir, scenario_name):
    output_files = _find_raster_output_files(raster_dir)
    summary_df = pd.DataFrame(
        [
            {"metric": "scenario_name", "label": "情景名称", "value": scenario_name},
            {"metric": "result_type", "label": "结果类型", "value": "历史栅格输出"},
        ]
    )
    return ReportBundle(scenario_name, summary_df, pd.DataFrame(), {}, None, None, None, output_files)


def _load_export_output_files(export_dir):
    raster_csv = _read_csv_if_exists(export_dir / "raster_outputs.csv")
    output_files = {}
    if raster_csv is not None and not raster_csv.empty:
        for _, row in raster_csv.iterrows():
            key = str(row.get("文件类型", ""))
            path = str(row.get("路径", ""))
            if key and path:
                output_files[key] = path

    if output_files:
        return output_files

    raster_dir = get_output_directory(get_config()) / "raster_predictions" / sanitize_scenario_name(export_dir.name)
    return _find_raster_output_files(raster_dir)


def _find_raster_output_files(raster_dir):
    raster_dir = Path(raster_dir)
    output_files = {}
    for file_name, key in RASTER_FILE_KEYS.items():
        path = raster_dir / file_name
        if path.exists():
            output_files[key] = str(path)
    return output_files


def _read_csv_if_exists(path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None
