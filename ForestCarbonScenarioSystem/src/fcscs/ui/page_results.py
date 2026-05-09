from pathlib import Path

import pandas as pd
import streamlit as st

from fcscs.domain.models import ReportBundle
from fcscs.ui.app_state import get_config
from fcscs.ui.common import get_output_directory
from fcscs.ui.result_views import (
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

    history_items = _discover_history_results()
    manual_item = _render_manual_result_picker()

    options = []
    if manual_item is not None:
        options.append(("history", "手动打开：" + manual_item["label"], manual_item))
    for item in history_items:
        options.append(("history", item["label"], item))

    if not options:
        st.info("还没有可查看的结果。完成一次模拟后，系统会自动保存结果；已有栅格输出也会显示在这里。")
        if st.button("返回启动运行", type="primary", use_container_width=True, key="results_back_to_run"):
            st.session_state["sidebar_panel"] = "workflow"
            st.session_state["workbench_wizard_step"] = 2
            st.rerun()
        return

    option_labels = [option[1] for option in options]
    selected_label = st.selectbox("结果来源", option_labels, key="result_source_select")
    selected_index = option_labels.index(selected_label)
    _, _, item = options[selected_index]
    _render_history_result(item)


def _render_manual_result_picker():
    with st.expander("打开其他结果文件夹", expanded=False):
        current_path = st.session_state.get("manual_result_folder", "")
        c_path, c_button = st.columns([5, 1])
        with c_path:
            folder_text = st.text_input("结果文件夹路径", value=current_path, key="manual_result_folder_input")
        with c_button:
            st.write("")
            if st.button("选择", use_container_width=True, key="manual_result_folder_browse"):
                picked_path = _open_windows_folder_dialog("选择结果文件夹")
                if picked_path:
                    st.session_state["manual_result_folder"] = picked_path
                    st.rerun()

        if st.button("打开该文件夹", use_container_width=True, key="manual_result_folder_open"):
            st.session_state["manual_result_folder"] = folder_text
            st.rerun()

    manual_path = st.session_state.get("manual_result_folder", "")
    if not manual_path:
        return None
    return _build_manual_result_item(manual_path)


def _render_history_result(item):
    report = _load_history_report(item)
    if report is None:
        st.warning("历史结果读取失败。")
        return

    st.caption("来源：" + str(item["path"]))
    _render_result_tabs(report)


def _render_result_tabs(report):
    overview_tab, map_tab, distribution_tab, output_tab, detail_tab = st.tabs(
        ["概览", "地图预览", "不确定性分布", "输出文件", "详细表格"]
    )
    with overview_tab:
        render_result_overview(report)
    with map_tab:
        render_result_maps(report.output_files)
    with distribution_tab:
        render_distribution_charts(report)
    with output_tab:
        render_output_files(report.output_files)
    with detail_tab:
        render_detail_tables(report)


def _discover_history_results():
    output_dir = get_output_directory(get_config(), create=False)
    items = []

    if output_dir.exists():
        for batch_dir in sorted(output_dir.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
            if not batch_dir.is_dir():
                continue
            report_dir = batch_dir / "report_exports"
            if not report_dir.exists():
                continue
            if not (report_dir / "summary.csv").exists() and not (report_dir / "simulation_distribution.csv").exists():
                continue
            items.append(
                {
                    "kind": "report",
                    "scenario_name": _read_summary_value(report_dir, "scenario_name", batch_dir.name),
                    "batch_name": batch_dir.name,
                    "path": report_dir,
                    "label": "历史批次：" + batch_dir.name,
                }
            )

    known_batches = {str(item["batch_name"]) for item in items}
    if output_dir.exists():
        for batch_dir in sorted(output_dir.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
            if not batch_dir.is_dir():
                continue
            if batch_dir.name in known_batches:
                continue
            raster_dir = batch_dir / "raster_predictions"
            output_files = _find_raster_output_files(raster_dir)
            if not output_files:
                continue
            items.append(
                {
                    "kind": "raster",
                    "scenario_name": batch_dir.name,
                    "batch_name": batch_dir.name,
                    "path": raster_dir,
                    "label": "历史栅格：" + batch_dir.name,
                }
            )

    return items


def _build_manual_result_item(folder_text):
    path = Path(str(folder_text).strip())
    if not path.exists() or not path.is_dir():
        st.warning("手动选择的结果文件夹不存在。")
        return None

    report_dir = path
    if (path / "report_exports").exists():
        report_dir = path / "report_exports"

    if (report_dir / "summary.csv").exists() or (report_dir / "simulation_distribution.csv").exists():
        batch_name = path.name
        if report_dir.name == "report_exports":
            batch_name = report_dir.parent.name
        return {
            "kind": "report",
            "scenario_name": _read_summary_value(report_dir, "scenario_name", batch_name),
            "batch_name": batch_name,
            "path": report_dir,
            "label": batch_name,
        }

    raster_dir = path
    if (path / "raster_predictions").exists():
        raster_dir = path / "raster_predictions"
    output_files = _find_raster_output_files(raster_dir)
    if output_files:
        return {
            "kind": "raster",
            "scenario_name": path.name,
            "batch_name": path.name,
            "path": raster_dir,
            "label": path.name,
        }

    st.warning("该文件夹中没有找到 summary.csv 或 GeoTIFF 结果文件。")
    return None


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

    raster_dir = export_dir.parent / "raster_predictions"
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


def _read_summary_value(export_dir, metric_name, default_value):
    summary_df = _read_csv_if_exists(export_dir / "summary.csv")
    if summary_df is None or summary_df.empty:
        return default_value
    for _, row in summary_df.iterrows():
        if str(row.get("metric", "")) == metric_name:
            value = row.get("value", default_value)
            if value is not None and str(value).strip() != "":
                return str(value)
    return default_value


def _open_windows_folder_dialog(title):
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        picked_path = filedialog.askdirectory(title=title)
        root.destroy()
        return picked_path
    except Exception:
        return ""
