import html

import pandas as pd
import streamlit as st

from fcscs.ui.raster_preview import render_raster_preview


CHART_LABELS = {
    "mean_agbd_per_ha": "平均 AGBD",
    "mean_agc_per_ha": "平均 AGC",
    "total_agbd": "总量 AGBD",
    "total_agc": "总量 AGC",
}

OUTPUT_FILE_NAMES = {
    "mean_AGBD_tif": "平均 AGBD 栅格",
    "mean_AGC_tif": "平均 AGC 栅格",
    "q05_AGBD_tif": "AGBD 5% 分位栅格",
    "q95_AGBD_tif": "AGBD 95% 分位栅格",
}

OVERVIEW_METRIC_LABELS = {
    "mean_agbd_per_ha": "平均 AGBD",
    "mean_agc_per_ha": "平均 AGC",
    "total_agbd_mean": "总 AGBD",
    "total_agc_mean": "总 AGC",
    "mean_reduction_per_ha": "平均损失强度",
    "mean_model_r2": "模型 R2",
    "mean_model_mae": "模型 MAE",
}

CHINESE_OVERVIEW_KEYS = ["平均AGBD", "平均AGC", "总AGBD", "总AGC", "平均损失强度", "模型R2", "模型MAE"]


def render_result_overview(report, bundle=None):
    metrics = _build_metric_items(report, bundle)
    if metrics:
        columns = st.columns(min(4, len(metrics)))
        for index, (label, value) in enumerate(metrics):
            columns[index % len(columns)].metric(label, _format_overview_value(value))

    if report.summary_df is not None and not report.summary_df.empty:
        st.subheader("结果摘要")
        st.dataframe(report.summary_df, use_container_width=True, hide_index=True)


def render_result_maps(output_files):
    if not output_files:
        st.info("当前运行没有生成 GeoTIFF 栅格输出。")
        return

    preview_items = []
    for key in ["mean_AGBD_tif", "mean_AGC_tif", "q05_AGBD_tif", "q95_AGBD_tif"]:
        if key in output_files:
            preview_items.append((key, output_files[key]))

    if not preview_items:
        st.info("当前输出文件中没有可预览的 GeoTIFF 栅格。")
        return

    selected_label = st.selectbox(
        "预览图层",
        [OUTPUT_FILE_NAMES.get(key, key) for key, _ in preview_items],
        key="result_preview_layer",
    )
    selected_index = [OUTPUT_FILE_NAMES.get(key, key) for key, _ in preview_items].index(selected_label)
    selected_key, selected_path = preview_items[selected_index]
    render_raster_preview(
        selected_path,
        OUTPUT_FILE_NAMES.get(selected_key, selected_key),
        key_prefix="result_preview_" + selected_key,
    )


def render_distribution_charts(report):
    if report.total_distribution_df is None or report.total_distribution_df.empty:
        st.info("暂无蒙特卡洛分布结果。")
        return

    _render_uncertainty_cards(report.total_distribution_df)

    chart_df = report.total_distribution_df
    if "simulation" in chart_df.columns:
        chart_df = chart_df.set_index("simulation")
    elif "sim_id" in chart_df.columns:
        chart_df = chart_df.set_index("sim_id")

    chart_specs = []
    mean_columns = ["mean_agbd_per_ha", "mean_agc_per_ha"]
    if _has_columns(chart_df, mean_columns):
        chart_specs.append(("平均碳储量变化", mean_columns))

    total_columns = ["total_agbd", "total_agc"]
    if _has_columns(chart_df, total_columns):
        chart_specs.append(("总量碳储量变化", total_columns))

    if chart_specs:
        st.subheader("分布图表")
        columns = st.columns(min(2, len(chart_specs)))
        for index, (title, column_names) in enumerate(chart_specs):
            with columns[index % len(columns)]:
                st.markdown(f"**{title}**")
                st.line_chart(_label_chart_columns(chart_df, column_names), use_container_width=True)
    else:
        st.info("当前分布表缺少可绘图的数值列。")

    _render_distribution_summary(report.total_distribution_df)

    with st.expander("每次模拟明细"):
        st.dataframe(report.total_distribution_df, use_container_width=True, hide_index=True)


def _render_uncertainty_cards(distribution_df):
    numeric_columns = [column for column in ["total_agbd", "total_agc", "mean_agbd_per_ha", "mean_agc_per_ha"] if column in distribution_df.columns]
    if not numeric_columns:
        return

    total_agbd = _series_or_none(distribution_df, "total_agbd")
    total_agc = _series_or_none(distribution_df, "total_agc")
    mean_agbd = _series_or_none(distribution_df, "mean_agbd_per_ha")

    cards = [
        ("MC", "实际模拟次数", _format_number(len(distribution_df), 0)),
    ]
    if total_agbd is not None:
        cards.append(("AGBD", "总 AGBD 均值", _format_number(total_agbd.mean(), 2)))
        cards.append(("P5-95", "总 AGBD 5%-95%区间", _format_range(total_agbd)))
    if total_agc is not None:
        cards.append(("AGC", "总 AGC 均值", _format_number(total_agc.mean(), 2)))
    elif mean_agbd is not None:
        cards.append(("MEAN", "平均 AGBD 均值", _format_number(mean_agbd.mean(), 3)))

    html_cards = []
    for icon, label, value in cards[:4]:
        html_cards.append(
            '<div class="uncertainty-card">'
            '<div class="uncertainty-icon">{icon}</div>'
            "<div>"
            '<div class="uncertainty-label">{label}</div>'
            '<div class="uncertainty-value">{value}</div>'
            "</div>"
            "</div>".format(
                icon=html.escape(str(icon)),
                label=html.escape(str(label)),
                value=html.escape(str(value)),
            )
        )

    st.markdown('<div class="uncertainty-card-grid">' + "".join(html_cards) + "</div>", unsafe_allow_html=True)


def render_output_files(output_files):
    if not output_files:
        st.info("暂无输出文件。")
        return

    rows = []
    for key, path in output_files.items():
        rows.append({"文件": OUTPUT_FILE_NAMES.get(key, key), "类型": key, "路径": str(path)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_detail_tables(report):
    if report.yearly_event_df is not None and not report.yearly_event_df.empty:
        with st.expander("年度事件统计"):
            st.dataframe(report.yearly_event_df, use_container_width=True, hide_index=True)

    if report.training_summary_df is not None and not report.training_summary_df.empty:
        with st.expander("模型训练摘要"):
            st.dataframe(report.training_summary_df, use_container_width=True, hide_index=True)


def export_report(report, export_dir):
    export_dir.mkdir(parents=True, exist_ok=True)

    report.summary_df.to_csv(export_dir / "summary.csv", index=False, encoding="utf-8-sig")
    report.total_distribution_df.to_csv(export_dir / "simulation_distribution.csv", index=False, encoding="utf-8-sig")

    metric_rows = []
    for key, value in report.metrics.items():
        metric_rows.append({"指标": key, "数值": value})
    pd.DataFrame(metric_rows).to_csv(export_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    if report.yearly_event_df is not None and not report.yearly_event_df.empty:
        report.yearly_event_df.to_csv(export_dir / "yearly_events.csv", index=False, encoding="utf-8-sig")

    if report.training_summary_df is not None and not report.training_summary_df.empty:
        report.training_summary_df.to_csv(export_dir / "model_training_summary.csv", index=False, encoding="utf-8-sig")

    if report.training_sample_df is not None and not report.training_sample_df.empty:
        report.training_sample_df.to_csv(export_dir / "model_training_sample_preview.csv", index=False, encoding="utf-8-sig")

    if report.output_files:
        raster_rows = []
        for key, value in report.output_files.items():
            raster_rows.append({"文件类型": key, "路径": str(value)})
        pd.DataFrame(raster_rows).to_csv(export_dir / "raster_outputs.csv", index=False, encoding="utf-8-sig")


def _build_metric_items(report, bundle):
    if bundle is not None:
        return [
            ("平均 AGBD", round(bundle.summary["mean_agbd_per_ha"], 3)),
            ("平均 AGC", round(bundle.summary["mean_agc_per_ha"], 3)),
            ("模型 R2", round(bundle.summary["mean_model_r2"], 4)),
            ("模拟次数", bundle.summary["n_simulations"]),
        ]

    if not report.metrics:
        return []

    items = []
    for key in CHINESE_OVERVIEW_KEYS:
        if key in report.metrics:
            items.append((key, report.metrics[key]))
    if items:
        return items

    for key, label in OVERVIEW_METRIC_LABELS.items():
        if key in report.metrics:
            items.append((label, report.metrics[key]))
    return items


def _format_overview_value(value):
    if value is None:
        return "无数据"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if pd.isna(number):
        return "无数据"
    return f"{number:,.2f}"


def _render_distribution_summary(distribution_df):
    numeric_columns = [
        column
        for column in ["mean_agbd_per_ha", "mean_agc_per_ha", "total_agbd", "total_agc"]
        if column in distribution_df.columns
    ]
    if not numeric_columns:
        return

    summary = distribution_df[numeric_columns].describe(percentiles=[0.05, 0.5, 0.95]).T.reset_index()
    summary = summary.rename(
        columns={
            "index": "指标",
            "mean": "均值",
            "std": "标准差",
            "5%": "5%分位",
            "50%": "中位数",
            "95%": "95%分位",
        }
    )
    display_columns = ["指标", "均值", "标准差", "5%分位", "中位数", "95%分位"]
    st.subheader("分布摘要")
    st.dataframe(summary[display_columns], use_container_width=True, hide_index=True)


def _label_chart_columns(frame, column_names):
    chart_frame = frame[column_names].copy()
    chart_frame = chart_frame.rename(columns=CHART_LABELS)
    return chart_frame


def _series_or_none(frame, column_name):
    if column_name not in frame.columns:
        return None
    series = pd.to_numeric(frame[column_name], errors="coerce").dropna()
    if series.empty:
        return None
    return series


def _format_range(series):
    low = series.quantile(0.05)
    high = series.quantile(0.95)
    return _format_number(low, 2) + " - " + _format_number(high, 2)


def _format_number(value, digits=2):
    if pd.isna(value):
        return "无数据"
    if digits <= 0:
        return f"{float(value):,.0f}"
    return f"{float(value):,.{digits}f}"


def _has_columns(frame, names):
    for name in names:
        if name not in frame.columns:
            return False
    return True
