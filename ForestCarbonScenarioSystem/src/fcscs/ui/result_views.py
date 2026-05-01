import pandas as pd
import streamlit as st

from fcscs.ui.raster_preview import render_raster_preview


OUTPUT_FILE_NAMES = {
    "mean_AGBD_tif": "平均 AGBD 栅格",
    "mean_AGC_tif": "平均 AGC 栅格",
    "q05_AGBD_tif": "AGBD 5% 分位栅格",
    "q95_AGBD_tif": "AGBD 95% 分位栅格",
}


def render_result_overview(report, bundle=None):
    metrics = _build_metric_items(report, bundle)
    if metrics:
        columns = st.columns(min(4, len(metrics)))
        for index, (label, value) in enumerate(metrics):
            columns[index % len(columns)].metric(label, value)

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

    chart_df = report.total_distribution_df
    if "sim_id" in chart_df.columns:
        chart_df = chart_df.set_index("sim_id")

    c1, c2 = st.columns(2)
    with c1:
        mean_columns = ["mean_agbd_per_ha", "mean_agc_per_ha"]
        if _has_columns(chart_df, mean_columns):
            st.caption("平均 AGBD / AGC")
            st.line_chart(chart_df[mean_columns])
    with c2:
        total_columns = ["total_agbd", "total_agc"]
        if _has_columns(chart_df, total_columns):
            st.caption("总量 AGBD / AGC")
            st.line_chart(chart_df[total_columns])

    _render_distribution_summary(report.total_distribution_df)

    with st.expander("每次模拟明细"):
        st.dataframe(report.total_distribution_df, use_container_width=True, hide_index=True)


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
    for key, value in report.metrics.items():
        items.append((key, value))
    return items


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


def _has_columns(frame, names):
    for name in names:
        if name not in frame.columns:
            return False
    return True
