from pathlib import Path

import numpy as np
import streamlit as st


def render_raster_preview(path_text, title):
    path = Path(str(path_text))
    if not path.exists():
        st.warning(title + " 文件不存在")
        return

    try:
        import rasterio

        with rasterio.open(path) as src:
            max_size = 700
            scale = max(src.width / max_size, src.height / max_size, 1)
            out_width = max(1, int(src.width / scale))
            out_height = max(1, int(src.height / scale))
            data = src.read(1, out_shape=(out_height, out_width))
            width = src.width
            height = src.height
            crs_text = str(src.crs) if src.crs is not None else "未设置"
            nodata = src.nodata
    except Exception as error:
        st.warning(title + " 预览失败：" + str(error))
        return

    array = _clean_array(data, nodata)
    image = _make_color_image(array)
    st.subheader(title)
    st.image(image, use_container_width=True)
    _render_raster_stats(array, width, height, crs_text)
    with st.expander("文件路径"):
        st.code(str(path), language="text")


def _clean_array(data, nodata):
    array = data.astype(np.float32)
    if nodata is not None:
        array[array == np.float32(nodata)] = np.nan
    return array


def _make_color_image(array):
    if np.isnan(array).all():
        return np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)

    low = float(np.nanpercentile(array, 2))
    high = float(np.nanpercentile(array, 98))
    if high <= low:
        high = low + 1.0

    norm = (array - low) / (high - low)
    norm = np.clip(norm, 0.0, 1.0)
    norm[np.isnan(norm)] = 0.0

    red = (42 + norm * 188).astype(np.uint8)
    green = (85 + norm * 132).astype(np.uint8)
    blue = (60 + (1.0 - norm) * 82).astype(np.uint8)
    return np.dstack([red, green, blue])


def _render_raster_stats(array, width, height, crs_text):
    valid_mask = ~np.isnan(array)
    valid_count = int(valid_mask.sum())
    total_count = int(array.size)
    valid_ratio = 0.0
    if total_count > 0:
        valid_ratio = valid_count / total_count * 100.0

    if valid_count == 0:
        mean_text = "无有效值"
        range_text = "无有效值"
    else:
        mean_text = f"{float(np.nanmean(array)):.3f}"
        min_text = f"{float(np.nanmin(array)):.3f}"
        max_text = f"{float(np.nanmax(array)):.3f}"
        range_text = min_text + " - " + max_text

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("尺寸", f"{width} x {height}")
    c2.metric("均值", mean_text)
    c3.metric("范围", range_text)
    c4.metric("有效像元", f"{valid_ratio:.1f}%")
    st.caption("坐标参考：" + crs_text)
