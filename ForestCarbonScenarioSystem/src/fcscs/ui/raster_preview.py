from io import BytesIO
from pathlib import Path
import base64

import numpy as np
import streamlit as st


PALETTES = {
    "森林绿": [(245, 247, 232), (156, 193, 116), (54, 118, 73), (17, 66, 45)],
    "蓝绿": [(241, 249, 247), (142, 208, 196), (37, 132, 128), (21, 73, 92)],
    "地形": [(53, 94, 59), (138, 166, 96), (216, 194, 135), (130, 104, 82)],
    "热力": [(255, 247, 188), (254, 196, 79), (236, 112, 20), (153, 52, 4)],
    "紫黄": [(49, 54, 149), (116, 173, 209), (254, 224, 144), (165, 0, 38)],
    "灰度": [(248, 248, 248), (176, 176, 176), (94, 94, 94), (28, 28, 28)],
}

STRETCH_OPTIONS = {
    "自动增强 2%-98%": (2.0, 98.0),
    "温和增强 5%-95%": (5.0, 95.0),
    "完整范围 0%-100%": (0.0, 100.0),
}

PREVIEW_SIZES = {
    "标准": 760,
    "放大": 1200,
    "超大": 1800,
}

DISPLAY_MODES = {
    "适应页面": "100%",
    "放大滚动": "135%",
    "超大滚动": "170%",
}


def render_raster_preview(path_text, title, key_prefix="raster_preview"):
    path = Path(str(path_text))
    if not path.exists():
        st.warning(title + " 文件不存在")
        return

    st.subheader(title)
    c1, c2, c3 = st.columns([1.15, 1.15, 1])
    with c1:
        palette_name = st.selectbox("颜色渲染", list(PALETTES.keys()), key=key_prefix + "_palette")
    with c2:
        stretch_name = st.selectbox("数值拉伸", list(STRETCH_OPTIONS.keys()), key=key_prefix + "_stretch")
    with c3:
        preview_size_name = st.selectbox("采样精度", list(PREVIEW_SIZES.keys()), index=1, key=key_prefix + "_size")

    c4, c5 = st.columns([1.15, 1])
    with c4:
        display_mode = st.radio(
            "显示大小",
            list(DISPLAY_MODES.keys()),
            index=0,
            horizontal=True,
            key=key_prefix + "_display_mode",
        )
    with c5:
        reverse_palette = st.checkbox("反转色带", key=key_prefix + "_reverse")

    low_percentile, high_percentile = STRETCH_OPTIONS[stretch_name]
    max_size = PREVIEW_SIZES[preview_size_name]

    try:
        import rasterio

        with rasterio.open(path) as src:
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
    image, low, high = _make_color_image(
        array,
        PALETTES[palette_name],
        reverse_palette,
        low_percentile,
        high_percentile,
    )
    legend = _make_legend_image(PALETTES[palette_name], reverse_palette)

    _render_map_image(image, display_mode)
    _render_legend(legend, low, high)
    _render_raster_stats(array, width, height, crs_text, out_width, out_height)
    with st.expander("文件路径"):
        st.code(str(path), language="text")


def _clean_array(data, nodata):
    array = data.astype(np.float32)
    if nodata is not None:
        array[array == np.float32(nodata)] = np.nan
    return array


def _make_color_image(array, palette, reverse_palette, low_percentile, high_percentile):
    if np.isnan(array).all():
        empty = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
        return empty, 0.0, 0.0

    low = float(np.nanpercentile(array, low_percentile))
    high = float(np.nanpercentile(array, high_percentile))
    if high <= low:
        high = low + 1.0

    norm = (array - low) / (high - low)
    norm = np.clip(norm, 0.0, 1.0)
    nan_mask = np.isnan(norm)
    norm[nan_mask] = 0.0

    colors = _interpolate_palette(norm, palette, reverse_palette)
    colors[nan_mask] = np.array([238, 238, 238], dtype=np.uint8)
    return colors, low, high


def _interpolate_palette(norm, palette, reverse_palette):
    points = np.array(palette, dtype=np.float32)
    if reverse_palette:
        points = points[::-1]

    segments = len(points) - 1
    scaled = norm * segments
    lower_index = np.floor(scaled).astype(np.int32)
    lower_index = np.clip(lower_index, 0, segments - 1)
    fraction = scaled - lower_index

    lower = points[lower_index]
    upper = points[lower_index + 1]
    rgb = lower + (upper - lower) * fraction[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _make_legend_image(palette, reverse_palette):
    gradient = np.linspace(0.0, 1.0, 320, dtype=np.float32)
    gradient = np.tile(gradient, (18, 1))
    return _interpolate_palette(gradient, palette, reverse_palette)


def _render_map_image(image, display_mode):
    image_width = DISPLAY_MODES.get(display_mode, "100%")
    if image_width == "100%":
        st.image(image, use_container_width=True)
        return

    try:
        from PIL import Image

        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        st.image(image, use_container_width=True)
        return

    st.markdown(
        f"""
        <div class="raster-zoom-frame">
            <img src="data:image/png;base64,{encoded}" style="width: {image_width}; max-width: none;" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_legend(legend, low, high):
    c1, c2, c3 = st.columns([1, 5, 1])
    c1.caption(f"{low:.3f}")
    c2.image(legend, use_container_width=True)
    c3.caption(f"{high:.3f}")


def _render_raster_stats(array, width, height, crs_text, preview_width, preview_height):
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
    st.caption(f"坐标参考：{crs_text}；当前预览采样：{preview_width} x {preview_height}")
