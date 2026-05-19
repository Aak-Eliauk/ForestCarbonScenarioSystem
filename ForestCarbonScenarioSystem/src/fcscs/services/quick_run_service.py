from fcscs.config.defaults import name_clean
from fcscs.engines.raster_tools import (
    deconstruct_envs,
    deconstruct_years,
    construct_path,
    get_out_dir,
    check_rasterAndalign,
)


def build_quick_config(config, quick_size):
    quick_config = build_quick_raster_config(config, quick_size)
    return quick_config


def build_quick_raster_config(config, quick_size):
    import numpy as np
    import rasterio

    required_items = [
        ("AGBD", config.agbd_raster_path),
        ("TCC", config.tcc_raster_path),
        ("基准年LULC", config.lulc_base_raster_path),
        ("目标年LULC", config.lulc_target_raster_path),
        ("Drivers", config.drivers_raster_path),
        ("保护区", config.reserve_raster_path),
    ]
    optional_items = []
    for name, path_in in deconstruct_envs(config.env_raster_paths):
        if construct_path(path_in).exists():
            optional_items.append(("环境因子-" + name, path_in))
    check_rasterAndalign(required_items + optional_items, "快速测试栅格")

    batch_dir = name_clean(config.batch_name + "_快速测试", default="运行批次")
    out_dir = get_out_dir(config.output_dir) / batch_dir / "quick_test_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    agbd_path = construct_path(config.agbd_raster_path)
    lulc_base_path = construct_path(config.lulc_base_raster_path)
    lulc_target_path = construct_path(config.lulc_target_raster_path)
    drivers_path = construct_path(config.drivers_raster_path)
    reserve_path = construct_path(config.reserve_raster_path)

    with rasterio.open(agbd_path) as src:
        rows = src.height
        cols = src.width

    size = min(int(quick_size), rows, cols)
    window = _pick_quick_window(lulc_base_path, lulc_target_path, drivers_path, reserve_path, config, size, np, rasterio)

    clipped_paths = {}
    clipped_paths["agbd"] = _clip_one_raster(config.agbd_raster_path, out_dir / "agbd.tif", window, rasterio)
    clipped_paths["tcc"] = _clip_one_raster(config.tcc_raster_path, out_dir / "tcc.tif", window, rasterio)
    clipped_paths["lulc_base"] = _clip_one_raster(config.lulc_base_raster_path, out_dir / "lulc_base.tif", window, rasterio)
    clipped_paths["lulc_target"] = _clip_one_raster(config.lulc_target_raster_path, out_dir / "lulc_target.tif", window, rasterio)

    clipped_paths["drivers"] = _clip_all_bands_raster(config.drivers_raster_path, out_dir / "drivers.tif", window, rasterio)
    clipped_paths["reserve"] = _clip_one_raster(config.reserve_raster_path, out_dir / "reserve.tif", window, rasterio)

    env_text = _clip_env_rasters(config.env_raster_paths, out_dir, window, rasterio)
    history_agbd_text = _clip_year_rasters(config.history_agbd_paths, out_dir / "history_agbd", window, rasterio)
    history_tcc_text = _clip_year_rasters(config.history_tcc_paths, out_dir / "history_tcc", window, rasterio)
    history_lulc_text = _clip_year_rasters(config.history_lulc_paths, out_dir / "history_lulc", window, rasterio)

    quick_config = config.copy()
    quick_config.scenario_name = name_clean(config.scenario_name + "_quick_test")
    quick_config.batch_name = batch_dir
    quick_config.agbd_raster_path = str(clipped_paths["agbd"])
    quick_config.tcc_raster_path = str(clipped_paths["tcc"])
    quick_config.lulc_base_raster_path = str(clipped_paths["lulc_base"])
    quick_config.lulc_target_raster_path = str(clipped_paths["lulc_target"])
    quick_config.drivers_raster_path = str(clipped_paths["drivers"])
    quick_config.reserve_raster_path = str(clipped_paths["reserve"])
    quick_config.env_raster_paths = env_text
    quick_config.history_agbd_paths = history_agbd_text
    quick_config.history_tcc_paths = history_tcc_text
    quick_config.history_lulc_paths = history_lulc_text
    quick_config.mc_n_simulations = min(config.mc_n_simulations, 3)
    quick_config.ml_sample_count = min(config.ml_sample_count, 1200)
    quick_config.logging_library_patch_count = min(config.logging_library_patch_count, 100)
    return quick_config


def _pick_quick_window(lulc_base_path, lulc_target_path, drivers_path, reserve_path, config, size, np, rasterio):
    with rasterio.open(lulc_base_path) as base_src:
        rows = base_src.height
        cols = base_src.width

    default_row = max(0, (rows - size) // 2)
    default_col = max(0, (cols - size) // 2)
    default_window = rasterio.windows.Window(default_col, default_row, size, size)

    forest_codes = _parse_simple_codes(config.forest_lulc_codes, [1, 2, 3, 4, 5])
    urban_codes = _parse_simple_codes(config.urban_lulc_codes, [8, 9])
    best_score = -1
    best_window = default_window

    with rasterio.open(lulc_base_path) as base_src, rasterio.open(lulc_target_path) as target_src, rasterio.open(drivers_path) as driver_src:
        with rasterio.open(reserve_path) as reserve_src:
            step = max(size // 2, 1)
            row = 0
            while row <= rows - size:
                col = 0
                while col <= cols - size:
                    window = rasterio.windows.Window(col, row, size, size)
                    base = base_src.read(1, window=window)
                    target = target_src.read(1, window=window)
                    drivers = driver_src.read(1, window=window)
                    reserve = reserve_src.read(1, window=window)
                    reserve_mask = reserve == config.reserve_value

                    forest_base = np.isin(base, forest_codes)
                    forest_target = np.isin(target, forest_codes)
                    urban_target = np.isin(target, urban_codes)
                    logging_count = int(((drivers == config.logging_driver_value) & forest_target & (~reserve_mask)).sum())
                    conv_count = int((forest_base & urban_target & (~reserve_mask)).sum())
                    score = logging_count + conv_count * 10
                    if score > best_score:
                        best_score = score
                        best_window = window
                    col = col + step
                row = row + step

    return best_window


def _clip_one_raster(source_path, output_path, raster_window, rasterio):
    source_path = construct_path(source_path)
    with rasterio.open(source_path) as src:
        data = src.read(1, window=raster_window)
        profile = src.profile.copy()
        profile.update({"height": data.shape[0], "width": data.shape[1], "transform": src.window_transform(raster_window)})
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)
    return output_path


def _clip_all_bands_raster(source_path, output_path, raster_window, rasterio):
    source_path = construct_path(source_path)
    with rasterio.open(source_path) as src:
        data = src.read(window=raster_window)
        profile = src.profile.copy()
        profile.update({"height": data.shape[1], "width": data.shape[2], "transform": src.window_transform(raster_window)})
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
    return output_path


def _clip_env_rasters(env_text, out_dir, raster_window, rasterio):
    lines = []
    env_items = deconstruct_envs(env_text)
    for name, path_in in env_items:
        if not construct_path(path_in).exists():
            continue
        output_path = out_dir / ("env_" + _safe_file_name(name) + ".tif")
        _clip_one_raster(path_in, output_path, raster_window, rasterio)
        lines.append(name + "=" + str(output_path))
    text = "\n".join(lines)
    return text


def _clip_year_rasters(year_text, out_dir, raster_window, rasterio):
    lines = []
    out_dir.mkdir(parents=True, exist_ok=True)
    year_items = deconstruct_years(year_text)
    for year in sorted(year_items.keys()):
        path_in = year_items[year]
        if not construct_path(path_in).exists():
            continue
        output_path = out_dir / (str(year) + ".tif")
        _clip_one_raster(path_in, output_path, raster_window, rasterio)
        lines.append(str(year) + "=" + str(output_path))
    text = "\n".join(lines)
    return text


def _safe_file_name(name):
    result = []
    for char in name:
        if char.isalnum() or char in ["_", "-"]:
            result.append(char)
        else:
            result.append("_")
    name_out = "".join(result) or "env"
    return name_out


def _parse_simple_codes(text, default_values):
    result = []
    for part in text.split(","):
        clean = part.strip()
        if clean:
            result.append(int(float(clean)))
    if result:
        return result
    values = list(default_values)
    return values
