from fcscs.engines.raster_tools import parse_env_raster_paths, path_exists, resolve_input_path, resolve_output_dir


def build_quick_config(config, quick_size):
    if not config.use_raster_data:
        quick_config = config.copy()
        quick_config.scenario_name = config.scenario_name + "_quick_test"
        quick_config.grid_rows = min(config.grid_rows, 96)
        quick_config.grid_cols = min(config.grid_cols, 96)
        quick_config.mc_n_simulations = min(config.mc_n_simulations, 3)
        quick_config.ml_sample_count = min(config.ml_sample_count, 1200)
        return quick_config

    return build_quick_raster_config(config, quick_size)


def build_quick_raster_config(config, quick_size):
    import numpy as np
    import rasterio

    required_paths = [
        config.agbd_raster_path,
        config.tcc_raster_path,
        config.lulc_base_raster_path,
        config.lulc_target_raster_path,
    ]
    for item in required_paths:
        if not path_exists(item):
            raise ValueError("缺少必要栅格：" + str(item))

    output_dir = resolve_output_dir(config.output_dir) / "quick_test_inputs" / config.scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)

    agbd_path = resolve_input_path(config.agbd_raster_path)
    lulc_base_path = resolve_input_path(config.lulc_base_raster_path)
    lulc_target_path = resolve_input_path(config.lulc_target_raster_path)
    drivers_path = resolve_input_path(config.drivers_raster_path)
    reserve_path = resolve_input_path(config.reserve_raster_path)

    with rasterio.open(agbd_path) as src:
        rows = src.height
        cols = src.width

    size = min(int(quick_size), rows, cols)
    window = _pick_quick_window(lulc_base_path, lulc_target_path, drivers_path, reserve_path, config, size, np, rasterio)

    clipped_paths = {}
    clipped_paths["agbd"] = _clip_one_raster(config.agbd_raster_path, output_dir / "agbd.tif", window, rasterio)
    clipped_paths["tcc"] = _clip_one_raster(config.tcc_raster_path, output_dir / "tcc.tif", window, rasterio)
    clipped_paths["lulc_base"] = _clip_one_raster(config.lulc_base_raster_path, output_dir / "lulc_base.tif", window, rasterio)
    clipped_paths["lulc_target"] = _clip_one_raster(config.lulc_target_raster_path, output_dir / "lulc_target.tif", window, rasterio)

    if path_exists(config.drivers_raster_path):
        clipped_paths["drivers"] = _clip_one_raster(config.drivers_raster_path, output_dir / "drivers.tif", window, rasterio)
    else:
        clipped_paths["drivers"] = config.drivers_raster_path

    if path_exists(config.reserve_raster_path):
        clipped_paths["reserve"] = _clip_one_raster(config.reserve_raster_path, output_dir / "reserve.tif", window, rasterio)
    else:
        clipped_paths["reserve"] = config.reserve_raster_path

    env_text = _clip_env_rasters(config.env_raster_paths, output_dir, window, rasterio)

    quick_config = config.copy()
    quick_config.scenario_name = config.scenario_name + "_quick_test"
    quick_config.agbd_raster_path = str(clipped_paths["agbd"])
    quick_config.tcc_raster_path = str(clipped_paths["tcc"])
    quick_config.lulc_base_raster_path = str(clipped_paths["lulc_base"])
    quick_config.lulc_target_raster_path = str(clipped_paths["lulc_target"])
    quick_config.drivers_raster_path = str(clipped_paths["drivers"])
    quick_config.reserve_raster_path = str(clipped_paths["reserve"])
    quick_config.env_raster_paths = env_text
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

    if not path_exists(drivers_path):
        return default_window

    forest_codes = _parse_simple_codes(config.forest_lulc_codes, [1, 2, 3, 4, 5])
    urban_codes = _parse_simple_codes(config.urban_lulc_codes, [8, 9])
    best_score = -1
    best_window = default_window

    with rasterio.open(lulc_base_path) as base_src, rasterio.open(lulc_target_path) as target_src, rasterio.open(drivers_path) as driver_src:
        reserve_src = None
        if path_exists(reserve_path):
            reserve_src = rasterio.open(reserve_path)

        step = max(size // 2, 1)
        row = 0
        while row <= rows - size:
            col = 0
            while col <= cols - size:
                window = rasterio.windows.Window(col, row, size, size)
                base = base_src.read(1, window=window)
                target = target_src.read(1, window=window)
                drivers = driver_src.read(1, window=window)
                reserve_mask = np.zeros(base.shape, dtype=bool)
                if reserve_src is not None:
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

        if reserve_src is not None:
            reserve_src.close()

    return best_window


def _clip_one_raster(source_path, output_path, raster_window, rasterio):
    source_path = resolve_input_path(source_path)
    with rasterio.open(source_path) as src:
        data = src.read(1, window=raster_window)
        profile = src.profile.copy()
        profile.update({"height": data.shape[0], "width": data.shape[1], "transform": src.window_transform(raster_window)})
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)
    return output_path


def _clip_env_rasters(env_text, output_dir, raster_window, rasterio):
    lines = []
    env_items = parse_env_raster_paths(env_text)
    for name, path_text in env_items:
        if not path_exists(path_text):
            continue
        output_path = output_dir / ("env_" + _safe_file_name(name) + ".tif")
        _clip_one_raster(path_text, output_path, raster_window, rasterio)
        lines.append(name + "=" + str(output_path))
    return "\n".join(lines)


def _safe_file_name(name):
    result = []
    for char in str(name):
        if char.isalnum() or char in ["_", "-"]:
            result.append(char)
        else:
            result.append("_")
    return "".join(result) or "env"


def _parse_simple_codes(text, default_values):
    result = []
    for part in str(text).split(","):
        clean = part.strip()
        if clean:
            result.append(int(float(clean)))
    if result:
        return result
    return list(default_values)
