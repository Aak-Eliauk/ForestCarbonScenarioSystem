from pathlib import Path
import numpy as np
import rasterio


def construct_path(path_in):
    path = Path(path_in)
    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[3]
    path_out = project_root / path
    return path_out


def get_out_dir(path_in):
    if path_in is None or path_in.strip() == "":
        path_in = "../ForestCarbonScenarioSystem_outputs"

    path = Path(path_in)
    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[3]
    path_out = project_root / path
    return path_out


def read_grid(path_in, to_float=False):
    path = construct_path(path_in)
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    if to_float:
        data = data.astype(np.float32)
        if nodata is not None:
            data[data == np.float32(nodata)] = np.nan

    return data, profile


def read_band(path_in, band=1, to_float=False):
    path = construct_path(path_in)
    with rasterio.open(path) as src:
        data = src.read(band)
        profile = src.profile.copy()
        nodata = src.nodata

    if to_float:
        data = data.astype(np.float32)
        if nodata is not None:
            data[data == np.float32(nodata)] = np.nan

    return data, profile


def read_meta(path_in):
    path = construct_path(path_in)
    with rasterio.open(path) as src:
        data = {
            "path": path,
            "shape": (src.height, src.width),
        }
    return data


def check_rasterAndalign(items, context="栅格"):
    metadata = []
    missing = []

    for name, path_in in items:
        if path_in is None or path_in.strip() == "":
            missing.append(f"{name}: {path_in}")
        elif not construct_path(path_in).exists():
            missing.append(f"{name}: {path_in}")
        else:
            metadata.append((name, read_meta(path_in)))

    if missing:
        raise ValueError(context + "缺少必要文件：\n" + "\n".join(missing))

    if len(metadata) <= 1:
        return metadata

    reference_name, reference = metadata[0]
    problems = []
    for name, item in metadata[1:]:
        if item["shape"] != reference["shape"]:
            message = f"{name}尺寸{item['shape']}与{reference_name}尺寸{reference['shape']}不一致。"
            problems.append(message)

    if problems:
        raise ValueError(context + "空间范围不一致：\n" + "\n".join(problems))

    return metadata


def write_grid(path_in, data, reference_profile, nodata=-9999.0):
    path = Path(path_in)

    out_data = data.astype(np.float32).copy()
    out_data[np.isnan(out_data)] = nodata

    profile = reference_profile.copy()
    profile["driver"] = "GTiff"
    profile["count"] = 1
    profile["dtype"] = "float32"
    profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out_data, 1)

    return path


def deconstruct_codes(text, default_values):
    if text is None:
        values = list(default_values)
        return values

    raw_text = text.strip()
    if raw_text == "":
        values = list(default_values)
        return values

    result = []
    parts = raw_text.split(",")
    for part in parts:
        clean_part = part.strip()
        if clean_part != "":
            result.append(int(float(clean_part)))

    if not result:
        values = list(default_values)
        return values

    return result


def deconstruct_envs(text):
    result = []
    if text is None:
        return result

    lines = text.strip().splitlines()
    for line in lines:
        clean_line = line.strip()
        if clean_line == "":
            continue
        name, path = clean_line.split("=", 1)
        name = name.strip()
        path = path.strip()
        if name != "" and path != "":
            result.append((name, path))

    return result


def deconstruct_years(text):
    result = {}
    if text is None:
        return result

    lines = text.strip().splitlines()
    for line in lines:
        clean_line = line.strip()
        if clean_line == "":
            continue
        year_text, path = clean_line.split("=", 1)
        year = int(float(year_text.strip()))
        path = path.strip()
        if path != "":
            result[year] = path

    return result
