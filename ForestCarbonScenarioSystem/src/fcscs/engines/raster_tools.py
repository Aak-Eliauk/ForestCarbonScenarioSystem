from pathlib import Path
import numpy as np
import rasterio


def find_path(path_text):
    path = Path(str(path_text))
    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[3]
    return project_root / path


def get_out_dir(path_text):
    if path_text is None or str(path_text).strip() == "":
        path_text = "../ForestCarbonScenarioSystem_outputs"

    path = Path(str(path_text))
    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[3]
    return project_root / path


def read_grid(path_text, make_float=False):
    path = find_path(path_text)
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    if make_float:
        data = data.astype(np.float32)
        if nodata is not None:
            data[data == np.float32(nodata)] = np.nan

    return data, profile


def read_band(path_text, band=1, make_float=False):
    path = find_path(path_text)
    with rasterio.open(path) as src:
        data = src.read(int(band))
        profile = src.profile.copy()
        nodata = src.nodata

    if make_float:
        data = data.astype(np.float32)
        if nodata is not None:
            data[data == np.float32(nodata)] = np.nan

    return data, profile


def read_meta(path_text):
    path = find_path(path_text)
    with rasterio.open(path) as src:
        return {
            "path": path,
            "shape": (int(src.height), int(src.width)),
        }


def check_rasters(items, context="栅格"):
    metadata = []
    missing = []

    for name, path_text in items:
        if path_text is None or str(path_text).strip() == "":
            missing.append(f"{name}: {path_text}")
        elif not find_path(path_text).exists():
            missing.append(f"{name}: {path_text}")
        else:
            metadata.append((name, read_meta(path_text)))

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


def write_grid(path_text, data, reference_profile, nodata=-9999.0):
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)

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


def parse_codes(text, default_values):
    if text is None:
        return list(default_values)

    raw_text = str(text).strip()
    if raw_text == "":
        return list(default_values)

    result = []
    parts = raw_text.split(",")
    for part in parts:
        clean_part = part.strip()
        if clean_part != "":
            result.append(int(float(clean_part)))

    if not result:
        return list(default_values)

    return result


def parse_envs(text):
    result = []
    if text is None:
        return result

    lines = str(text).strip().splitlines()
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


def parse_years(text):
    result = {}
    if text is None:
        return result

    lines = str(text).strip().splitlines()
    for line in lines:
        clean_line = line.strip()
        if clean_line == "":
            continue
        year_text, path_text = clean_line.split("=", 1)
        year = int(float(year_text.strip()))
        path_text = path_text.strip()
        if path_text != "":
            result[year] = path_text

    return result
