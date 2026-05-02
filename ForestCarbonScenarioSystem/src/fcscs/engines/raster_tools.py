from pathlib import Path

import numpy as np


def rasterio_is_available():
    try:
        import rasterio  # noqa: F401
    except Exception:
        return False
    return True


def path_has_value(path_text):
    if path_text is None:
        return False
    if str(path_text).strip() == "":
        return False
    return True


def path_exists(path_text):
    if not path_has_value(path_text):
        return False
    return resolve_input_path(path_text).exists()


def resolve_input_path(path_text):
    path = Path(str(path_text))
    if path.exists():
        return path

    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[3]
    project_path = project_root / path
    if project_path.exists():
        return project_path

    outer_path = project_root.parent / path
    if outer_path.exists():
        return outer_path

    return path


def resolve_output_dir(path_text):
    if not path_has_value(path_text):
        path_text = "../ForestCarbonScenarioSystem_outputs"

    path = Path(str(path_text))
    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[3]
    return project_root / path


def read_raster(path_text, make_float=False):
    import rasterio

    path = resolve_input_path(path_text)
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    if make_float:
        data = data.astype(np.float32)
        if nodata is not None:
            data[data == np.float32(nodata)] = np.nan

    return data, profile


def read_raster_metadata(path_text):
    import rasterio

    path = resolve_input_path(path_text)
    with rasterio.open(path) as src:
        transform = tuple(round(float(value), 9) for value in src.transform)
        crs_text = None
        if src.crs is not None:
            crs_text = src.crs.to_string()
        return {
            "path": path,
            "shape": (int(src.height), int(src.width)),
            "transform": transform,
            "crs": crs_text,
        }


def validate_raster_alignment(items, context="栅格"):
    metadata = []
    missing = []
    for name, path_text in items:
        if not path_exists(path_text):
            missing.append(f"{name}: {path_text}")
            continue
        metadata.append((name, read_raster_metadata(path_text)))

    if missing:
        raise ValueError(context + "缺少必要文件：\n" + "\n".join(missing))
    if len(metadata) <= 1:
        return metadata

    reference_name, reference = metadata[0]
    problems = []
    for name, item in metadata[1:]:
        if item["shape"] != reference["shape"]:
            problems.append(
                f"{name} 尺寸 {item['shape']} 与 {reference_name} 尺寸 {reference['shape']} 不一致。"
            )
        if item["crs"] != reference["crs"]:
            problems.append(f"{name} 坐标系与 {reference_name} 不一致。")
        if item["transform"] != reference["transform"]:
            problems.append(f"{name} 空间变换与 {reference_name} 不一致。")

    if problems:
        raise ValueError(context + "空间范围不一致：\n" + "\n".join(problems))
    return metadata


def write_float_raster(path_text, data, reference_profile, nodata=-9999.0):
    import rasterio

    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)

    out_data = data.astype(np.float32).copy()
    out_data[np.isnan(out_data)] = nodata

    profile = reference_profile.copy()
    profile["driver"] = "GTiff"
    profile["count"] = 1
    profile["dtype"] = "float32"
    profile["nodata"] = nodata
    profile["compress"] = "lzw"

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out_data, 1)

    return path


def parse_code_list(text, default_values):
    if text is None:
        return list(default_values)

    raw_text = str(text).strip()
    if raw_text == "":
        return list(default_values)

    result = []
    parts = raw_text.replace("，", ",").split(",")
    for part in parts:
        clean_part = part.strip()
        if clean_part == "":
            continue
        result.append(int(float(clean_part)))

    if not result:
        return list(default_values)
    return result


def parse_env_raster_paths(text):
    result = []
    if text is None:
        return result

    raw_text = str(text).strip()
    if raw_text == "":
        return result

    pieces = raw_text.replace("\r", "\n").replace(";", "\n").split("\n")
    index = 1
    for piece in pieces:
        clean_piece = piece.strip()
        if clean_piece == "":
            continue

        if "=" in clean_piece:
            name, path = clean_piece.split("=", 1)
            name = name.strip()
            path = path.strip()
        else:
            path = clean_piece
            name = f"env_{index}"

        if name and path:
            result.append((name, path))
            index = index + 1

    return result
