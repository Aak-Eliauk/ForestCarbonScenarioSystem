from pathlib import Path
from datetime import datetime

import yaml


INVALID_SCENARIO_NAME_CHARS = set('<>:"/\\|?*')
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


DEFAULT_HISTORY_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]


def build_default_year_paths(folder_name, file_prefix):
    lines = []
    for year in DEFAULT_HISTORY_YEARS:
        lines.append(f"{year}=../data/{folder_name}/{file_prefix}_{year}.tif")
    return "\n".join(lines)


def sanitize_scenario_name(value, default="基准情景", max_length=80):
    text = str(value).strip()
    if not text:
        text = str(default)

    safe_chars = []
    for char in text:
        if ord(char) < 32 or char in INVALID_SCENARIO_NAME_CHARS:
            safe_chars.append("_")
        else:
            safe_chars.append(char)

    safe_name = "".join(safe_chars)
    while ".." in safe_name:
        safe_name = safe_name.replace("..", "_")
    safe_name = safe_name.strip(" ._")

    if not safe_name:
        safe_name = str(default)
    if safe_name.upper() in WINDOWS_RESERVED_NAMES:
        safe_name = safe_name + "_scenario"
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length].rstrip(" ._")
    return safe_name or str(default)


def build_default_batch_name():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "运行批次_" + timestamp


class ScenarioConfig:
    def __init__(
        self,
        scenario_name="基准情景",
        batch_name=None,
        base_year=2022,
        target_year=2035,
        future_years=None,
        logging_area_reduction=0.0,
        logging_severity_reduction=0.0,
        logging_severity_cap_quantile=None,
        urban_area_reduction=0.0,
        urban_speed_shift=0.0,
        urban_severity_reduction=0.0,
        reserve_ratio=0.10,
        urban_center_count=3,
        logging_patch_min_size=6,
        logging_patch_max_size=18,
        logging_library_years=6,
        logging_library_patch_count=180,
        mc_n_simulations=100,
        severity_method="S1",
        base_seed=42,
        grid_rows=128,
        grid_cols=128,
        agbd_to_agc_factor=0.47,
        pixel_area_ha=1.0,
        ml_sample_count=3000,
        ml_n_estimators=60,
        ml_max_depth=10,
        use_history_training=True,
        history_agbd_paths=None,
        history_tcc_paths=None,
        history_lulc_paths=None,
        use_driver_sample_weight=True,
        logging_probability_band=2,
        urban_probability_band=3,
        driver_probability_scale=250.0,
        severity_sample_count=4000,
        use_raster_data=True,
        agbd_raster_path="../data/AGBD/Hubei_AGB_2022.tif",
        tcc_raster_path="../data/TCC/Hubei_TCC_2022.tif",
        lulc_base_raster_path="../data/LULC/Hubei_LULC_2022.tif",
        lulc_target_raster_path="../data/PLUS_predictions/BAUSimulation_1.tif",
        drivers_raster_path="../data/森林损失/Hubei_LossDriversAndYear_2001_2023.tif",
        reserve_raster_path="../data/自然保护区/Hubei_NatureReserve.tif",
        env_raster_paths=(
            "slope=../data/environmentFactors/地形/Hubei_Slope.tif\n"
            "moisture=../data/environmentFactors/温度降水/Hubei_MAP_2000_2022.tif\n"
            "accessibility=../data/欧氏距离/Hubei_Dist_RoadNet.tif\n"
            "DEM=../data/environmentFactors/地形/Hubei_DEM.tif\n"
            "TPI=../data/environmentFactors/地形/Hubei_TPI_100m.tif\n"
            "MAT=../data/environmentFactors/温度降水/Hubei_MAT_2000_2022.tif\n"
            "AET=../data/environmentFactors/温度降水/Hubei_Annual_AET_2000_2022.tif\n"
            "Nightlight=../data/environmentFactors/社会经济/Hubei_Nightlight_2022.tif\n"
            "PopDensity=../data/environmentFactors/社会经济/Hubei_Pop_Density_2020.tif\n"
            "NPP=../data/Hubei_NPP_Mean_2000_2022.tif"
        ),
        forest_lulc_codes="4,5,6",
        urban_lulc_codes="13",
        logging_driver_value=4,
        reserve_value=1,
        write_raster_outputs=True,
        output_dir="../ForestCarbonScenarioSystem_outputs",
    ):
        self.scenario_name = sanitize_scenario_name(scenario_name)
        if batch_name is None:
            batch_name = build_default_batch_name()
        self.batch_name = sanitize_scenario_name(batch_name, default=build_default_batch_name())
        self.base_year = int(base_year)
        self.target_year = int(target_year)
        if future_years is None:
            self.future_years = self.build_future_years(self.base_year, self.target_year)
        else:
            self.future_years = [int(year) for year in future_years]

        self.logging_area_reduction = float(logging_area_reduction)
        self.logging_severity_reduction = float(logging_severity_reduction)
        self.logging_severity_cap_quantile = logging_severity_cap_quantile
        self.urban_area_reduction = float(urban_area_reduction)
        self.urban_speed_shift = float(urban_speed_shift)
        self.urban_severity_reduction = float(urban_severity_reduction)
        self.reserve_ratio = float(reserve_ratio)
        self.urban_center_count = int(urban_center_count)
        self.logging_patch_min_size = int(logging_patch_min_size)
        self.logging_patch_max_size = int(logging_patch_max_size)
        self.logging_library_years = int(logging_library_years)
        self.logging_library_patch_count = int(logging_library_patch_count)
        self.mc_n_simulations = int(mc_n_simulations)
        self.severity_method = str(severity_method).upper()
        self.base_seed = int(base_seed)
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.agbd_to_agc_factor = float(agbd_to_agc_factor)
        self.pixel_area_ha = float(pixel_area_ha)
        self.ml_sample_count = int(ml_sample_count)
        self.ml_n_estimators = int(ml_n_estimators)
        self.ml_max_depth = int(ml_max_depth)
        self.use_history_training = bool(use_history_training)
        if history_agbd_paths is None:
            history_agbd_paths = build_default_year_paths("AGBD", "Hubei_AGB")
        if history_tcc_paths is None:
            history_tcc_paths = build_default_year_paths("TCC", "Hubei_TCC")
        if history_lulc_paths is None:
            history_lulc_paths = build_default_year_paths("LULC", "Hubei_LULC")
        self.history_agbd_paths = str(history_agbd_paths)
        self.history_tcc_paths = str(history_tcc_paths)
        self.history_lulc_paths = str(history_lulc_paths)
        self.use_driver_sample_weight = bool(use_driver_sample_weight)
        self.logging_probability_band = int(logging_probability_band)
        self.urban_probability_band = int(urban_probability_band)
        self.driver_probability_scale = float(driver_probability_scale)
        self.severity_sample_count = int(severity_sample_count)
        self.use_raster_data = bool(use_raster_data)
        self.agbd_raster_path = str(agbd_raster_path)
        self.tcc_raster_path = str(tcc_raster_path)
        self.lulc_base_raster_path = str(lulc_base_raster_path)
        self.lulc_target_raster_path = str(lulc_target_raster_path)
        self.drivers_raster_path = str(drivers_raster_path)
        self.reserve_raster_path = str(reserve_raster_path)
        self.env_raster_paths = str(env_raster_paths)
        self.forest_lulc_codes = str(forest_lulc_codes)
        self.urban_lulc_codes = str(urban_lulc_codes)
        self.logging_driver_value = int(logging_driver_value)
        self.reserve_value = int(reserve_value)
        self.write_raster_outputs = bool(write_raster_outputs)
        self.output_dir = str(output_dir)

    @staticmethod
    def build_future_years(base_year, target_year):
        years = []
        year = int(base_year) + 1
        while year <= int(target_year):
            years.append(year)
            year += 1
        return years

    def to_dict(self):
        return {
            "scenario_name": self.scenario_name,
            "batch_name": self.batch_name,
            "base_year": self.base_year,
            "target_year": self.target_year,
            "future_years": list(self.future_years),
            "logging_area_reduction": self.logging_area_reduction,
            "logging_severity_reduction": self.logging_severity_reduction,
            "logging_severity_cap_quantile": self.logging_severity_cap_quantile,
            "urban_area_reduction": self.urban_area_reduction,
            "urban_speed_shift": self.urban_speed_shift,
            "urban_severity_reduction": self.urban_severity_reduction,
            "reserve_ratio": self.reserve_ratio,
            "urban_center_count": self.urban_center_count,
            "logging_patch_min_size": self.logging_patch_min_size,
            "logging_patch_max_size": self.logging_patch_max_size,
            "logging_library_years": self.logging_library_years,
            "logging_library_patch_count": self.logging_library_patch_count,
            "mc_n_simulations": self.mc_n_simulations,
            "severity_method": self.severity_method,
            "base_seed": self.base_seed,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "agbd_to_agc_factor": self.agbd_to_agc_factor,
            "pixel_area_ha": self.pixel_area_ha,
            "ml_sample_count": self.ml_sample_count,
            "ml_n_estimators": self.ml_n_estimators,
            "ml_max_depth": self.ml_max_depth,
            "use_history_training": self.use_history_training,
            "history_agbd_paths": self.history_agbd_paths,
            "history_tcc_paths": self.history_tcc_paths,
            "history_lulc_paths": self.history_lulc_paths,
            "use_driver_sample_weight": self.use_driver_sample_weight,
            "logging_probability_band": self.logging_probability_band,
            "urban_probability_band": self.urban_probability_band,
            "driver_probability_scale": self.driver_probability_scale,
            "severity_sample_count": self.severity_sample_count,
            "use_raster_data": self.use_raster_data,
            "agbd_raster_path": self.agbd_raster_path,
            "tcc_raster_path": self.tcc_raster_path,
            "lulc_base_raster_path": self.lulc_base_raster_path,
            "lulc_target_raster_path": self.lulc_target_raster_path,
            "drivers_raster_path": self.drivers_raster_path,
            "reserve_raster_path": self.reserve_raster_path,
            "env_raster_paths": self.env_raster_paths,
            "forest_lulc_codes": self.forest_lulc_codes,
            "urban_lulc_codes": self.urban_lulc_codes,
            "logging_driver_value": self.logging_driver_value,
            "reserve_value": self.reserve_value,
            "write_raster_outputs": self.write_raster_outputs,
            "output_dir": self.output_dir,
        }

    def copy(self):
        return ScenarioConfig(**self.to_dict())

    def save_yaml(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            yaml.safe_dump(self.to_dict(), file, allow_unicode=True, sort_keys=False)
        return path

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        return cls(**data)


def list_preset_names():
    return ["基准情景", "生态保育", "城镇控制", "采伐控制", "平衡发展"]


def build_preset_config(preset_name):
    if preset_name == "BAU":
        preset_name = "基准情景"

    if preset_name == "生态保育":
        return ScenarioConfig(
            scenario_name="生态保育",
            logging_area_reduction=0.35,
            logging_severity_reduction=0.20,
            urban_area_reduction=0.25,
            urban_severity_reduction=0.15,
            urban_speed_shift=0.25,
            reserve_ratio=0.18,
            urban_center_count=2,
            logging_patch_min_size=5,
            logging_patch_max_size=14,
            logging_library_patch_count=200,
            base_seed=123,
        )

    if preset_name == "城镇控制":
        return ScenarioConfig(
            scenario_name="城镇控制",
            urban_area_reduction=0.40,
            urban_severity_reduction=0.25,
            urban_speed_shift=0.45,
            reserve_ratio=0.15,
            urban_center_count=2,
            logging_patch_min_size=6,
            logging_patch_max_size=18,
            logging_library_patch_count=180,
            base_seed=56,
        )

    if preset_name == "采伐控制":
        return ScenarioConfig(
            scenario_name="采伐控制",
            logging_area_reduction=0.45,
            logging_severity_reduction=0.30,
            reserve_ratio=0.14,
            urban_center_count=3,
            logging_patch_min_size=4,
            logging_patch_max_size=12,
            logging_library_patch_count=220,
            base_seed=77,
        )

    if preset_name == "平衡发展":
        return ScenarioConfig(
            scenario_name="平衡发展",
            logging_area_reduction=0.15,
            logging_severity_reduction=0.10,
            urban_area_reduction=0.12,
            urban_severity_reduction=0.08,
            urban_speed_shift=0.10,
            reserve_ratio=0.12,
            urban_center_count=4,
            logging_patch_min_size=6,
            logging_patch_max_size=16,
            logging_library_patch_count=180,
            base_seed=88,
        )

    return ScenarioConfig()


class DataCatalog:
    def __init__(
        self,
        agbd_path="data/baseline/agbd_2022.tif",
        tcc_path="data/baseline/tcc_2022.tif",
        lulc_path="data/baseline/lulc_2022.tif",
        drivers_path="data/drivers/drivers_classification.tif",
        reserve_path="data/drivers/nature_reserve.tif",
        static_env_dir="data/env",
        outputs_dir="../ForestCarbonScenarioSystem_outputs",
        models_dir="models",
    ):
        self.agbd_path = agbd_path
        self.tcc_path = tcc_path
        self.lulc_path = lulc_path
        self.drivers_path = drivers_path
        self.reserve_path = reserve_path
        self.static_env_dir = static_env_dir
        self.outputs_dir = outputs_dir
        self.models_dir = models_dir

    def to_dict(self):
        return {
            "agbd_path": self.agbd_path,
            "tcc_path": self.tcc_path,
            "lulc_path": self.lulc_path,
            "drivers_path": self.drivers_path,
            "reserve_path": self.reserve_path,
            "static_env_dir": self.static_env_dir,
            "outputs_dir": self.outputs_dir,
            "models_dir": self.models_dir,
        }
