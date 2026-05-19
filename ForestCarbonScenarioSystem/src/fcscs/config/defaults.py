from datetime import datetime
from pathlib import Path
import yaml


HISTORY_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]
ENV_RASTER = (
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
)


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
        history_agbd_paths=None,
        history_tcc_paths=None,
        history_lulc_paths=None,
        use_driver_sample_weight=True,
        logging_probability_band=2,
        urban_probability_band=3,
        driver_probability_scale=250.0,
        severity_sample_count=4000,
        agbd_raster_path="../data/AGBD/Hubei_AGB_2022.tif",
        tcc_raster_path="../data/TCC/Hubei_TCC_2022.tif",
        lulc_base_raster_path="../data/LULC/Hubei_LULC_2022.tif",
        lulc_target_raster_path="../data/PLUS_predictions/BAUSimulation_1.tif",
        drivers_raster_path="../data/森林损失/Hubei_LossDriversAndYear_2001_2023.tif",
        reserve_raster_path="../data/自然保护区/Hubei_NatureReserve.tif",
        env_raster_paths=ENV_RASTER,
        forest_lulc_codes="4,5,6",
        urban_lulc_codes="13",
        logging_driver_value=4,
        reserve_value=1,
        write_raster_outputs=True,
        output_dir="../ForestCarbonScenarioSystem_outputs",
    ):
        #基本信息和年份
        self.scenario_name = name_clean(scenario_name, default="基准情景")
        if batch_name is None:
            batch_name = construct_batch(self.scenario_name)
        self.batch_name = name_clean(batch_name, default=construct_batch(self.scenario_name))
        self.base_year = int(base_year)
        self.target_year = int(target_year)

        if future_years is None:
            self.future_years = self.build_future_years(self.base_year, self.target_year)
        else:
            self.future_years = []
            for year in future_years:
                self.future_years.append(int(year))

        #情景参数
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

        #蒙特卡洛和模型参数
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

        #历史栅格
        if history_agbd_paths is None:
            history_agbd_paths = construct_year_file("AGBD", "Hubei_AGB")
        if history_tcc_paths is None:
            history_tcc_paths = construct_year_file("TCC", "Hubei_TCC")
        if history_lulc_paths is None:
            history_lulc_paths = construct_year_file("LULC", "Hubei_LULC")
        self.history_agbd_paths = str(history_agbd_paths)
        self.history_tcc_paths = str(history_tcc_paths)
        self.history_lulc_paths = str(history_lulc_paths)

        self.use_driver_sample_weight = bool(use_driver_sample_weight)
        self.logging_probability_band = int(logging_probability_band)
        self.urban_probability_band = int(urban_probability_band)
        self.driver_probability_scale = float(driver_probability_scale)
        self.severity_sample_count = int(severity_sample_count)

        #基准栅格和约束栅格路径
        self.agbd_raster_path = str(agbd_raster_path)
        self.tcc_raster_path = str(tcc_raster_path)
        self.lulc_base_raster_path = str(lulc_base_raster_path)
        self.lulc_target_raster_path = str(lulc_target_raster_path)
        self.drivers_raster_path = str(drivers_raster_path)
        self.reserve_raster_path = str(reserve_raster_path)
        self.env_raster_paths = str(env_raster_paths)

        #LULC类别编码和输出
        self.forest_lulc_codes = str(forest_lulc_codes)
        self.urban_lulc_codes = str(urban_lulc_codes)
        self.logging_driver_value = int(logging_driver_value)
        self.reserve_value = int(reserve_value)
        self.write_raster_outputs = bool(write_raster_outputs)
        self.output_dir = str(output_dir)

    def build_future_years(self, base_year, target_year):
        years = []
        year = int(base_year) + 1
        while year <= int(target_year):
            years.append(year)
            year = year + 1
        return years

    def to_dict(self):
        data = dict(self.__dict__)
        data["future_years"] = list(self.future_years)
        return data

    def copy(self):
        data = self.to_dict()
        return ScenarioConfig(**data)

    def save_yaml(self, path):
        path = Path(path)
        folder = path.parent
        folder.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
        return path

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if data is None:
            data = {}
        return cls(**data)


def construct_year_file(folder_name, file_prefix):
    lines = []
    for year in HISTORY_YEARS:
        line = str(year) + "=../data/" + folder_name + "/" + file_prefix + "_" + str(year) + ".tif"
        lines.append(line)
    return "\n".join(lines)


def name_clean(value, default="情景", max_length=80):
    text = str(value).strip()
    if text == "":
        text = str(default)

    result = text.strip(" ._")
    if result == "":
        result = str(default)

    if len(result) > max_length:
        result = result[:max_length]
        result = result.strip(" ._")

    if result == "":
        result = str(default)

    return result


def construct_batch(scenario_name="运行批次"):
    time_text = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_text = name_clean(scenario_name, default="运行批次", max_length=42)
    return name_clean(clean_text + "_" + time_text, default="运行批次_" + time_text)



def list_preset_names():
    return ["基准情景", "生态保育", "城镇控制", "采伐控制", "平衡发展"]


def build_preset_config(preset_name):
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
