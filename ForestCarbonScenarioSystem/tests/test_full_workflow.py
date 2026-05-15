import unittest
from pathlib import Path
import sys
import shutil
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from fcscs import __version__
from fcscs.config.defaults import ScenarioConfig
from fcscs.domain.models import EventTable
from fcscs.engines.monte_carlo_engine import AGBDModelEngine, MonteCarloEngine, ReportEngine
from fcscs.engines.scenario_engine import ScenarioEngine, SeverityEngine


class FullWorkflowTest(unittest.TestCase):
    def setUp(self):
        self._temp_dirs = []

    def tearDown(self):
        for temp_dir in self._temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def get_test_output_dir(self):
        temp_root = PROJECT_ROOT / ".test_outputs"
        temp_root.mkdir(parents=True, exist_ok=True)
        output_dir = temp_root / ("fcscs_test_" + uuid.uuid4().hex)
        self._temp_dirs.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def build_small_config(self, severity_method="S1"):
        output_dir = self.get_test_output_dir() / ("workflow_" + severity_method)
        output_dir.mkdir(parents=True, exist_ok=True)
        return self.build_history_raster_config(output_dir, severity_method)

    def build_history_raster_config(self, output_dir, severity_method="S1"):
        rows = 20
        cols = 20
        agbd_2021 = np.full((rows, cols), 96.0, dtype=np.float32)
        agbd_2022 = np.full((rows, cols), 102.0, dtype=np.float32)
        agbd_2022[2:6, 2:6] = 54.0
        agbd_2022[1:7, 1:7] = np.minimum(agbd_2022[1:7, 1:7], 82.0)
        agbd_2022[10:14, 10:14] = 48.0

        tcc_2021 = np.full((rows, cols), 0.68, dtype=np.float32)
        tcc_2022 = np.full((rows, cols), 0.72, dtype=np.float32)
        tcc_2022[2:6, 2:6] = 0.24
        tcc_2022[1:7, 1:7] = np.minimum(tcc_2022[1:7, 1:7], 0.48)
        tcc_2022[10:14, 10:14] = 0.18

        lulc_base = np.full((rows, cols), 1, dtype=np.uint8)
        lulc_2021 = np.full((rows, cols), 1, dtype=np.uint8)
        lulc_2022 = np.full((rows, cols), 1, dtype=np.uint8)
        lulc_2022[2:6, 2:6] = 8
        lulc_target = lulc_2022.copy()

        drivers = np.zeros((4, rows, cols), dtype=np.uint8)
        drivers[0, 10:14, 10:14] = 4
        drivers[1, 10:14, 10:14] = 220
        drivers[2, 2:6, 2:6] = 200
        drivers[3, 10:14, 10:14] = 22

        reserve = np.zeros((rows, cols), dtype=np.uint8)
        slope = np.tile(np.linspace(1, 20, cols, dtype=np.float32), (rows, 1))
        moisture = np.tile(np.linspace(0.3, 0.9, rows, dtype=np.float32).reshape(rows, 1), (1, cols))
        road = np.tile(np.linspace(0.2, 0.8, cols, dtype=np.float32), (rows, 1))

        agbd_2021_path = output_dir / "agbd_2021.tif"
        agbd_2022_path = output_dir / "agbd_2022.tif"
        tcc_2021_path = output_dir / "tcc_2021.tif"
        tcc_2022_path = output_dir / "tcc_2022.tif"
        lulc_base_path = output_dir / "lulc_base.tif"
        lulc_2021_path = output_dir / "lulc_2021.tif"
        lulc_2022_path = output_dir / "lulc_2022.tif"
        lulc_target_path = output_dir / "lulc_target.tif"
        drivers_path = output_dir / "drivers.tif"
        reserve_path = output_dir / "reserve.tif"
        slope_path = output_dir / "slope.tif"
        moisture_path = output_dir / "moisture.tif"
        road_path = output_dir / "road.tif"

        self.write_test_raster(agbd_2021_path, agbd_2021, "float32", -9999.0)
        self.write_test_raster(agbd_2022_path, agbd_2022, "float32", -9999.0)
        self.write_test_raster(tcc_2021_path, tcc_2021, "float32", -9999.0)
        self.write_test_raster(tcc_2022_path, tcc_2022, "float32", -9999.0)
        self.write_test_raster(lulc_base_path, lulc_base, "uint8", 255)
        self.write_test_raster(lulc_2021_path, lulc_2021, "uint8", 255)
        self.write_test_raster(lulc_2022_path, lulc_2022, "uint8", 255)
        self.write_test_raster(lulc_target_path, lulc_target, "uint8", 255)
        self.write_multiband_test_raster(drivers_path, drivers, "uint8", 255)
        self.write_test_raster(reserve_path, reserve, "uint8", 255)
        self.write_test_raster(slope_path, slope, "float32", -9999.0)
        self.write_test_raster(moisture_path, moisture, "float32", -9999.0)
        self.write_test_raster(road_path, road, "float32", -9999.0)

        return ScenarioConfig(
            scenario_name=f"test_{severity_method}",
            mc_n_simulations=6,
            ml_sample_count=1200,
            ml_n_estimators=20,
            ml_max_depth=6,
            severity_method=severity_method,
            base_seed=42,
            agbd_raster_path=str(agbd_2022_path),
            tcc_raster_path=str(tcc_2022_path),
            lulc_base_raster_path=str(lulc_base_path),
            lulc_target_raster_path=str(lulc_target_path),
            drivers_raster_path=str(drivers_path),
            reserve_raster_path=str(reserve_path),
            env_raster_paths=f"slope={slope_path}\nmoisture={moisture_path}\naccessibility={road_path}",
            history_agbd_paths=f"2021={agbd_2021_path}\n2022={agbd_2022_path}",
            history_tcc_paths=f"2021={tcc_2021_path}\n2022={tcc_2022_path}",
            history_lulc_paths=f"2021={lulc_2021_path}\n2022={lulc_2022_path}",
            forest_lulc_codes="1",
            urban_lulc_codes="8",
            logging_driver_value=4,
            reserve_value=1,
            output_dir=str(output_dir),
        )

    def run_complete_flow(self, severity_method):
        config = self.build_small_config(severity_method)
        scenario_engine = ScenarioEngine()
        events = scenario_engine.generate_all_events(config)
        severity_events = SeverityEngine().assign_all(events, config)
        bundle = MonteCarloEngine().run(severity_events, config)
        report = ReportEngine().build_report(bundle)
        return config, scenario_engine, severity_events, bundle, report

    def test_config_yaml_round_trip(self):
        config = self.build_small_config("S1")
        output_dir = self.get_test_output_dir()
        path = output_dir / "scenario_round_trip.yaml"
        config.save_yaml(path)
        loaded = ScenarioConfig.from_yaml(path)

        self.assertEqual(loaded.scenario_name, config.scenario_name)
        self.assertEqual(loaded.batch_name, config.batch_name)
        self.assertEqual(loaded.grid_rows, config.grid_rows)
        self.assertEqual(loaded.grid_cols, config.grid_cols)
        self.assertEqual(loaded.ml_sample_count, config.ml_sample_count)

    def test_version_uses_registration_label(self):
        self.assertEqual(__version__, "V1.0")

    def test_scenario_name_is_sanitized_for_file_paths(self):
        config = ScenarioConfig(scenario_name="../bad/name:one*")

        self.assertEqual(config.scenario_name, "bad_name_one")
        self.assertNotIn("..", config.scenario_name)
        self.assertNotIn("/", config.scenario_name)
        self.assertNotIn(":", config.scenario_name)

        reserved = ScenarioConfig(scenario_name="CON")
        self.assertEqual(reserved.scenario_name, "CON_scenario")

    def test_invalid_year_is_rejected(self):
        config = ScenarioConfig(base_year=2035, target_year=2035)
        with self.assertRaises(ValueError):
            ScenarioEngine().generate_all_events(config)

    def test_logging_generation_rejects_unplaceable_small_grid(self):
        output_dir = self.get_test_output_dir() / "unplaceable_patch"
        output_dir.mkdir(parents=True, exist_ok=True)
        config = self.build_history_raster_config(output_dir, "S1")
        config.logging_patch_min_size = 50
        config.logging_patch_max_size = 50
        config.logging_library_patch_count = 5

        with self.assertRaisesRegex(ValueError, "采伐事件生成失败"):
            ScenarioEngine().generate_all_events(config)

    def test_urban_conv_severity_uses_conversion_floor(self):
        records = pd.DataFrame(
            [
                {"pixel_id": 1, "row": 0, "col": 0, "type": "urban_conv", "y_event": 2025},
                {"pixel_id": 2, "row": 0, "col": 1, "type": "urban_conv", "y_event": 2026},
            ]
        )

        event_table = SeverityEngine().assign(EventTable("urban_conv", records), ScenarioConfig())

        self.assertGreaterEqual(float(event_table.records["Severity"].min()), 0.62)

    def test_s1_full_workflow(self):
        config, scenario_engine, events, bundle, report = self.run_complete_flow("S1")
        self.assertEqual(config.severity_method, "S1")
        self.assertEqual(len(events), 3)
        self.assertGreater(scenario_engine.last_patch_library.summary()["patch_count"], 0)
        self.assertGreater(bundle.summary["mean_agbd_per_ha"], 0)
        self.assertGreater(bundle.summary["mean_agc_per_ha"], 0)
        self.assertGreaterEqual(len(bundle.training_summary_df), 2)
        self.assertFalse(bundle.training_sample_df.empty)
        self.assertFalse(report.total_distribution_df.empty)

    def test_s2_full_workflow(self):
        config, scenario_engine, events, bundle, report = self.run_complete_flow("S2")
        self.assertEqual(config.severity_method, "S2")
        self.assertEqual(len(events), 3)
        self.assertGreater(bundle.summary["mean_model_r2"], 0)
        self.assertIn("平均AGBD", report.metrics)
        self.assertIn("平均AGC", report.metrics)
        self.assertFalse(report.training_sample_df.empty)

    def test_report_can_be_exported_to_csv(self):
        _, _, _, bundle, report = self.run_complete_flow("S1")
        output_dir = self.get_test_output_dir()
        summary_path = output_dir / "summary.csv"
        distribution_path = output_dir / "distribution.csv"
        training_path = output_dir / "training_samples.csv"

        report.summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        report.total_distribution_df.to_csv(distribution_path, index=False, encoding="utf-8-sig")
        report.training_sample_df.to_csv(training_path, index=False, encoding="utf-8-sig")

        summary_df = pd.read_csv(summary_path)
        distribution_df = pd.read_csv(distribution_path)
        training_df = pd.read_csv(training_path)

        self.assertFalse(summary_df.empty)
        self.assertEqual(len(distribution_df), bundle.summary["n_simulations"])
        self.assertFalse(training_df.empty)

    def test_raster_workflow_reads_geotiff_and_writes_outputs(self):
        output_dir = self.get_test_output_dir() / "raster_case"
        output_dir.mkdir(parents=True, exist_ok=True)
        config = self.build_history_raster_config(output_dir, "S1")
        config.scenario_name = "raster_test"
        config.mc_n_simulations = 2
        config.ml_sample_count = 120
        config.ml_n_estimators = 10
        config.ml_max_depth = 4

        scenario_engine = ScenarioEngine()
        events = scenario_engine.generate_all_events(config)
        events = SeverityEngine().assign_all(events, config)
        bundle = MonteCarloEngine().run(events, config)
        report = ReportEngine().build_report(bundle)

        self.assertEqual(bundle.summary["data_mode"], "raster")
        self.assertGreater(bundle.summary["mean_agbd_per_ha"], 0)
        self.assertIn("mean_AGBD_tif", bundle.output_files)
        self.assertTrue(Path(bundle.output_files["mean_AGBD_tif"]).exists())
        self.assertIn(config.batch_name, bundle.output_files["mean_AGBD_tif"])
        self.assertTrue(report.output_files)

    def test_history_training_uses_driver_sample_weight(self):
        output_dir = self.get_test_output_dir() / "history_case"
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = 14
        cols = 14
        agbd_2020 = np.full((rows, cols), 90.0, dtype=np.float32)
        agbd_2022 = np.full((rows, cols), 96.0, dtype=np.float32)
        agbd_2022[3:6, 3:6] = 50.0
        agbd_2022[8:11, 8:11] = 42.0

        tcc_2020 = np.full((rows, cols), 0.65, dtype=np.float32)
        tcc_2022 = np.full((rows, cols), 0.70, dtype=np.float32)
        tcc_2022[3:6, 3:6] = 0.20
        tcc_2022[8:11, 8:11] = 0.12

        lulc_2020 = np.full((rows, cols), 1, dtype=np.uint8)
        lulc_2022 = np.full((rows, cols), 1, dtype=np.uint8)
        lulc_2022[3:6, 3:6] = 8

        drivers = np.zeros((4, rows, cols), dtype=np.uint8)
        drivers[0, 8:11, 8:11] = 4
        drivers[1, 8:11, 8:11] = 220
        drivers[2, 3:6, 3:6] = 200
        drivers[3, 8:11, 8:11] = 22

        slope = np.tile(np.linspace(1, 14, cols, dtype=np.float32), (rows, 1))
        moisture = np.tile(np.linspace(0.2, 0.9, rows, dtype=np.float32).reshape(rows, 1), (1, cols))
        road = np.tile(np.linspace(0.1, 0.8, cols, dtype=np.float32), (rows, 1))
        reserve = np.zeros((rows, cols), dtype=np.uint8)

        agbd_2020_path = output_dir / "agbd_2020.tif"
        agbd_2022_path = output_dir / "agbd_2022.tif"
        tcc_2020_path = output_dir / "tcc_2020.tif"
        tcc_2022_path = output_dir / "tcc_2022.tif"
        lulc_2020_path = output_dir / "lulc_2020.tif"
        lulc_2022_path = output_dir / "lulc_2022.tif"
        drivers_path = output_dir / "drivers_multiband.tif"
        slope_path = output_dir / "slope.tif"
        moisture_path = output_dir / "moisture.tif"
        road_path = output_dir / "road.tif"
        reserve_path = output_dir / "reserve.tif"

        self.write_test_raster(agbd_2020_path, agbd_2020, "float32", -9999.0)
        self.write_test_raster(agbd_2022_path, agbd_2022, "float32", -9999.0)
        self.write_test_raster(tcc_2020_path, tcc_2020, "float32", -9999.0)
        self.write_test_raster(tcc_2022_path, tcc_2022, "float32", -9999.0)
        self.write_test_raster(lulc_2020_path, lulc_2020, "uint8", 255)
        self.write_test_raster(lulc_2022_path, lulc_2022, "uint8", 255)
        self.write_multiband_test_raster(drivers_path, drivers, "uint8", 255)
        self.write_test_raster(slope_path, slope, "float32", -9999.0)
        self.write_test_raster(moisture_path, moisture, "float32", -9999.0)
        self.write_test_raster(road_path, road, "float32", -9999.0)
        self.write_test_raster(reserve_path, reserve, "uint8", 255)

        config = ScenarioConfig(
            scenario_name="history_training",
            base_year=2022,
            target_year=2025,
            mc_n_simulations=1,
            ml_sample_count=80,
            ml_n_estimators=10,
            ml_max_depth=4,
            logging_patch_min_size=1,
            logging_patch_max_size=5,
            logging_library_patch_count=10,
            use_driver_sample_weight=True,
            agbd_raster_path=str(agbd_2022_path),
            tcc_raster_path=str(tcc_2022_path),
            lulc_base_raster_path=str(lulc_2022_path),
            lulc_target_raster_path=str(lulc_2022_path),
            drivers_raster_path=str(drivers_path),
            reserve_raster_path=str(reserve_path),
            env_raster_paths=f"slope={slope_path}\nmoisture={moisture_path}\naccessibility={road_path}",
            history_agbd_paths=f"2021={agbd_2020_path}\n2022={agbd_2022_path}",
            history_tcc_paths=f"2021={tcc_2020_path}\n2022={tcc_2022_path}",
            history_lulc_paths=f"2021={lulc_2020_path}\n2022={lulc_2022_path}",
            forest_lulc_codes="1",
            urban_lulc_codes="8",
            logging_driver_value=4,
            output_dir=str(output_dir),
        )

        events = ScenarioEngine().generate_all_events(config)
        events = SeverityEngine().assign_all(events, config)
        bundle = MonteCarloEngine().run(events, config)

        self.assertFalse(bundle.training_sample_df.empty)
        self.assertIn("sample_weight", bundle.training_sample_df.columns)
        self.assertGreater(float(bundle.training_sample_df["sample_weight"].max()), 0.5)
        self.assertGreater(bundle.summary["mean_agbd_per_ha"], 0)

    def test_raster_shape_mismatch_is_rejected_before_simulation(self):
        output_dir = self.get_test_output_dir() / "raster_mismatch"
        output_dir.mkdir(parents=True, exist_ok=True)

        agbd_path = output_dir / "agbd.tif"
        tcc_path = output_dir / "tcc.tif"
        lulc_base_path = output_dir / "lulc_base.tif"
        lulc_target_path = output_dir / "lulc_target.tif"
        drivers_path = output_dir / "drivers.tif"
        reserve_path = output_dir / "reserve.tif"

        self.write_test_raster(agbd_path, np.ones((20, 20), dtype=np.float32), "float32", -9999.0)
        self.write_test_raster(tcc_path, np.ones((20, 20), dtype=np.float32), "float32", -9999.0)
        self.write_test_raster(lulc_base_path, np.ones((20, 20), dtype=np.uint8), "uint8", 255)
        self.write_test_raster(lulc_target_path, np.ones((21, 20), dtype=np.uint8), "uint8", 255)
        self.write_test_raster(drivers_path, np.zeros((20, 20), dtype=np.uint8), "uint8", 255)
        self.write_test_raster(reserve_path, np.zeros((20, 20), dtype=np.uint8), "uint8", 255)

        config = ScenarioConfig(
            agbd_raster_path=str(agbd_path),
            tcc_raster_path=str(tcc_path),
            lulc_base_raster_path=str(lulc_base_path),
            lulc_target_raster_path=str(lulc_target_path),
            drivers_raster_path=str(drivers_path),
            reserve_raster_path=str(reserve_path),
            env_raster_paths="",
        )

        with self.assertRaisesRegex(ValueError, "空间范围不一致"):
            ScenarioEngine().generate_all_events(config)

    def test_all_nodata_agbd_is_rejected(self):
        output_dir = self.get_test_output_dir() / "raster_nodata"
        output_dir.mkdir(parents=True, exist_ok=True)

        agbd_path = output_dir / "agbd.tif"
        tcc_path = output_dir / "tcc.tif"
        nodata = -9999.0

        self.write_test_raster(agbd_path, np.full((10, 10), nodata, dtype=np.float32), "float32", nodata)
        self.write_test_raster(tcc_path, np.full((10, 10), 0.5, dtype=np.float32), "float32", nodata)

        config = ScenarioConfig(
            agbd_raster_path=str(agbd_path),
            tcc_raster_path=str(tcc_path),
            env_raster_paths="",
        )

        with self.assertRaisesRegex(ValueError, "AGBD 有效像元不足"):
            AGBDModelEngine()._build_raster_feature_surfaces(config, np.random.default_rng(1))

    def write_test_raster(self, path, data, dtype, nodata):
        profile = {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "dtype": dtype,
            "crs": None,
            "transform": from_origin(0, 20, 1, 1),
            "nodata": nodata,
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)

    def write_multiband_test_raster(self, path, data, dtype, nodata):
        profile = {
            "driver": "GTiff",
            "height": data.shape[1],
            "width": data.shape[2],
            "count": data.shape[0],
            "dtype": dtype,
            "crs": None,
            "transform": from_origin(0, 20, 1, 1),
            "nodata": nodata,
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)


if __name__ == "__main__":
    unittest.main()
