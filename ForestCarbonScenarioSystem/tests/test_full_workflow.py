import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from fcscs.config.defaults import ScenarioConfig
from fcscs.engines.monte_carlo_engine import MonteCarloEngine, ReportEngine
from fcscs.engines.scenario_engine import ScenarioEngine, SeverityEngine


class FullWorkflowTest(unittest.TestCase):
    def get_test_output_dir(self):
        root = Path(__file__).resolve().parents[1]
        output_dir = root.parent / "ForestCarbonScenarioSystem_outputs" / "test_runtime"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def build_small_config(self, severity_method="S1"):
        return ScenarioConfig(
            scenario_name=f"test_{severity_method}",
            grid_rows=64,
            grid_cols=64,
            mc_n_simulations=6,
            ml_sample_count=1200,
            ml_n_estimators=20,
            ml_max_depth=6,
            severity_method=severity_method,
            base_seed=42,
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
        self.assertEqual(loaded.grid_rows, config.grid_rows)
        self.assertEqual(loaded.grid_cols, config.grid_cols)
        self.assertEqual(loaded.ml_sample_count, config.ml_sample_count)

    def test_invalid_year_is_rejected(self):
        config = ScenarioConfig(base_year=2035, target_year=2035)
        with self.assertRaises(ValueError):
            ScenarioEngine().generate_all_events(config)

    def test_s1_full_workflow(self):
        config, scenario_engine, events, bundle, report = self.run_complete_flow("S1")
        self.assertEqual(config.severity_method, "S1")
        self.assertEqual(len(events), 3)
        self.assertEqual(scenario_engine.last_patch_library.summary()["patch_count"], config.logging_library_patch_count)
        self.assertGreater(bundle.summary["mean_agbd_per_ha"], 0)
        self.assertGreater(bundle.summary["mean_agc_per_ha"], 0)
        self.assertEqual(len(bundle.training_summary_df), 4)
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

        agbd_path = output_dir / "agbd.tif"
        tcc_path = output_dir / "tcc.tif"
        lulc_base_path = output_dir / "lulc_base.tif"
        lulc_target_path = output_dir / "lulc_target.tif"
        drivers_path = output_dir / "drivers.tif"
        reserve_path = output_dir / "reserve.tif"

        rows = 20
        cols = 20
        agbd = np.full((rows, cols), 95.0, dtype=np.float32)
        tcc = np.full((rows, cols), 0.55, dtype=np.float32)
        lulc_base = np.full((rows, cols), 1, dtype=np.uint8)
        lulc_target = np.full((rows, cols), 1, dtype=np.uint8)
        lulc_target[3:8, 3:8] = 8
        drivers = np.zeros((rows, cols), dtype=np.uint8)
        drivers[10:15, 10:15] = 4
        reserve = np.zeros((rows, cols), dtype=np.uint8)
        reserve[0:2, 0:2] = 1

        self.write_test_raster(agbd_path, agbd, "float32", -9999.0)
        self.write_test_raster(tcc_path, tcc, "float32", -9999.0)
        self.write_test_raster(lulc_base_path, lulc_base, "uint8", 255)
        self.write_test_raster(lulc_target_path, lulc_target, "uint8", 255)
        self.write_test_raster(drivers_path, drivers, "uint8", 255)
        self.write_test_raster(reserve_path, reserve, "uint8", 255)

        config = ScenarioConfig(
            scenario_name="raster_test",
            base_year=2022,
            target_year=2025,
            mc_n_simulations=2,
            ml_sample_count=120,
            ml_n_estimators=10,
            ml_max_depth=4,
            logging_patch_min_size=1,
            logging_patch_max_size=8,
            logging_library_patch_count=20,
            use_raster_data=True,
            agbd_raster_path=str(agbd_path),
            tcc_raster_path=str(tcc_path),
            lulc_base_raster_path=str(lulc_base_path),
            lulc_target_raster_path=str(lulc_target_path),
            drivers_raster_path=str(drivers_path),
            reserve_raster_path=str(reserve_path),
            forest_lulc_codes="1",
            urban_lulc_codes="8",
            logging_driver_value=4,
            reserve_value=1,
            pixel_area_ha=1.0,
        )

        scenario_engine = ScenarioEngine()
        events = scenario_engine.generate_all_events(config)
        events = SeverityEngine().assign_all(events, config)
        bundle = MonteCarloEngine().run(events, config)
        report = ReportEngine().build_report(bundle)

        self.assertEqual(bundle.summary["data_mode"], "raster")
        self.assertGreater(bundle.summary["mean_agbd_per_ha"], 0)
        self.assertIn("mean_AGBD_tif", bundle.output_files)
        self.assertTrue(Path(bundle.output_files["mean_AGBD_tif"]).exists())
        self.assertTrue(report.output_files)

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


if __name__ == "__main__":
    unittest.main()
