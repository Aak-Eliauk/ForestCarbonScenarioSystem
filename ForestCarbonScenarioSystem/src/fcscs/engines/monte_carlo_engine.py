import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from fcscs.config.defaults import name_clean
from fcscs.domain.models import ReportBundle, SimulationBundle
from fcscs.engines.raster_tools import (
    deconstruct_codes,
    deconstruct_envs,
    deconstruct_years,
    construct_path,
    read_grid,
    read_band,
    get_out_dir,
    check_rasterAndalign,
    write_grid,
)


class MonteCarloEngine:
    def run(self, event_tables, config):
        rng = np.random.default_rng(config.base_seed + 200)
        model_engine = AGBDModelEngine()
        # 先用历史数据训练模型，再进入蒙特卡洛预测
        model_context = model_engine.prepare_context(config, event_tables)

        baseline_agbd_surface = model_context["baseline_prediction_surface"]
        agbd_stack = self._run_simulations(config, event_tables, model_context, rng)

        summary = self._build_summary(config, baseline_agbd_surface, agbd_stack, model_context)
        event_summaries = []
        for table in event_tables:
            event_summaries.append(table.summary())
        yearly_event_df = self._build_yearly_event_df(event_tables)
        output_files = self._write_raster_outputs(config, agbd_stack, model_context)

        bundle = SimulationBundle(
            config.scenario_name,
            agbd_stack,
            summary,
            event_summaries,
            yearly_event_df,
            model_context["training_summary_df"],
            model_context["training_sample_df"],
            output_files,
        )
        return bundle

    def _run_simulations(self, config, event_tables, model_context, rng):
        rows = config.grid_rows
        cols = config.grid_cols
        # 每一层保存一次完整蒙特卡洛模拟的AGBD栅格
        agbd_stack = np.zeros((config.mc_n_simulations, rows, cols), dtype=np.float32)

        for sim_index in range(config.mc_n_simulations):
            simulation_surface = self._run_single_simulation(config, event_tables, model_context, rng)
            agbd_stack[sim_index] = simulation_surface

        return agbd_stack

    def _run_single_simulation(self, config, event_tables, model_context, rng):
        model_engine = AGBDModelEngine()
        climate_shift = float(rng.normal(0.0, 1.4))
        climate_noise = rng.normal(0.0, 1.8, size=(config.grid_rows, config.grid_cols)).astype(np.float32)

        baseline_surface = model_engine.predict_baseline_surface(
            config,
            model_context["surfaces"],
            model_context["models"]["baseline"]["model"],
            climate_shift,
        )
        simulation_surface = baseline_surface.copy()

        for event_table in event_tables:
            if event_table.records.empty:
                continue

            model_item = model_context["models"].get(event_table.event_type)
            if model_item is None:
                continue

            rows, cols, predicted = model_engine.predict_event_values(
                config,
                event_table,
                model_context["surfaces"],
                model_item["model"],
                rng,
                climate_shift,
            )
            for index in range(len(predicted)):
                row = int(rows[index])
                col = int(cols[index])
                simulation_surface[row, col] = predicted[index]

        simulation_surface = simulation_surface + climate_noise
        simulation_surface = simulation_surface + climate_shift
        simulation_surface = np.maximum(simulation_surface, 0.0)
        simulation_surface = simulation_surface.astype(np.float32)
        return simulation_surface

    def _build_summary(self, config, baseline_agbd_surface, agbd_stack, model_context):
        mean_surface = np.nanmean(agbd_stack, axis=0)
        mean_agbd_per_sim = np.nanmean(agbd_stack, axis=(1, 2))
        mean_agc_per_sim = mean_agbd_per_sim * config.agbd_to_agc_factor
        total_agbd_per_sim = np.nansum(agbd_stack * config.pixel_area_ha, axis=(1, 2))
        total_agc_per_sim = total_agbd_per_sim * config.agbd_to_agc_factor
        reduction_surface = np.maximum(baseline_agbd_surface - mean_surface, 0.0)
        training_summary_df = model_context["training_summary_df"]
        baseline_agbd_mean = float(np.nanmean(baseline_agbd_surface))

        return {
            "scenario_name": config.scenario_name,
            "batch_name": config.batch_name,
            "n_simulations": int(config.mc_n_simulations),
            "grid_rows": int(config.grid_rows),
            "grid_cols": int(config.grid_cols),
            "agbd_to_agc_factor": float(config.agbd_to_agc_factor),
            "pixel_area_ha": float(config.pixel_area_ha),
            "ml_enabled": True,
            "ml_algorithm": "RandomForestRegressor",
            "ml_model_count": int(len(training_summary_df)),
            "data_mode": "raster",
            "baseline_agbd_mean": baseline_agbd_mean,
            "baseline_agc_mean": float(baseline_agbd_mean * config.agbd_to_agc_factor),
            "mean_agbd_per_ha": float(np.nanmean(mean_agbd_per_sim)),
            "std_agbd_per_ha": float(np.nanstd(mean_agbd_per_sim)),
            "mean_agc_per_ha": float(np.nanmean(mean_agc_per_sim)),
            "std_agc_per_ha": float(np.nanstd(mean_agc_per_sim)),
            "total_agbd_mean": float(np.nanmean(total_agbd_per_sim)),
            "total_agbd_std": float(np.nanstd(total_agbd_per_sim)),
            "total_agc_mean": float(np.nanmean(total_agc_per_sim)),
            "total_agc_std": float(np.nanstd(total_agc_per_sim)),
            "mean_reduction_per_ha": float(np.nanmean(reduction_surface)),
            "max_reduction_per_ha": float(np.nanmax(reduction_surface)),
            "mean_model_r2": float(training_summary_df["r2"].mean()),
            "mean_model_mae": float(training_summary_df["mae"].mean()),
        }

    def _build_yearly_event_df(self, event_tables):
        frames = []
        for table in event_tables:
            annual_df = table.annual_summary()
            if not annual_df.empty:
                frames.append(annual_df)

        if not frames:
            result = pd.DataFrame(columns=["event_type", "year", "count", "severity_mean"])
            return result

        result = pd.concat(frames, ignore_index=True)
        return result

    def _write_raster_outputs(self, config, agbd_stack, model_context):
        output_files = {}
        if not config.write_raster_outputs:
            return output_files

        profile = model_context.get("reference_profile")
        if profile is None:
            return output_files

        out_dir = model_context.get("raster_out_dir")
        if out_dir is None:
            return output_files

        mean_agbd = np.nanmean(agbd_stack, axis=0)
        q05_agbd = np.nanquantile(agbd_stack, 0.05, axis=0).astype(np.float32)
        q95_agbd = np.nanquantile(agbd_stack, 0.95, axis=0).astype(np.float32)
        mean_agc = mean_agbd * config.agbd_to_agc_factor

        mean_agbd_path = out_dir / "mean_AGBD.tif"
        q05_agbd_path = out_dir / "q05_AGBD.tif"
        q95_agbd_path = out_dir / "q95_AGBD.tif"
        mean_agc_path = out_dir / "mean_AGC.tif"

        write_grid(mean_agbd_path, mean_agbd, profile)
        write_grid(q05_agbd_path, q05_agbd, profile)
        write_grid(q95_agbd_path, q95_agbd, profile)
        write_grid(mean_agc_path, mean_agc, profile)

        output_files["mean_AGBD_tif"] = str(mean_agbd_path)
        output_files["q05_AGBD_tif"] = str(q05_agbd_path)
        output_files["q95_AGBD_tif"] = str(q95_agbd_path)
        output_files["mean_AGC_tif"] = str(mean_agc_path)
        return output_files


class AGBDModelEngine:
    def prepare_context(self, config, event_tables):
        rng = np.random.default_rng(config.base_seed + 150)
        surfaces = self._build_raster_feature_surfaces(config, rng)

        models, training_sample_df = self._train_models(config, event_tables, surfaces, rng)
        training_summary_df = self._build_training_summary_df(models)
        baseline_prediction_surface = self.predict_baseline_surface(config, surfaces, models["baseline"]["model"])

        return {
            "surfaces": surfaces,
            "models": models,
            "training_summary_df": training_summary_df,
            "training_sample_df": training_sample_df,
            "baseline_prediction_surface": baseline_prediction_surface,
            "reference_profile": surfaces.get("reference_profile"),
            "raster_out_dir": surfaces.get("raster_out_dir"),
        }

    def predict_baseline_surface(self, config, surfaces, model, climate_shift=0.0):
        rows = config.grid_rows
        cols = config.grid_cols
        span = config.target_year - config.base_year
        env_names = surfaces.get("env_feature_names", ["slope", "moisture", "accessibility"])
        feature_count = 2 + len(env_names) + 1
        feature_matrix = np.zeros((rows * cols, feature_count), dtype=np.float32)

        index = 0
        for row in range(rows):
            for col in range(cols):
                feature_matrix[index, 0] = surfaces["agbd_pre"][row, col]
                feature_matrix[index, 1] = surfaces["tcc_pre"][row, col]
                feature_col = 2
                for env_name in env_names:
                    value = surfaces[env_name][row, col]
                    if env_name == "moisture":
                        value = self._clip_moisture(value + climate_shift * 0.03)
                    feature_matrix[index, feature_col] = value
                    feature_col += 1
                feature_matrix[index, feature_col] = span
                index += 1

        feature_matrix = self._fill_nan_matrix(feature_matrix)
        predicted = model.predict(feature_matrix).astype(np.float32)
        predicted = np.maximum(predicted, 0.0)
        predicted_surface = predicted.reshape(rows, cols)
        return predicted_surface

    def predict_event_values(self, config, event_table, surfaces, model, rng, climate_shift):
        records = event_table.records
        if records.empty:
            rows = np.array([], dtype=int)
            cols = np.array([], dtype=int)
            predicted = np.array([], dtype=np.float32)
            return rows, cols, predicted

        feature_matrix = self._build_event_feature_matrix(config, records, surfaces, rng, climate_shift)
        feature_matrix = self._fill_nan_matrix(feature_matrix)
        predicted = model.predict(feature_matrix).astype(np.float32)
        predicted = np.maximum(predicted, 0.0)
        rows = records["row"].to_numpy(dtype=int)
        cols = records["col"].to_numpy(dtype=int)
        return rows, cols, predicted

    def _train_models(self, config, event_tables, surfaces, rng):
        if not self._history_training_ready(config):
            raise ValueError("历史训练数据不足：请提供至少两个共同年份的AGBD、树冠覆盖度和土地利用栅格。")

        history_result = self._train_models_from_history(config, event_tables, surfaces, rng)
        if history_result is not None:
            return history_result
        raise ValueError("历史训练样本为空：请检查历史AGBD、树冠覆盖度、土地利用和森林损失驱动因素栅格。")

    def _history_training_ready(self, config):
        agbd_paths = deconstruct_years(config.history_agbd_paths)
        tcc_paths = deconstruct_years(config.history_tcc_paths)
        lulc_paths = deconstruct_years(config.history_lulc_paths)
        common_years = sorted(set(agbd_paths.keys()) & set(tcc_paths.keys()) & set(lulc_paths.keys()))
        if len(common_years) < 2:
            return False
        for index in range(len(common_years) - 1):
            if int(common_years[index + 1]) - int(common_years[index]) == 1:
                return True
        return False

    def _train_models_from_history(self, config, event_tables, surfaces, rng):
        history = self._load_history_rasters(config)
        if history is None:
            return None

        baseline_df = self._build_history_baseline_training_df(config, surfaces, history, rng)
        if baseline_df.empty:
            return None

        baseline_features = self._baseline_feature_columns(surfaces)
        event_features = self._event_feature_columns(surfaces)

        models = {}
        models["baseline"] = self._fit_model("baseline", baseline_df, baseline_features, config, config.base_seed + 701)

        logging_df = self._build_history_event_training_df(config, surfaces, history, "logging", rng)
        if logging_df.empty:
            if self._future_event_has_rows(event_tables, "logging"):
                raise ValueError("历史采伐训练样本为空：请检查Drivers分类、loss year波段、历史TCC和历史AGBD。")
        else:
            models["logging"] = self._fit_model("logging", logging_df, event_features, config, config.base_seed + 702)

        edge_df = self._build_history_event_training_df(config, surfaces, history, "urban_edge", rng)
        if edge_df.empty:
            if self._future_event_has_rows(event_tables, "urban_edge"):
                raise ValueError("历史城镇边缘扰动训练样本为空：请检查历史土地利用和树冠覆盖度。")
        else:
            models["urban_edge"] = self._fit_model("urban_edge", edge_df, event_features, config, config.base_seed + 703)

        conv_df = self._build_history_event_training_df(config, surfaces, history, "urban_conv", rng)
        if conv_df.empty:
            if self._future_event_has_rows(event_tables, "urban_conv"):
                raise ValueError("历史城镇转换训练样本为空：请检查历史土地利用和树冠覆盖度。")
        else:
            models["urban_conv"] = self._fit_model("urban_conv", conv_df, event_features, config, config.base_seed + 704)

        training_sample_df = self._build_training_sample_df(baseline_df, logging_df, edge_df, conv_df)
        return models, training_sample_df

    def _future_event_has_rows(self, event_tables, event_type):
        for event_table in event_tables:
            if event_table.event_type != event_type:
                continue
            if not event_table.records.empty:
                return True
        return False

    def _load_history_rasters(self, config):
        agbd_paths = deconstruct_years(config.history_agbd_paths)
        tcc_paths = deconstruct_years(config.history_tcc_paths)
        lulc_paths = deconstruct_years(config.history_lulc_paths)
        # 三类历史栅格必须有共同年份，后面才可以构造相邻年份样本
        common_years = sorted(set(agbd_paths.keys()) & set(tcc_paths.keys()) & set(lulc_paths.keys()))
        if len(common_years) < 2:
            return None

        agbd_by_year = {}
        tcc_by_year = {}
        lulc_by_year = {}
        reference_shape = None

        for year in common_years:
            if not construct_path(agbd_paths[year]).exists():
                continue
            if not construct_path(tcc_paths[year]).exists():
                continue
            if not construct_path(lulc_paths[year]).exists():
                continue
            agbd, _ = read_grid(agbd_paths[year], to_float=True)
            tcc, _ = read_grid(tcc_paths[year], to_float=True)
            lulc, _ = read_grid(lulc_paths[year])
            tcc = self._normalize_percent_surface(tcc)
            if reference_shape is None:
                reference_shape = agbd.shape
            if agbd.shape != reference_shape or tcc.shape != reference_shape or lulc.shape != reference_shape:
                continue
            agbd_by_year[year] = agbd
            tcc_by_year[year] = tcc
            lulc_by_year[year] = lulc

        years = sorted(set(agbd_by_year.keys()) & set(tcc_by_year.keys()) & set(lulc_by_year.keys()))
        if len(years) < 2:
            return None

        drivers_class = None
        drivers_loss_year = None
        logging_probability = None
        urban_probability = None
        drivers_class, _ = read_band(config.drivers_raster_path, 1)
        if drivers_class.shape != reference_shape:
            raise ValueError("Drivers分类波段尺寸必须和历史训练栅格一致。")

        drivers_loss_year, _ = read_band(config.drivers_raster_path, 4)
        if drivers_loss_year.shape != reference_shape:
            raise ValueError("Drivers loss year波段尺寸必须和历史训练栅格一致。")

        if config.use_driver_sample_weight:
            logging_probability = self._read_driver_probability(config, config.logging_probability_band, reference_shape)
            urban_probability = self._read_driver_probability(config, config.urban_probability_band, reference_shape)

        return {
            "years": years,
            "agbd": agbd_by_year,
            "tcc": tcc_by_year,
            "lulc": lulc_by_year,
            "drivers_class": drivers_class,
            "drivers_loss_year": drivers_loss_year,
            "logging_probability": logging_probability,
            "urban_probability": urban_probability,
        }

    def _read_driver_probability(self, config, band, shape):
        try:
            data, _ = read_band(config.drivers_raster_path, int(band), to_float=True)
        except Exception as error:
            raise ValueError("Drivers概率波段读取失败：" + str(error))
        if data.shape != shape:
            raise ValueError("Drivers概率波段尺寸必须和历史训练栅格一致。")
        scale = float(config.driver_probability_scale)
        if scale <= 0:
            scale = 250.0
        data = data / scale
        data = np.clip(data, 0.0, 1.0)
        data = np.where(np.isfinite(data), data, 0.5)
        data = data.astype(np.float32)
        return data

    def _build_history_baseline_training_df(self, config, surfaces, history, rng):
        rows = []
        year_pairs = self._build_history_year_pairs(history["years"])
        forest_codes = deconstruct_codes(config.forest_lulc_codes, [1, 2, 3, 4, 5])
        max_per_pair = max(20, int(config.ml_sample_count / max(len(year_pairs), 1)))

        for start_year, end_year in year_pairs:
            agbd_pre = history["agbd"][start_year]
            agbd_end = history["agbd"][end_year]
            tcc_pre = history["tcc"][start_year]
            valid_mask = np.isfinite(agbd_pre) & np.isfinite(agbd_end) & np.isfinite(tcc_pre)
            if start_year in history["lulc"]:
                valid_mask = valid_mask & np.isin(history["lulc"][start_year], forest_codes)
            if end_year in history["lulc"]:
                valid_mask = valid_mask & np.isin(history["lulc"][end_year], forest_codes)

            cell_ids = np.where(valid_mask.ravel())[0]
            if len(cell_ids) == 0:
                continue
            if len(cell_ids) > max_per_pair:
                cell_ids = rng.choice(cell_ids, size=max_per_pair, replace=False)

            for cell_id in cell_ids:
                row = int(cell_id // config.grid_cols)
                col = int(cell_id % config.grid_cols)
                record = {
                    "AGBD_pre": float(agbd_pre[row, col]),
                    "TCC_pre": float(tcc_pre[row, col]),
                    "span": float(end_year - start_year),
                    "target_agbd": float(max(agbd_end[row, col], 0.0)),
                    "sample_weight": 1.0,
                }
                self._add_env_values_to_record(record, surfaces, row, col)
                rows.append(record)

        if len(rows) > config.ml_sample_count:
            picked = rng.choice(len(rows), size=config.ml_sample_count, replace=False)
            rows = [rows[int(index)] for index in picked]
        data = pd.DataFrame(rows)
        return data

    def _build_history_event_training_df(self, config, surfaces, history, event_type, rng):
        # 事件模型只学习历史中发生过对应扰动的像元
        rows = []
        year_pairs = self._build_history_year_pairs(history["years"])
        max_per_pair = max(20, int(config.ml_sample_count / max(len(year_pairs), 1)))
        forest_codes = deconstruct_codes(config.forest_lulc_codes, [1, 2, 3, 4, 5])
        urban_codes = deconstruct_codes(config.urban_lulc_codes, [8, 9])

        for start_year, end_year in year_pairs:
            mask = self._build_history_event_mask(config, history, event_type, start_year, end_year, forest_codes, urban_codes)
            if mask is None or not mask.any():
                continue

            cell_ids = np.where(mask.ravel())[0]
            if len(cell_ids) > max_per_pair:
                cell_ids = rng.choice(cell_ids, size=max_per_pair, replace=False)

            agbd_pre = history["agbd"][start_year]
            agbd_end = history["agbd"][end_year]
            tcc_pre = history["tcc"][start_year]
            tcc_end = history["tcc"][end_year]
            weight_surface = None
            if event_type == "logging":
                weight_surface = history.get("logging_probability")
            elif event_type == "urban_conv":
                weight_surface = history.get("urban_probability")

            for cell_id in cell_ids:
                row = int(cell_id // config.grid_cols)
                col = int(cell_id % config.grid_cols)
                if not np.isfinite(agbd_pre[row, col]) or not np.isfinite(agbd_end[row, col]):
                    continue
                if not np.isfinite(tcc_pre[row, col]) or not np.isfinite(tcc_end[row, col]):
                    continue

                tcc_before = float(tcc_pre[row, col])
                tcc_after = float(tcc_end[row, col])
                if tcc_before <= 0.05:
                    severity = 0.0
                else:
                    severity = (tcc_before - tcc_after) / tcc_before
                    severity = float(np.clip(severity, 0.0, 0.95))

                if event_type == "urban_conv" and severity < 0.62:
                    severity = 0.62
                weight = 1.0
                if weight_surface is not None:
                    weight = float(weight_surface[row, col])
                    if not np.isfinite(weight) or weight <= 0:
                        weight = 0.5
                    weight = max(weight, 0.05)

                record = {
                    "AGBD_pre": float(agbd_pre[row, col]),
                    "TCC_pre": float(tcc_pre[row, col]),
                    "Severity": float(severity),
                    "tau": float(end_year - start_year),
                    "gap": 1.0,
                    "patch_size": 1.0,
                    "target_agbd": float(max(agbd_end[row, col], 0.0)),
                    "sample_weight": float(weight),
                }
                self._add_env_values_to_record(record, surfaces, row, col)
                rows.append(record)

        if len(rows) > config.ml_sample_count:
            picked = rng.choice(len(rows), size=config.ml_sample_count, replace=False)
            rows = [rows[int(index)] for index in picked]
        data = pd.DataFrame(rows)
        return data

    def _build_history_year_pairs(self, years):
        pairs = []
        for index in range(len(years) - 1):
            pairs.append((int(years[index]), int(years[index + 1])))
        return pairs

    def _build_history_event_mask(self, config, history, event_type, start_year, end_year, forest_codes, urban_codes):
        agbd_pre = history["agbd"][start_year]
        agbd_end = history["agbd"][end_year]
        valid_mask = np.isfinite(agbd_pre) & np.isfinite(agbd_end)

        if event_type == "logging":
            drivers_class = history.get("drivers_class")
            if drivers_class is None:
                return None
            mask = drivers_class == int(config.logging_driver_value)
            loss_year = history.get("drivers_loss_year")
            if loss_year is not None:
                encoded_year = loss_year.astype(np.int32) + 2000
                mask = mask & (encoded_year == int(end_year))
            if end_year in history["lulc"]:
                mask = mask & np.isin(history["lulc"][end_year], forest_codes)
            return mask & valid_mask

        if start_year not in history["lulc"] or end_year not in history["lulc"]:
            return None

        lulc_start = history["lulc"][start_year]
        lulc_end = history["lulc"][end_year]
        conv_mask = np.isin(lulc_start, forest_codes) & np.isin(lulc_end, urban_codes)
        if event_type == "urban_conv":
            return conv_mask & valid_mask

        edge_mask = self._build_edge_mask_from_conversion(conv_mask, np.isin(lulc_end, forest_codes))
        return edge_mask & valid_mask

    def _build_edge_mask_from_conversion(self, conv_mask, forest_mask):
        rows, cols = conv_mask.shape
        edge_mask = np.zeros(conv_mask.shape, dtype=bool)
        conv_rows, conv_cols = np.where(conv_mask)
        for index in range(len(conv_rows)):
            row = int(conv_rows[index])
            col = int(conv_cols[index])
            for row_delta in [-1, 0, 1]:
                for col_delta in [-1, 0, 1]:
                    if row_delta == 0 and col_delta == 0:
                        continue
                    next_row = row + row_delta
                    next_col = col + col_delta
                    if next_row < 0 or next_row >= rows:
                        continue
                    if next_col < 0 or next_col >= cols:
                        continue
                    if forest_mask[next_row, next_col]:
                        edge_mask[next_row, next_col] = True
        return edge_mask

    def _build_training_sample_df(self, baseline_df, logging_df, urban_edge_df, urban_conv_df):
        rows = []
        max_rows = 30
        sample_groups = [
            ("baseline", baseline_df),
            ("logging", logging_df),
            ("urban_edge", urban_edge_df),
            ("urban_conv", urban_conv_df),
        ]
        for model_name, source_df in sample_groups:
            row_index = 0
            while row_index < len(source_df) and row_index < max_rows:
                source_row = source_df.iloc[row_index]
                output_row = {"model_name": model_name, "sample_no": row_index + 1}
                for column_name in source_df.columns:
                    output_row[column_name] = source_row[column_name]
                rows.append(output_row)
                row_index += 1
        data = pd.DataFrame(rows)
        return data

    def _baseline_feature_columns(self, surfaces):
        columns = ["AGBD_pre", "TCC_pre"]
        for name in surfaces.get("env_feature_names", ["slope", "moisture", "accessibility"]):
            columns.append(name)
        columns.append("span")
        return columns

    def _event_feature_columns(self, surfaces):
        columns = ["AGBD_pre", "TCC_pre"]
        for name in surfaces.get("env_feature_names", ["slope", "moisture", "accessibility"]):
            columns.append(name)
        columns.extend(["Severity", "tau", "gap", "patch_size"])
        return columns

    def _fit_model(self, model_name, training_df, feature_columns, config, seed):
        training_df = self._clean_training_df(training_df, feature_columns)

        model = RandomForestRegressor(
            n_estimators=config.ml_n_estimators,
            max_depth=config.ml_max_depth,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=1,
        )

        train_df, test_df = self._split_train_test(training_df, seed)
        x_train = train_df[feature_columns].to_numpy(dtype=np.float32)
        y_train = train_df["target_agbd"].to_numpy(dtype=np.float32)
        x_test = test_df[feature_columns].to_numpy(dtype=np.float32)
        y_test = test_df["target_agbd"].to_numpy(dtype=np.float32)
        x_train = self._fill_nan_matrix(x_train)
        x_test = self._fill_nan_matrix(x_test)

        sample_weight = None
        if config.use_driver_sample_weight:
            if "sample_weight" in train_df.columns:
                sample_weight = train_df["sample_weight"].to_numpy(dtype=np.float32)
                sample_weight = np.where(np.isfinite(sample_weight), sample_weight, 1.0)
                sample_weight = np.maximum(sample_weight, 0.05)

        if sample_weight is None:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, sample_weight=sample_weight)
        predicted = model.predict(x_test)

        mae_value = float(mean_absolute_error(y_test, predicted))
        if len(test_df) >= 2:
            r2_value = float(r2_score(y_test, predicted))
        else:
            r2_value = 0.0

        return {
            "name": model_name,
            "model": model,
            "feature_columns": feature_columns,
            "sample_count": int(len(training_df)),
            "train_count": int(len(train_df)),
            "test_count": int(len(test_df)),
            "mae": round(mae_value, 4),
            "r2": round(r2_value, 4),
        }

    def _clean_training_df(self, training_df, feature_columns):
        clean_df = training_df.copy()
        keep_columns = list(feature_columns)
        keep_columns.append("target_agbd")

        for column in keep_columns:
            clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")

        clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
        clean_df = clean_df.dropna(subset=["target_agbd"])

        if len(clean_df) < 5:
            checked_df = training_df.copy()
            for column in keep_columns:
                checked_df[column] = pd.to_numeric(checked_df[column], errors="coerce")
            checked_df = checked_df.replace([np.inf, -np.inf], np.nan)
            checked_df["target_agbd"] = checked_df["target_agbd"].fillna(0.0)
            return checked_df

        return clean_df

    def _fill_nan_matrix(self, matrix):
        clean_matrix = matrix.astype(np.float32).copy()
        if not np.isnan(clean_matrix).any():
            return clean_matrix

        column_medians = np.nanmedian(clean_matrix, axis=0)
        column_medians = np.where(np.isnan(column_medians), 0.0, column_medians)
        row_ids, col_ids = np.where(np.isnan(clean_matrix))
        clean_matrix[row_ids, col_ids] = column_medians[col_ids]
        return clean_matrix

    def _split_train_test(self, training_df, seed):
        if len(training_df) <= 10:
            train_df = training_df.copy()
            test_df = training_df.copy()
            return train_df, test_df

        rng = np.random.default_rng(seed)
        indices = np.arange(len(training_df))
        rng.shuffle(indices)

        split_index = int(len(indices) * 0.8)
        if split_index <= 0:
            split_index = 1
        if split_index >= len(indices):
            split_index = len(indices) - 1

        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        train_df = training_df.iloc[train_indices].reset_index(drop=True)
        test_df = training_df.iloc[test_indices].reset_index(drop=True)
        return train_df, test_df

    def _build_training_summary_df(self, models):
        rows = []

        for model_name in ["baseline", "logging", "urban_edge", "urban_conv"]:
            if model_name not in models:
                continue

            item = models[model_name]
            rows.append(
                {
                    "model_name": item["name"],
                    "algorithm": "RandomForestRegressor",
                    "sample_count": item["sample_count"],
                    "train_count": item["train_count"],
                    "test_count": item["test_count"],
                    "mae": item["mae"],
                    "r2": item["r2"],
                }
            )

        data = pd.DataFrame(rows)
        return data

    def _build_raster_feature_surfaces(self, config, rng):
        self._check_raster_inputs_for_prediction(config)
        agbd_pre, profile = read_grid(config.agbd_raster_path, to_float=True)
        tcc_pre, _ = read_grid(config.tcc_raster_path, to_float=True)
        self._ensure_min_valid_pixels(agbd_pre, "AGBD")
        self._ensure_min_valid_pixels(tcc_pre, "TCC")
        tcc_pre = self._normalize_percent_surface(tcc_pre)

        if agbd_pre.shape != tcc_pre.shape:
            raise ValueError("AGBD 栅格和 TCC 栅格尺寸必须一致。")

        rows, cols = agbd_pre.shape
        config.grid_rows = int(rows)
        config.grid_cols = int(cols)

        env_map = self._load_env_rasters(config, agbd_pre.shape)
        slope = self._get_required_surface(env_map, ["slope", "Slope", "坡度", "地形"], "地形因子")
        moisture = self._get_required_surface(env_map, ["moisture", "MAP", "AET", "水分", "降水"], "气候水分因子")
        accessibility = self._get_required_surface(env_map, ["accessibility", "DistRoadNet", "distance", "可达性", "道路", "人为"], "人为活动因子")
        slope = self._normalize_slope_surface(slope)
        moisture = self._normalize_unit_surface(moisture, invert=False)
        accessibility = self._normalize_unit_surface(accessibility, invert=True)

        surfaces = {
            "agbd_pre": agbd_pre,
            "tcc_pre": tcc_pre,
            "slope": slope,
            "moisture": moisture,
            "accessibility": accessibility,
            "reference_profile": profile,
            "raster_out_dir": self.construct_raster_dir(config),
        }
        env_feature_names = ["slope", "moisture", "accessibility"]
        used_names = set(env_feature_names)
        used_paths = set()
        for name, path_in in deconstruct_envs(config.env_raster_paths):
            clean_name = self._clean_feature_name(name)
            if clean_name in used_names:
                continue
            path_key = path_in.strip()
            if path_key in used_paths:
                continue
            if name not in env_map:
                continue
            extra_surface = self._normalize_unit_surface(env_map[name], invert=False)
            surfaces[clean_name] = extra_surface
            env_feature_names.append(clean_name)
            used_names.add(clean_name)
            used_paths.add(path_key)

        surfaces["env_feature_names"] = env_feature_names
        return surfaces

    def _get_required_surface(self, env_map, candidate_names, label):
        for name in candidate_names:
            if name in env_map:
                surface = env_map[name].astype(np.float32)
                return surface

        for name in env_map:
            lower_name = name.lower()
            for candidate in candidate_names:
                if candidate.lower() in lower_name:
                    surface = env_map[name].astype(np.float32)
                    return surface

        raise ValueError(label + "缺失或尺寸不一致，请在数据准备中选择对应环境因子栅格。")

    def _clean_feature_name(self, name):
        text = str(name).strip()
        if text == "":
            return "env"
        clean_chars = []
        for char in text:
            if char.isalnum() or char == "_":
                clean_chars.append(char)
            else:
                clean_chars.append("_")
        clean_text = "".join(clean_chars).strip("_")
        if clean_text == "":
            clean_text = "env"
        return clean_text

    def _normalize_percent_surface(self, surface):
        result = surface.astype(np.float32).copy()
        max_value = np.nanmax(result)
        if max_value > 1.5:
            result = result / 100.0
        result = np.clip(result, 0.0, 1.0)
        return result

    def _normalize_slope_surface(self, surface):
        result = surface.astype(np.float32).copy()
        result = np.clip(result, 0.0, 80.0)
        return result

    def _normalize_unit_surface(self, surface, invert=False):
        result = surface.astype(np.float32).copy()
        max_value = np.nanmax(result)
        min_value = np.nanmin(result)

        if max_value <= 1.5 and min_value >= 0.0:
            result = np.clip(result, 0.0, 1.0)
        else:
            low_value = float(np.nanpercentile(result, 2))
            high_value = float(np.nanpercentile(result, 98))
            if high_value <= low_value:
                high_value = low_value + 1.0
            result = (result - low_value) / (high_value - low_value)
            result = np.clip(result, 0.0, 1.0)

        if invert:
            result = 1.0 - result
        return result

    def _check_raster_inputs_for_prediction(self, config):
        check_rasterAndalign(
            [
                ("AGBD", config.agbd_raster_path),
                ("TCC", config.tcc_raster_path),
            ],
            "真实栅格预测",
        )

    def _load_env_rasters(self, config, expected_shape):
        env_map = {}
        env_items = deconstruct_envs(config.env_raster_paths)
        for name, path in env_items:
            if not construct_path(path).exists():
                continue
            data, _ = read_grid(path, to_float=True)
            if data.shape != expected_shape:
                continue
            if not np.isfinite(data).any():
                continue
            env_map[name] = data
        return env_map

    def construct_raster_dir(self, config):
        batch_dir = name_clean(config.batch_name, default="运行批次")
        out_dir = get_out_dir(config.output_dir) / batch_dir / "raster_predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _clip_moisture(self, value):
        value = float(np.clip(value, 0.10, 0.95))
        return value

    def _ensure_min_valid_pixels(self, surface, name, minimum=5):
        valid_count = int(np.isfinite(surface).sum())
        required = min(int(surface.size), int(minimum))
        if valid_count < required:
            raise ValueError(f"{name} 有效像元不足：至少需要 {required} 个，当前 {valid_count} 个。")

    def _add_env_values_to_record(self, record, surfaces, row, col):
        for env_name in surfaces.get("env_feature_names", ["slope", "moisture", "accessibility"]):
            record[env_name] = float(surfaces[env_name][row, col])
        return record

    def _build_event_feature_matrix(self, config, records, surfaces, rng, climate_shift):
        sample_count = len(records)
        env_names = surfaces.get("env_feature_names", ["slope", "moisture", "accessibility"])
        feature_count = 2 + len(env_names) + 4
        feature_matrix = np.zeros((sample_count, feature_count), dtype=np.float32)
        rows = records["row"].to_numpy(dtype=int)
        cols = records["col"].to_numpy(dtype=int)

        severity_values = self._sample_prediction_severity(records, rng)
        tau_values, gap_values = self._sample_prediction_time_values(config, records, rng)

        for index in range(sample_count):
            row = rows[index]
            col = cols[index]
            feature_matrix[index, 0] = surfaces["agbd_pre"][row, col]
            feature_matrix[index, 1] = surfaces["tcc_pre"][row, col]
            feature_col = 2
            for env_name in env_names:
                value = surfaces[env_name][row, col]
                if env_name == "moisture":
                    value = self._clip_moisture(value + climate_shift * 0.03)
                feature_matrix[index, feature_col] = value
                feature_col += 1
            feature_matrix[index, feature_col] = severity_values[index]
            feature_matrix[index, feature_col + 1] = tau_values[index]
            feature_matrix[index, feature_col + 2] = gap_values[index]
            feature_matrix[index, feature_col + 3] = float(records.iloc[index].get("patch_size", 1))

        return feature_matrix

    def _sample_prediction_severity(self, records, rng):
        base_values = records["Severity"].to_numpy(dtype=np.float32)
        random_part = rng.normal(0.0, 0.03, size=len(records))
        factor_part = rng.uniform(0.92, 1.08, size=len(records))
        severity_values = base_values * factor_part + random_part
        severity_values = np.clip(severity_values, 0.0, 0.95).astype(np.float32)
        return severity_values

    def _sample_prediction_time_values(self, config, records, rng):
        tau_values = []

        for _, record in records.iterrows():
            year = int(record.get("y_event", config.base_year))
            tau = max(int(config.target_year) - year, 0)
            tau_values.append(tau)

        tau_array = np.asarray(tau_values, dtype=np.float32)
        gap_array = np.maximum(tau_array + rng.normal(0.0, 0.5, size=len(tau_array)), 0.0)
        return tau_array, gap_array


class ReportEngine:
    def build_report(self, bundle):
        summary = dict(bundle.summary)
        summary_df = self._build_summary_df(summary)
        total_distribution_df = self._build_total_distribution_df(bundle)
        metrics = self._build_metrics(summary)

        report = ReportBundle(
            bundle.scenario_name,
            summary_df,
            total_distribution_df,
            metrics,
            bundle.yearly_event_df,
            bundle.training_summary_df,
            bundle.training_sample_df,
            bundle.output_files,
        )
        return report

    def _build_summary_df(self, summary):
        rows = []
        labels = {
            "scenario_name": "情景名称",
            "batch_name": "运行批次",
            "n_simulations": "模拟次数",
            "grid_rows": "网格行数",
            "grid_cols": "网格列数",
            "data_mode": "数据模式",
            "baseline_agbd_mean": "基准 AGBD 均值",
            "baseline_agc_mean": "基准 AGC 均值",
            "mean_agbd_per_ha": "模拟 AGBD 均值",
            "std_agbd_per_ha": "模拟 AGBD 标准差",
            "mean_agc_per_ha": "模拟 AGC 均值",
            "std_agc_per_ha": "模拟 AGC 标准差",
            "total_agbd_mean": "总 AGBD 均值",
            "total_agbd_std": "总 AGBD 标准差",
            "total_agc_mean": "总 AGC 均值",
            "total_agc_std": "总 AGC 标准差",
            "mean_reduction_per_ha": "平均损失强度",
            "max_reduction_per_ha": "最大损失强度",
            "mean_model_r2": "模型 R2 均值",
            "mean_model_mae": "模型 MAE 均值",
        }
        for key, value in summary.items():
            rows.append({"metric": key, "label": labels.get(key, key), "value": value})
        data = pd.DataFrame(rows)
        return data

    def _build_total_distribution_df(self, bundle):
        stack = bundle.sim_stack
        mean_agbd = np.nanmean(stack, axis=(1, 2))
        mean_agc = mean_agbd * float(bundle.summary.get("agbd_to_agc_factor", 0.47))
        totals = np.nansum(stack * float(bundle.summary.get("pixel_area_ha", 1.0)), axis=(1, 2))
        data = pd.DataFrame(
            {
                "simulation": np.arange(1, len(totals) + 1, dtype=int),
                "mean_agbd_per_ha": mean_agbd.astype(float),
                "mean_agc_per_ha": mean_agc.astype(float),
                "total_agbd": totals.astype(float),
                "total_agc": totals.astype(float) * float(bundle.summary.get("agbd_to_agc_factor", 0.47)),
            }
        )
        return data

    def _build_metrics(self, summary):
        return {
            "平均AGBD": summary.get("mean_agbd_per_ha"),
            "平均AGC": summary.get("mean_agc_per_ha"),
            "总AGBD": summary.get("total_agbd_mean"),
            "总AGC": summary.get("total_agc_mean"),
            "平均损失强度": summary.get("mean_reduction_per_ha"),
            "模型R2": summary.get("mean_model_r2"),
            "模型MAE": summary.get("mean_model_mae"),
            "mean_agbd_per_ha": summary.get("mean_agbd_per_ha"),
            "mean_agc_per_ha": summary.get("mean_agc_per_ha"),
            "total_agbd_mean": summary.get("total_agbd_mean"),
            "total_agc_mean": summary.get("total_agc_mean"),
            "mean_reduction_per_ha": summary.get("mean_reduction_per_ha"),
            "mean_model_r2": summary.get("mean_model_r2"),
            "mean_model_mae": summary.get("mean_model_mae"),
        }
