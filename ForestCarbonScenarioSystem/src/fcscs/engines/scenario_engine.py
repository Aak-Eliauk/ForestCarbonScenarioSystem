import math

import numpy as np
import pandas as pd

from fcscs.domain.models import EventTable
from fcscs.engines.raster_tools import (
    parse_code_list,
    parse_env_raster_paths,
    parse_year_raster_paths,
    path_exists,
    read_raster,
    read_raster_band,
    rasterio_is_available,
    validate_raster_alignment,
)


# 情景生成模块负责通过Drivers和LULC生成未来扰动事件。


class ScenarioEngine:
    def __init__(self):
        self.last_patch_library = None

    def generate_all_events(self, config):
        self._check_config(config)
        if not config.use_raster_data:
            raise ValueError("演示网格模式已删除，请使用真实栅格数据运行系统。")
        return self._generate_all_events_from_rasters(config)

    def _generate_all_events_from_rasters(self, config):
        # 读取基准年和目标年 LULC 后，先判断森林、城镇和保护区范围。
        self._check_raster_config(config)
        rng = np.random.default_rng(config.base_seed)

        lulc_base, _ = read_raster(config.lulc_base_raster_path)
        lulc_target, _ = read_raster(config.lulc_target_raster_path)

        rows, cols = lulc_base.shape
        config.grid_rows = int(rows)
        config.grid_cols = int(cols)

        reserve_mask = self._build_raster_reserve_mask(config, lulc_base.shape)
        forest_codes = parse_code_list(config.forest_lulc_codes, [1, 2, 3, 4, 5])
        urban_codes = parse_code_list(config.urban_lulc_codes, [8, 9])

        forest_base = np.isin(lulc_base, forest_codes)
        forest_target = np.isin(lulc_target, forest_codes)
        urban_target = np.isin(lulc_target, urban_codes)

        # 采伐事件先生成，避免与后续城镇事件冲突。
        logging_events = self._generate_raster_logging_events(config, rng, forest_target, reserve_mask)
        logging_pixels = self._build_pixel_id_set(logging_events.records)

        urban_conv_events = self._generate_raster_urban_conv_events(
            config,
            rng,
            forest_base,
            urban_target,
            reserve_mask,
            logging_pixels,
        )
        conv_pixels = self._build_pixel_id_set(urban_conv_events.records)

        urban_edge_events = self._generate_raster_urban_edge_events(
            config,
            urban_conv_events.records,
            forest_target,
            reserve_mask,
            logging_pixels,
            conv_pixels,
        )

        return [logging_events, urban_conv_events, urban_edge_events]

    def _check_raster_config(self, config):
        if not rasterio_is_available():
            raise ValueError("当前环境缺少 rasterio，不能读取 GeoTIFF 栅格。")

        required_items = [
            ("AGBD", config.agbd_raster_path),
            ("TCC", config.tcc_raster_path),
            ("基准年LULC", config.lulc_base_raster_path),
            ("目标年LULC", config.lulc_target_raster_path),
            ("Drivers", config.drivers_raster_path),
            ("保护区", config.reserve_raster_path),
        ]
        optional_items = []
        for name, path_text in parse_env_raster_paths(config.env_raster_paths):
            if path_exists(path_text):
                optional_items.append(("环境因子-" + name, path_text))

        validate_raster_alignment(required_items + optional_items, "真实栅格模式")

    def _build_raster_reserve_mask(self, config, shape):
        if path_exists(config.reserve_raster_path):
            reserve, _ = read_raster(config.reserve_raster_path)
            if reserve.shape != shape:
                raise ValueError("保护区栅格尺寸必须和 LULC 栅格一致。")
            return reserve == config.reserve_value

        return np.zeros(shape, dtype=bool)

    def _generate_raster_logging_events(self, config, rng, forest_target, reserve_mask):
        # Drivers 确定采伐候选区，采伐斑块库从这些区域中提取。
        if path_exists(config.drivers_raster_path):
            drivers, _ = read_raster(config.drivers_raster_path)
            if drivers.shape != forest_target.shape:
                raise ValueError("Drivers 栅格尺寸必须和 LULC 栅格一致。")
            logging_source = drivers == config.logging_driver_value
        else:
            logging_source = forest_target.copy()

        potential_mask = logging_source & forest_target & (~reserve_mask)
        target_count = int(potential_mask.sum() * (1 - config.logging_area_reduction))

        patch_library = RasterLoggingPatchLibrary()
        patch_library.build_from_mask(potential_mask, config)
        self.last_patch_library = patch_library

        records = patch_library.sample_future_events(config, rng, reserve_mask, target_count)
        return EventTable("logging", records)

    def _generate_raster_urban_conv_events(self, config, rng, forest_base, urban_target, reserve_mask, logging_pixels):
        # 城镇转换基准年是森林、目标年变为城镇的像元。
        conv_mask = forest_base & urban_target & (~reserve_mask)
        rows, cols = np.where(conv_mask)
        raw_count = len(rows)

        if raw_count == 0:
            return EventTable("urban_conv", self._empty_event_frame())

        keep_count = int(raw_count * (1 - config.urban_area_reduction))
        if keep_count < 0:
            keep_count = 0
        if keep_count > raw_count:
            keep_count = raw_count

        if keep_count == 0:
            return EventTable("urban_conv", self._empty_event_frame())

        picked_index = rng.choice(raw_count, size=keep_count, replace=False)
        picked_rows = rows[picked_index]
        picked_cols = cols[picked_index]
        years = self._pick_years(config.future_years, keep_count, config.urban_speed_shift, rng)

        records = []
        for index in range(keep_count):
            row = int(picked_rows[index])
            col = int(picked_cols[index])
            pixel_id = self._build_pixel_id(config, row, col)
            if pixel_id in logging_pixels:
                continue
            records.append(
                {
                    "pixel_id": pixel_id,
                    "row": row,
                    "col": col,
                    "type": "urban_conv",
                    "y_event": int(years[index]),
                }
            )

        if not records:
            return EventTable("urban_conv", self._empty_event_frame())
        return EventTable("urban_conv", pd.DataFrame(records))

    def _generate_raster_urban_edge_events(
        self,
        config,
        conv_df,
        forest_target,
        reserve_mask,
        logging_pixels,
        conv_pixels,
    ):
        # 城镇边缘扰动只是记录新增城镇周边的森林像元。
        edge_year_map = {}
        edge_position_map = {}
        blocked_pixels = set(logging_pixels)
        for pixel_id in conv_pixels:
            blocked_pixels.add(pixel_id)

        for _, item in conv_df.iterrows():
            conv_row = int(item["row"])
            conv_col = int(item["col"])
            conv_year = int(item["y_event"])
            neighbors = self._get_neighbor_cells(conv_row, conv_col, config.grid_rows, config.grid_cols)

            for row, col in neighbors:
                if reserve_mask[row, col]:
                    continue
                if not forest_target[row, col]:
                    continue

                pixel_id = self._build_pixel_id(config, row, col)
                if pixel_id in blocked_pixels:
                    continue

                if pixel_id not in edge_year_map:
                    edge_year_map[pixel_id] = conv_year
                    edge_position_map[pixel_id] = (row, col)
                elif conv_year < edge_year_map[pixel_id]:
                    edge_year_map[pixel_id] = conv_year
                    edge_position_map[pixel_id] = (row, col)

        records = []
        for pixel_id in sorted(edge_year_map.keys()):
            row, col = edge_position_map[pixel_id]
            records.append(
                {
                    "pixel_id": pixel_id,
                    "row": row,
                    "col": col,
                    "type": "urban_edge",
                    "y_event": int(edge_year_map[pixel_id]),
                }
            )

        if not records:
            return EventTable("urban_edge", self._empty_event_frame())
        return EventTable("urban_edge", pd.DataFrame(records))

    def _empty_event_frame(self):
        return pd.DataFrame(columns=["pixel_id", "row", "col", "type", "y_event"])

    def _build_reserve_mask(self, config):
        mask = np.zeros((config.grid_rows, config.grid_cols), dtype=bool)
        reserve_ratio = config.reserve_ratio
        if reserve_ratio <= 0:
            return mask
        reserve_edge_ratio = math.sqrt(reserve_ratio)
        reserve_rows = int(config.grid_rows * reserve_edge_ratio)
        reserve_cols = int(config.grid_cols * reserve_edge_ratio)

        if reserve_rows < 1:
            reserve_rows = 1
        if reserve_cols < 1:
            reserve_cols = 1

        row = 0
        while row < reserve_rows:
            col = 0
            while col < reserve_cols:
                mask[row, col] = True
                col += 1
            row += 1

        right_rows = reserve_rows // 2
        right_cols = reserve_cols // 2
        if right_rows < 1:
            right_rows = 1
        if right_cols < 1:
            right_cols = 1

        start_row = config.grid_rows - right_rows
        start_col = config.grid_cols - right_cols
        row = start_row
        while row < config.grid_rows:
            col = start_col
            while col < config.grid_cols:
                mask[row, col] = True
                col += 1
            row += 1

        return mask

    def _build_pixel_id_set(self, records):
        pixel_ids = set()
        if records.empty:
            return pixel_ids

        for pixel_id in records["pixel_id"].tolist():
            pixel_ids.add(int(pixel_id))
        return pixel_ids

    def _build_pixel_id(self, config, row, col):
        return row * config.grid_cols + col

    def _get_neighbor_cells(self, row, col, max_rows, max_cols):
        neighbors = []
        for row_offset in [-1, 0, 1]:
            for col_offset in [-1, 0, 1]:
                if row_offset == 0 and col_offset == 0:
                    continue

                next_row = row + row_offset
                next_col = col + col_offset
                if next_row < 0 or next_row >= max_rows:
                    continue
                if next_col < 0 or next_col >= max_cols:
                    continue

                neighbors.append((next_row, next_col))
        return neighbors

    def _pick_years(self, future_years, count, speed_shift, rng):
        if count <= 0:
            return []

        weights = self._build_year_weights(future_years, speed_shift)
        raw_years = rng.choice(future_years, size=count, replace=True, p=weights)
        picked_years = []
        for raw_year in raw_years:
            picked_years.append(int(raw_year))
        return picked_years

    def _build_year_weights(self, future_years, speed_shift):
        if not future_years:
            raise ValueError("未来年份列表为空，请把目标年设置为大于基准年的数值。")

        if len(future_years) == 1:
            return [1.0]

        weights = []
        last_index = len(future_years) - 1
        index = 0
        while index < len(future_years):
            progress = index / last_index
            weight = 1.0
            if speed_shift > 0:
                weight = weight + progress * speed_shift * 3
            elif speed_shift < 0:
                weight = weight + (1 - progress) * abs(speed_shift) * 3

            if weight < 0.1:
                weight = 0.1

            weights.append(weight)
            index += 1

        total = sum(weights)
        normalized_weights = []
        for weight in weights:
            normalized_weights.append(weight / total)
        return normalized_weights

    def _check_config(self, config):
        if config.target_year <= config.base_year:
            raise ValueError("目标年必须大于基准年。")
        if not config.future_years:
            raise ValueError("未来年份列表为空，请重新保存情景配置。")
        if config.grid_rows <= 0 or config.grid_cols <= 0:
            raise ValueError("网格行列数必须大于 0。")
        if config.reserve_ratio < 0 or config.reserve_ratio >= 0.6:
            raise ValueError("保育区比例请设置在 0 到 0.6 之间。")
        if config.logging_patch_min_size <= 0:
            raise ValueError("采伐斑块最小尺寸必须大于 0。")
        if config.logging_patch_max_size < config.logging_patch_min_size:
            raise ValueError("采伐斑块最大尺寸不能小于最小尺寸。")
        if config.logging_library_years <= 0:
            raise ValueError("采伐斑块库历史年份数必须大于 0。")
        if config.logging_library_patch_count <= 0:
            raise ValueError("采伐斑块库样本数必须大于 0。")


class LoggingPatch:
    def __init__(self, patch_id, source_year, row_offsets, col_offsets):
        self.patch_id = int(patch_id)
        self.source_year = int(source_year)
        self.row_offsets = list(row_offsets)
        self.col_offsets = list(col_offsets)
        self.size = len(self.row_offsets)

    def to_dict(self):
        return {
            "patch_id": self.patch_id,
            "source_year": self.source_year,
            "size": self.size,
            "row_offsets": list(self.row_offsets),
            "col_offsets": list(self.col_offsets),
        }


class LoggingPatchLibrary:
    def __init__(self):
        self.patches = []

    def sample_future_events(self, config, rng, reserve_mask, target_count):
        records = []
        used_pixels = set()
        patch_id = 1
        target_count = min(max(int(target_count), 0), self._available_cell_count(reserve_mask))
        if target_count <= 0:
            return pd.DataFrame(records)

        years = self._pick_future_years(config, rng, target_count)
        max_attempts = max(target_count * 30, len(self.patches) * 20, 300)
        attempts = 0

        while len(records) < target_count and attempts < max_attempts:
            attempts += 1
            future_year = self._pick_one_future_year(config, years, rng)
            patch = self._sample_patch_template(rng)
            if patch is None:
                break

            placed_cells = self._place_patch(config, rng, patch, reserve_mask, used_pixels)
            if not placed_cells:
                continue

            for row, col in placed_cells:
                pixel_id = row * config.grid_cols + col
                records.append(
                    {
                        "pixel_id": pixel_id,
                        "row": row,
                        "col": col,
                        "type": "logging",
                        "y_event": future_year,
                        "patch_id": patch_id,
                        "patch_size": len(placed_cells),
                        "source_patch_id": patch.patch_id,
                        "source_year": patch.source_year,
                    }
                )
                if len(records) >= target_count:
                    break

            patch_id += 1

        if len(records) < target_count:
            raise ValueError(
                "采伐事件生成失败：可用网格不足或斑块尺寸过大，"
                f"目标像元 {target_count}，已放置 {len(records)}。"
            )
        return pd.DataFrame(records)

    def _available_cell_count(self, reserve_mask):
        if reserve_mask is None:
            return 0
        return int(reserve_mask.size - reserve_mask.sum())

    def _pick_future_years(self, config, rng, target_count):
        if target_count <= 0:
            return []

        weights = self._build_year_weights(config.future_years)
        raw_years = rng.choice(config.future_years, size=target_count, replace=True, p=weights)
        years = []
        for year in raw_years:
            years.append(int(year))
        return years

    def _build_year_weights(self, future_years):
        weights = []
        for _ in future_years:
            weights.append(1.0)

        total = sum(weights)
        normalized_weights = []
        for weight in weights:
            normalized_weights.append(weight / total)
        return normalized_weights

    def _pick_one_future_year(self, config, years, rng):
        if years:
            index = int(rng.integers(0, len(years)))
            return int(years[index])
        return int(config.base_year + 1)

    def _sample_patch_template(self, rng):
        if not self.patches:
            return None

        sizes = self.get_patch_sizes()
        total_size = float(sum(sizes))
        weights = []
        for size in sizes:
            weights.append(float(size) / total_size)

        patch_indices = np.arange(len(self.patches))
        index = int(rng.choice(patch_indices, p=weights))
        return self.patches[index]

    def _place_patch(self, config, rng, patch, reserve_mask, used_pixels):
        try_count = 0
        while try_count < 200:
            seed_row = int(rng.integers(0, config.grid_rows))
            seed_col = int(rng.integers(0, config.grid_cols))
            placed_cells = self._convert_offsets_to_cells(config, patch, seed_row, seed_col, reserve_mask, used_pixels)
            if placed_cells:
                return placed_cells
            try_count += 1
        return []

    def _convert_offsets_to_cells(self, config, patch, seed_row, seed_col, reserve_mask, used_pixels):
        cells = []
        index = 0
        while index < patch.size:
            row = seed_row + patch.row_offsets[index]
            col = seed_col + patch.col_offsets[index]
            if row < 0 or row >= config.grid_rows:
                return []
            if col < 0 or col >= config.grid_cols:
                return []
            if reserve_mask[row, col]:
                return []

            pixel_id = row * config.grid_cols + col
            if pixel_id in used_pixels:
                return []

            cells.append((row, col))
            index += 1

        for row, col in cells:
            pixel_id = row * config.grid_cols + col
            used_pixels.add(pixel_id)
        return cells

    def get_patch_sizes(self):
        sizes = []
        for patch in self.patches:
            sizes.append(patch.size)
        return sizes

    def summary(self):
        sizes = self.get_patch_sizes()
        if not sizes:
            return {
                "patch_count": 0,
                "history_year_min": None,
                "history_year_max": None,
                "size_min": None,
                "size_mean": None,
                "size_max": None,
            }

        years = []
        for patch in self.patches:
            years.append(patch.source_year)

        return {
            "patch_count": len(self.patches),
            "history_year_min": min(years),
            "history_year_max": max(years),
            "size_min": min(sizes),
            "size_mean": round(float(np.mean(sizes)), 2),
            "size_max": max(sizes),
        }

    def to_frame(self):
        rows = []
        for patch in self.patches:
            rows.append(
                {
                    "patch_id": patch.patch_id,
                    "source_year": patch.source_year,
                    "size": patch.size,
                    "row_offsets": str(patch.row_offsets),
                    "col_offsets": str(patch.col_offsets),
                }
            )
        return pd.DataFrame(rows)


class RasterLoggingPatchLibrary(LoggingPatchLibrary):
    def build_from_mask(self, potential_mask, config):
        self.patches = []
        rng = np.random.default_rng(config.base_seed + 17)
        components = self._find_patch_components(potential_mask, config)

        if not components:
            return self

        if len(components) > config.logging_library_patch_count:
            picked = rng.choice(len(components), size=config.logging_library_patch_count, replace=False)
            picked = sorted(picked.tolist())
        else:
            picked = list(range(len(components)))

        patch_id = 1
        for component_index in picked:
            cells = components[component_index]
            cells = self._trim_component_cells(cells, config, rng)
            if not cells:
                continue

            base_row, base_col = cells[0]
            row_offsets = []
            col_offsets = []
            for row, col in cells:
                row_offsets.append(int(row - base_row))
                col_offsets.append(int(col - base_col))

            source_year = config.base_year - int(rng.integers(0, max(config.logging_library_years, 1)))
            patch = LoggingPatch(patch_id, source_year, row_offsets, col_offsets)
            self.patches.append(patch)
            patch_id = patch_id + 1

        return self

    def _find_patch_components(self, mask, config):
        rows, cols = mask.shape
        visited = np.zeros(mask.shape, dtype=bool)
        components = []

        for row in range(rows):
            for col in range(cols):
                if visited[row, col]:
                    continue
                if not mask[row, col]:
                    visited[row, col] = True
                    continue

                cells = self._grow_one_component(mask, visited, row, col, rows, cols, config)
                if len(cells) >= config.logging_patch_min_size:
                    components.append(cells)

        return components

    def _grow_one_component(self, mask, visited, start_row, start_col, rows, cols, config):
        stack = [(start_row, start_col)]
        visited[start_row, start_col] = True
        cells = []
        max_keep = max(config.logging_patch_max_size * 4, config.logging_patch_max_size)

        while stack:
            row, col = stack.pop()
            cells.append((row, col))

            if len(cells) >= max_keep:
                continue

            for row_delta, col_delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_row = row + row_delta
                next_col = col + col_delta
                if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                    continue
                if visited[next_row, next_col]:
                    continue
                visited[next_row, next_col] = True
                if mask[next_row, next_col]:
                    stack.append((next_row, next_col))

        return cells

    def _trim_component_cells(self, cells, config, rng):
        if len(cells) <= config.logging_patch_max_size:
            return cells

        indexes = np.arange(len(cells))
        rng.shuffle(indexes)
        picked = indexes[: config.logging_patch_max_size]
        picked = sorted(picked.tolist())
        return [cells[index] for index in picked]


class SeverityEngine:
    def assign_all(self, event_tables, config):
        return [self.assign(table, config) for table in event_tables]

    def assign(self, event_table, config):
        records = event_table.records.copy()
        if records.empty:
            return EventTable(event_table.event_type, records)

        if "Severity" not in records.columns:
            records["Severity"] = self._build_severity(records, event_table.event_type, config)

        records["Severity"] = records["Severity"].astype(float).clip(0.0, 1.0)
        if "Severity_Class" not in records.columns:
            records["Severity_Class"] = records["Severity"].apply(self._classify)
        return EventTable(event_table.event_type, records)

    def _build_severity(self, records, event_type, config):
        if getattr(config, "use_history_training", False) and getattr(config, "use_raster_data", False):
            values = self._sample_empirical_severity(records, event_type, config)
            if values is not None:
                return values
            raise ValueError("历史扰动强度样本为空：请检查历史TCC、Drivers、历史土地利用和环境因子栅格。")
        raise ValueError("简化随机扰动强度已删除，请使用历史数据计算经验强度分布。")

    def _sample_empirical_severity(self, records, event_type, config):
        distribution = self._build_empirical_distribution(event_type, config)
        if distribution is None or distribution.empty:
            return None

        rng = np.random.default_rng(int(config.base_seed) + self._severity_seed_offset(event_type))
        future_features = self._build_future_severity_features(records, config)
        picked = self._sample_by_strata(distribution, future_features, rng)
        if str(getattr(config, "severity_method", "S1")).upper() == "S2":
            picked = self._adjust_severity_by_environment(picked, future_features)
        if event_type in {"urban_conv", "urban_conversion"}:
            picked = np.maximum(picked, 0.62)
        if event_type == "urban_edge":
            picked = picked * (1.0 - float(config.urban_severity_reduction))
        if event_type == "logging":
            picked = picked * (1.0 - float(config.logging_severity_reduction))
            cap_quantile = getattr(config, "logging_severity_cap_quantile", None)
            if cap_quantile is not None:
                cap_value = float(np.quantile(distribution["Severity"].to_numpy(dtype=np.float32), float(cap_quantile)))
                picked = np.minimum(picked, cap_value)
        return np.clip(picked, 0.0, 0.95)

    def _build_empirical_distribution(self, event_type, config):
        tcc_paths = parse_year_raster_paths(getattr(config, "history_tcc_paths", ""))
        lulc_paths = parse_year_raster_paths(getattr(config, "history_lulc_paths", ""))
        years = sorted(set(tcc_paths.keys()) & set(lulc_paths.keys()))
        if len(years) < 2:
            return None

        rng = np.random.default_rng(int(config.base_seed) + self._severity_seed_offset(event_type) + 90)
        forest_codes = parse_code_list(config.forest_lulc_codes, [1, 2, 3, 4, 5])
        urban_codes = parse_code_list(config.urban_lulc_codes, [8, 9])
        samples = []
        max_samples = max(200, int(getattr(config, "severity_sample_count", 4000)))

        drivers_class = None
        drivers_loss_year = None
        if event_type == "logging" and path_exists(getattr(config, "drivers_raster_path", "")):
            try:
                drivers_class, _ = read_raster_band(config.drivers_raster_path, 1)
                drivers_loss_year, _ = read_raster_band(config.drivers_raster_path, 4)
            except Exception:
                drivers_class = None
                drivers_loss_year = None

        env_surfaces = None

        for index in range(len(years) - 1):
            start_year = years[index]
            end_year = years[index + 1]
            if int(end_year) - int(start_year) != 1:
                continue
            if not path_exists(tcc_paths[start_year]) or not path_exists(tcc_paths[end_year]):
                continue
            tcc_start, _ = read_raster(tcc_paths[start_year], make_float=True)
            tcc_end, _ = read_raster(tcc_paths[end_year], make_float=True)
            tcc_start = self._normalize_tcc(tcc_start)
            tcc_end = self._normalize_tcc(tcc_end)
            mask = np.isfinite(tcc_start) & np.isfinite(tcc_end)
            if env_surfaces is None:
                env_surfaces = self._load_severity_env_surfaces(config, tcc_start.shape)

            if event_type == "logging":
                if drivers_class is None or drivers_class.shape != tcc_start.shape:
                    continue
                mask = mask & (drivers_class == int(config.logging_driver_value))
                if drivers_loss_year is not None and drivers_loss_year.shape == tcc_start.shape:
                    encoded_year = drivers_loss_year.astype(np.int32) + 2000
                    mask = mask & (encoded_year == int(end_year))
            else:
                if start_year not in lulc_paths or end_year not in lulc_paths:
                    continue
                if not path_exists(lulc_paths[start_year]) or not path_exists(lulc_paths[end_year]):
                    continue
                lulc_start, _ = read_raster(lulc_paths[start_year])
                lulc_end, _ = read_raster(lulc_paths[end_year])
                if lulc_start.shape != tcc_start.shape or lulc_end.shape != tcc_start.shape:
                    continue
                conv_mask = np.isin(lulc_start, forest_codes) & np.isin(lulc_end, urban_codes)
                if event_type in {"urban_conv", "urban_conversion"}:
                    mask = mask & conv_mask
                else:
                    edge_mask = self._build_edge_mask(conv_mask, np.isin(lulc_end, forest_codes))
                    mask = mask & edge_mask

            row_ids, col_ids = np.where(mask)
            if len(row_ids) == 0:
                continue
            keep_count = min(len(row_ids), max_samples - len(samples))
            if keep_count <= 0:
                break
            picked = rng.choice(len(row_ids), size=keep_count, replace=False)
            for picked_index in picked:
                row = int(row_ids[picked_index])
                col = int(col_ids[picked_index])
                severity = self._calculate_severity_value(tcc_start[row, col], tcc_end[row, col])
                if severity <= 0:
                    continue
                item = {
                    "Severity": severity,
                    "TCC_pre": float(tcc_start[row, col]),
                }
                self._add_env_value_to_severity_item(item, env_surfaces, row, col)
                samples.append(item)
            if len(samples) >= max_samples:
                break

        if not samples:
            return None
        return pd.DataFrame(samples)

    def _load_severity_env_surfaces(self, config, shape):
        env_items = parse_env_raster_paths(getattr(config, "env_raster_paths", ""))
        env_map = {}
        for name, path_text in env_items:
            if not path_exists(path_text):
                continue
            data, _ = read_raster(path_text, make_float=True)
            if data.shape != shape:
                continue
            env_map[name] = self._normalize_env_surface(data)

        result = {}
        result["slope"] = self._find_env_surface(env_map, ["slope", "坡度", "terrain"])
        result["moisture"] = self._find_env_surface(env_map, ["moisture", "MAP", "AET", "降水", "水分"])
        result["accessibility"] = self._find_env_surface(env_map, ["accessibility", "Dist", "Road", "道路", "人为"])
        return result

    def _find_env_surface(self, env_map, names):
        for key in env_map:
            text = str(key).lower()
            for name in names:
                if str(name).lower() in text:
                    return env_map[key]
        raise ValueError("环境因子栅格不完整：地形因子、气候水分因子和人为活动因子均为必填。")

    def _normalize_env_surface(self, surface):
        data = surface.astype(np.float32).copy()
        low = float(np.nanpercentile(data, 2))
        high = float(np.nanpercentile(data, 98))
        if high <= low:
            high = low + 1.0
        data = (data - low) / (high - low)
        return np.clip(data, 0.0, 1.0)

    def _add_env_value_to_severity_item(self, item, env_surfaces, row, col):
        for name in ["slope", "moisture", "accessibility"]:
            item[name] = float(env_surfaces[name][row, col])

    def _build_future_severity_features(self, records, config):
        tcc, _ = read_raster(config.tcc_raster_path, make_float=True)
        tcc = self._normalize_tcc(tcc)
        env_surfaces = self._load_severity_env_surfaces(config, tcc.shape)
        rows = records["row"].to_numpy(dtype=int)
        cols = records["col"].to_numpy(dtype=int)
        items = []
        for index in range(len(records)):
            row = int(rows[index])
            col = int(cols[index])
            item = {"TCC_pre": float(tcc[row, col])}
            self._add_env_value_to_severity_item(item, env_surfaces, row, col)
            items.append(item)
        return pd.DataFrame(items)

    def _sample_by_strata(self, distribution, future_features, rng):
        strat_cols = ["TCC_pre", "slope", "moisture", "accessibility"]
        edges = {}
        dist = distribution.copy()
        future = future_features.copy()
        for col in strat_cols:
            col_edges = np.nanquantile(dist[col].to_numpy(dtype=np.float32), np.linspace(0, 1, 5))
            col_edges = np.unique(col_edges)
            edges[col] = col_edges
            dist[col + "_bin"] = np.digitize(dist[col].to_numpy(dtype=np.float32), col_edges[1:-1])
            future[col + "_bin"] = np.digitize(future[col].to_numpy(dtype=np.float32), col_edges[1:-1])

        result = []
        for index in range(len(future)):
            mask = np.ones(len(dist), dtype=bool)
            for col in strat_cols:
                mask = mask & (dist[col + "_bin"].to_numpy(dtype=int) == int(future.iloc[index][col + "_bin"]))
            pool = dist.loc[mask, "Severity"].to_numpy(dtype=np.float32)
            if len(pool) == 0:
                tcc_mask = dist["TCC_pre_bin"].to_numpy(dtype=int) == int(future.iloc[index]["TCC_pre_bin"])
                pool = dist.loc[tcc_mask, "Severity"].to_numpy(dtype=np.float32)
            if len(pool) == 0:
                pool = dist["Severity"].to_numpy(dtype=np.float32)
            result.append(float(rng.choice(pool)))
        return np.asarray(result, dtype=np.float32)

    def _adjust_severity_by_environment(self, severity_values, future_features):
        adjusted = severity_values.astype(np.float32).copy()
        for index in range(len(adjusted)):
            slope = float(future_features.iloc[index].get("slope", 0.5))
            moisture = float(future_features.iloc[index].get("moisture", 0.5))
            accessibility = float(future_features.iloc[index].get("accessibility", 0.5))

            pressure = 0.0
            pressure = pressure + (accessibility - 0.5) * 0.18
            pressure = pressure + (slope - 0.5) * 0.08
            pressure = pressure + (0.5 - moisture) * 0.10
            factor = 1.0 + pressure
            adjusted[index] = adjusted[index] * factor

        return np.clip(adjusted, 0.0, 0.95)

    def _normalize_tcc(self, surface):
        result = surface.astype(np.float32).copy()
        max_value = np.nanmax(result)
        if max_value > 1.5:
            result = result / 100.0
        return np.clip(result, 0.0, 1.0)

    def _calculate_severity_value(self, before, after):
        before = float(before)
        after = float(after)
        if before <= 0.05:
            return 0.0
        return float(np.clip((before - after) / before, 0.0, 0.95))

    def _build_edge_mask(self, conv_mask, forest_mask):
        rows, cols = conv_mask.shape
        result = np.zeros(conv_mask.shape, dtype=bool)
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
                        result[next_row, next_col] = True
        return result

    def _severity_seed_offset(self, event_type):
        return {
            "logging": 301,
            "urban_conv": 401,
            "urban_conversion": 401,
            "urban_edge": 501,
        }.get(event_type, 601)

    def _classify(self, value):
        value = float(value)
        if value < 0.33:
            return "low"
        if value < 0.66:
            return "medium"
        return "high"
