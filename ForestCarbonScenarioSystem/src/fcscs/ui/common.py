from fcscs.engines.raster_tools import get_out_dir
from fcscs.config.defaults import name_clean


def get_out_directory(config, create=True):
    out_dir = get_out_dir(config.output_dir)
    if create:
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_batch_out_directory(config, create=True):
    out_dir = get_out_directory(config, create=create)
    batch_name = config.batch_name
    batch_dir = out_dir / name_clean(batch_name, default="运行批次")
    if create:
        batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir
