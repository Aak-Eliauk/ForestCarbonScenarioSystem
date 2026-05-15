from fcscs.engines.raster_tools import resolve_output_dir
from fcscs.config.defaults import clean_name


def get_output_directory(config, create=True):
    output_dir = resolve_output_dir(config.output_dir)
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_batch_output_directory(config, create=True):
    output_dir = get_output_directory(config, create=create)
    batch_name = config.batch_name
    batch_dir = output_dir / clean_name(batch_name, default="运行批次")
    if create:
        batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir
