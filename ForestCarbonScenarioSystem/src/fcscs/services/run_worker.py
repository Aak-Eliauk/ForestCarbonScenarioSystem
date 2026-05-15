import sys
import traceback

from fcscs.config.defaults import ScenarioConfig
from fcscs.services.background_run_service import get_batch_output_dir, read_status, write_status
from fcscs.services.workflow_service import run_simulation_workflow
from fcscs.ui.result_views import export_report


def main():
    if len(sys.argv) < 4:
        raise SystemExit("参数不足：需要配置文件、状态文件和运行方式。")

    config_path = sys.argv[1]
    status_path = sys.argv[2]
    run_mode = sys.argv[3]

    try:
        config = ScenarioConfig.from_yaml(config_path)
        _update(status_path, "running", 5, "读取配置", "后台进程已读取运行配置。")

        result = run_simulation_workflow(
            config,
            progress_callback=lambda percent, stage, message: _update(
                status_path,
                "running",
                percent,
                stage,
                message,
            ),
        )

        _update(status_path, "running", 92, "保存结果", "正在写出报表和结果索引。")
        batch_dir = get_batch_output_dir(config, create=True)
        export_dir = batch_dir / "report_exports"
        export_report(result.report_bundle, export_dir)

        status = read_status(status_path)
        status["state"] = "finished"
        status["percent"] = 100
        status["stage"] = "完成"
        status["message"] = "后台运行完成，结果已保存。"
        status["run_mode"] = run_mode
        status["batch_name"] = config.batch_name
        status["scenario_name"] = config.scenario_name
        status["report_dir"] = str(export_dir)
        status["raster_dir"] = str(batch_dir / "raster_predictions")
        status["output_files"] = result.report_bundle.output_files
        write_status(status_path, status)
        return 0
    except Exception as error:
        status = read_status(status_path)
        status["state"] = "failed"
        status["percent"] = 0
        status["stage"] = "运行失败"
        status["message"] = str(error)
        status["error"] = traceback.format_exc()
        write_status(status_path, status)
        return 1


def _update(status_path, state, percent, stage, message):
    status = read_status(status_path)
    status["state"] = state
    status["percent"] = int(percent)
    status["stage"] = str(stage)
    status["message"] = str(message)
    write_status(status_path, status)


if __name__ == "__main__":
    raise SystemExit(main())
