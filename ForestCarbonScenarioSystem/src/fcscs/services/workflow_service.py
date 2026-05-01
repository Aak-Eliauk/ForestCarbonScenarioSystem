from dataclasses import dataclass

from fcscs.engines.monte_carlo_engine import MonteCarloEngine, ReportEngine
from fcscs.engines.scenario_engine import ScenarioEngine, SeverityEngine


@dataclass
class WorkflowResult:
    events: list
    patch_library: object
    simulation_bundle: object
    report_bundle: object


def run_simulation_workflow(config, progress_callback=None):
    scenario_engine = ScenarioEngine()
    severity_engine = SeverityEngine()
    monte_carlo_engine = MonteCarloEngine()
    report_engine = ReportEngine()

    _notify(progress_callback, 18, "生成事件", "正在生成采伐、城镇转换和边缘扰动事件。")
    raw_events = scenario_engine.generate_all_events(config)
    patch_library = scenario_engine.last_patch_library

    _notify(progress_callback, 36, "计算强度", "正在为扰动事件计算强度。")
    severity_events = severity_engine.assign_all(raw_events, config)

    _notify(progress_callback, 58, "模型预测", "正在训练模型并执行蒙特卡洛模拟。")
    bundle = monte_carlo_engine.run(severity_events, config)

    _notify(progress_callback, 86, "汇总结果", "正在汇总 AGBD、AGC 和不确定性结果。")
    report = report_engine.build_report(bundle)

    return WorkflowResult(
        events=severity_events,
        patch_library=patch_library,
        simulation_bundle=bundle,
        report_bundle=report,
    )


def _notify(callback, percent, stage, message):
    if callback is not None:
        callback(percent, stage, message)
