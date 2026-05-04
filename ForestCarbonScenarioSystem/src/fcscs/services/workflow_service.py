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

    _notify(progress_callback, 36, "经验强度采样", "正在根据历史经验分布或默认分布为扰动事件抽取强度。")
    severity_events = severity_engine.assign_all(raw_events, config)

    _notify(progress_callback, 52, "训练模型", "正在构造训练样本并训练 AGBD 响应模型。")
    _notify(progress_callback, 66, "蒙特卡洛模拟", "正在执行多次未来情景模拟并预测 AGBD/AGC。")
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
