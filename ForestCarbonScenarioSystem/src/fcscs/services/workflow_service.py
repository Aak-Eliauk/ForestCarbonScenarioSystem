from dataclasses import dataclass

from fcscs.engines.monte_carlo_engine import MonteCarloEngine, ReportEngine
from fcscs.engines.scenario_engine import ScenarioEngine, SeverityEngine


# 主流程只负责串联五个阶段，具体算法放在engines目录。


@dataclass
class WorkflowResult:
    report_bundle: object


def run_simulation_workflow(config, progress_callback=None):
    scenario_engine = ScenarioEngine()
    severity_engine = SeverityEngine()
    monte_carlo_engine = MonteCarloEngine()
    report_engine = ReportEngine()

    # 根据LULC、Drivers和保护区生成未来扰动事件。
    _notify(progress_callback, 18, "生成事件", "正在生成采伐、城镇转换和边缘扰动事件。")
    raw_events = scenario_engine.generate_all_events(config)

    # 根据历史树冠覆盖度变化（即损失强度），结合环境因子生成经验分布，给每个事件抽取扰动强度。
    _notify(progress_callback, 36, "经验强度采样", "正在准备历史扰动强度样本。")
    severity_events = severity_engine.assign_all(raw_events, config)

    # 训练AGBD模型，并进行多次蒙特卡洛预测。
    _notify(progress_callback, 52, "训练模型", "正在构造训练样本并训练 AGBD 响应模型。")
    _notify(progress_callback, 66, "蒙特卡洛模拟", "正在执行多次未来情景模拟并预测 AGBD/AGC。")
    bundle = monte_carlo_engine.run(severity_events, config)

    # 把模拟结果整理成界面和CSV都能读取的报表对象。
    _notify(progress_callback, 86, "汇总结果", "正在汇总 AGBD、AGC 和不确定性结果。")
    report = report_engine.build_report(bundle)

    return WorkflowResult(
        report_bundle=report,
    )


def _notify(callback, percent, stage, message):
    if callback is not None:
        callback(percent, stage, message)
