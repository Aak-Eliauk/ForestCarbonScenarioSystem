import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from fcscs.config.defaults import sanitize_scenario_name
from fcscs.engines.raster_tools import resolve_output_dir
from fcscs.services.quick_run_service import build_quick_config


STATUS_FILE_NAME = "run_status.json"


# 后台运行服务负责创建批次目录、启动子进程并记录状态


def get_batch_output_dir(config, create=True):
    batch_name = sanitize_scenario_name(config.batch_name, default="运行批次")
    batch_dir = resolve_output_dir(config.output_dir) / batch_name
    if create:
        batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def start_background_run(config, run_mode, quick_size):
    run_config = config
    if run_mode == "快速测试":
        run_config = build_quick_config(config, quick_size)

    batch_dir = get_batch_output_dir(run_config, create=True)
    log_dir = batch_dir / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    config_path = batch_dir / "run_config.yaml"
    status_path = batch_dir / STATUS_FILE_NAME
    run_log_path = log_dir / "run_events.log"
    run_config.save_yaml(config_path)

    write_status(
        status_path,
        {
            "state": "starting",
            "percent": 1,
            "stage": "启动后台进程",
            "message": "后台计算进程正在启动。",
            "batch_name": run_config.batch_name,
            "scenario_name": run_config.scenario_name,
            "run_mode": run_mode,
            "config_path": str(config_path),
            "batch_dir": str(batch_dir),
            "run_log_path": str(run_log_path),
        },
    )

    # 子进程需要能找到src目录下的fcscs包，所以这里补上PYTHONPATH
    project_root = Path(__file__).resolve().parents[3]
    src_dir = project_root / "src"
    env = os.environ.copy()
    old_python_path = env.get("PYTHONPATH", "")
    if old_python_path:
        env["PYTHONPATH"] = str(src_dir) + os.pathsep + old_python_path
    else:
        env["PYTHONPATH"] = str(src_dir)

    command = [
        sys.executable,
        "-m",
        "fcscs.services.run_worker",
        str(config_path),
        str(status_path),
        str(run_mode),
    ]

    log_file = open(run_log_path, "a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(project_root),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=_creation_flags(),
    )
    log_file.close()

    status = read_status(status_path)
    status["pid"] = int(process.pid)
    status["state"] = "running"
    status["message"] = "后台计算进程已启动，可以刷新页面后继续查看进度。"
    write_status(status_path, status)

    return {
        "config": run_config,
        "status_path": status_path,
        "batch_dir": batch_dir,
        "run_log_path": run_log_path,
        "pid": int(process.pid),
    }


def write_status(status_path, status):
    status_path = Path(status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(status)
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_status_file(status_path, data)
    _append_run_log(data)


def _save_status_file(status_path, data):
    temp_path = status_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temp_path.replace(status_path)


def _append_run_log(status):
    log_path_text = status.get("run_log_path")
    if not log_path_text:
        return

    try:
        log_path = Path(log_path_text)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = _build_log_line(status)
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(line + "\n")
            error_text = str(status.get("error", "")).strip()
            if error_text:
                file.write(error_text + "\n")
    except Exception:
        return


def _build_log_line(status):
    time_text = str(status.get("updated_at", ""))
    state_text = str(status.get("state", ""))
    percent_text = str(status.get("percent", ""))
    stage_text = str(status.get("stage", ""))
    message_text = str(status.get("message", ""))
    return time_text + " | " + state_text + " | " + percent_text + "% | " + stage_text + " | " + message_text


def read_status(status_path):
    status_path = Path(status_path)
    if not status_path.exists():
        return {}
    try:
        with open(status_path, "r", encoding="utf-8") as file:
            status = json.load(file)
    except Exception:
        return {}
    return _mark_dead_process_as_failed(status_path, status)


def _mark_dead_process_as_failed(status_path, status):
    state = str(status.get("state", ""))
    if state not in {"starting", "running"}:
        return status

    pid = status.get("pid")
    if pid is None:
        return status
    if _process_is_alive(pid):
        return status

    status["state"] = "failed"
    status["stage"] = "运行异常结束"
    status["message"] = "后台进程已经退出，但没有写入完成状态，请查看日志文件中的错误信息。"
    status["error"] = str(status.get("error", "") or "后台进程异常退出。")
    status["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_status_file(status_path, status)
    _append_run_log(status)
    return status


def _process_is_alive(pid):
    try:
        pid = int(pid)
    except Exception:
        return True

    if os.name == "nt":
        try:
            import ctypes

            access = 0x1000
            still_active = 259
            handle = ctypes.windll.kernel32.OpenProcess(access, False, pid)
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            return bool(ok) and int(exit_code.value) == still_active
        except Exception:
            return True

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_recent_jobs(output_dir, limit=10):
    # 输出目录下每个批次都有一个run_status.json
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []

    jobs = []
    for status_path in output_dir.glob("*/" + STATUS_FILE_NAME):
        status = read_status(status_path)
        if not status:
            continue
        status["status_path"] = str(status_path)
        status["batch_dir"] = str(status_path.parent)
        jobs.append(status)

    jobs = sorted(jobs, key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return jobs[:limit]


def terminate_background_run(status_path):
    # 终止时优先杀掉记录的进程号，再把状态写回文件
    status_path = Path(status_path)
    status = read_status(status_path)
    if not status:
        return False, "没有找到该后台任务状态。"

    state = str(status.get("state", ""))
    if state in {"finished", "failed", "stopped"}:
        return False, "该任务已经结束。"

    pid = status.get("pid")
    if pid is None:
        status["state"] = "stopped"
        status["percent"] = int(status.get("percent", 0))
        status["stage"] = "已终止"
        status["message"] = "未记录进程号，已将任务标记为终止。"
        write_status(status_path, status)
        return True, "已将任务标记为终止。"

    try:
        pid = int(pid)
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception as error:
        return False, "终止进程失败：" + str(error)

    status["state"] = "stopped"
    status["stage"] = "已终止"
    status["message"] = "用户已终止本次后台运行。"
    write_status(status_path, status)
    return True, "后台任务已终止。"


def _creation_flags():
    if os.name == "nt":
        return subprocess.CREATE_NO_WINDOW
    return 0
