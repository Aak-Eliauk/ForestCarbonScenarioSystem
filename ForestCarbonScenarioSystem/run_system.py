import importlib.util
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
APP_FILE = PROJECT_ROOT / "app.py"
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_OUTPUT_DIR = (PROJECT_ROOT.parent / "ForestCarbonScenarioSystem_outputs").resolve()
HOST = "localhost"
PORT = 8501
SOFTWARE_FULL_NAME = "森林损失情景控制与蒙特卡洛碳储量模拟系统 V1.0"
SOFTWARE_SHORT_NAME = "森林损失控制碳模拟系统"


def main():
    sys.dont_write_bytecode = True
    print_title()
    check_project_files()
    check_python_version()
    check_required_packages()
    add_src_to_pythonpath()
    port = pick_available_port(PORT)
    url = build_url(port)

    if "--check" in sys.argv:
        print(f"默认输出目录: {DEFAULT_OUTPUT_DIR}")
        print("启动检查完成。")
        return

    print_run_explain(port, url)
    start_browser_thread(port, url)
    run_streamlit_server(port)


def print_title():
    print("")
    print(SOFTWARE_FULL_NAME)
    print("=" * 48)
    print(f"简称: {SOFTWARE_SHORT_NAME}")
    print("这个窗口就是系统的运行终端，请不要关闭。")
    print("")


def check_project_files():
    if not APP_FILE.exists():
        print("未找到 app.py，请确认脚本放在项目根目录。")
        pause_and_exit(1)

    if not SRC_DIR.exists():
        print("未找到 src 目录，请确认项目文件完整。")
        pause_and_exit(1)


def check_python_version():
    major = sys.version_info.major
    minor = sys.version_info.minor
    print(f"Python 版本: {major}.{minor}")
    if major < 3:
        print("当前 Python 版本过低，请使用 Python 3。")
        pause_and_exit(1)


def check_required_packages():
    package_names = [
        "streamlit",
        "numpy",
        "pandas",
        "yaml",
        "sklearn",
        "rasterio",
    ]

    missing_packages = []
    for package_name in package_names:
        if importlib.util.find_spec(package_name) is None:
            missing_packages.append(package_name)

    if missing_packages:
        print("")
        print("缺少运行依赖：")
        for package_name in missing_packages:
            print(f"  - {package_name}")
        print("")
        print("请先在当前目录运行：")
        print("python -m pip install -r requirements.txt")
        pause_and_exit(1)

    print("依赖检查: 通过")


def add_src_to_pythonpath():
    src_text = str(SRC_DIR)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


def print_run_explain(port, url):
    print("")
    print("启动方式说明：")
    print("1. Streamlit 是一个 Python 进程。")
    print("2. 这个进程同时负责网页界面和后台计算。")
    print("3. 浏览器只是访问界面，真正的事件生成、机器学习和蒙特卡洛计算都在这个终端进程里执行。")
    print("")
    if port != PORT:
        print(f"默认端口 {PORT} 已被占用，本次改用端口 {port}。")
    print(f"正在启动系统: {url}")
    print(f"默认输出目录: {DEFAULT_OUTPUT_DIR}")
    print("")


def start_browser_thread(port, url):
    thread = threading.Thread(target=open_browser_when_ready, args=(port, url))
    thread.daemon = True
    thread.start()


def open_browser_when_ready(port, url):
    ready = wait_for_server(port)
    if ready:
        webbrowser.open(url)


def wait_for_server(port):
    max_wait_seconds = 40
    start_time = time.time()

    while True:
        if is_port_open(HOST, port):
            return True

        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            print("浏览器自动打开超时，你可以手动访问：")
            print(build_url(port))
            return False

        time.sleep(1)


def is_port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def pick_available_port(preferred_port):
    port = int(preferred_port)
    while port < preferred_port + 100:
        if not is_port_open(HOST, port):
            return port
        port += 1
    print("8501-8600 端口都已被占用，请关闭占用程序后重试。")
    pause_and_exit(1)


def build_url(port):
    return f"http://{HOST}:{int(port)}"


def run_streamlit_server(port):
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_FILE),
        "--server.port",
        str(port),
        "--server.address",
        HOST,
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "auto",
        "--server.runOnSave",
        "true",
    ]

    try:
        result = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        if result.returncode not in (0, None):
            print("")
            print(f"Streamlit 服务异常退出，返回码：{result.returncode}")
            pause_and_exit(result.returncode)
    except KeyboardInterrupt:
        print("")
        print("系统已停止。")
    except Exception as error:
        print("")
        print("启动失败：")
        print(error)
        pause_and_exit(1)


def pause_and_exit(code):
    print("")
    try:
        input("按回车键退出...")
    except EOFError:
        pass
    raise SystemExit(code)


if __name__ == "__main__":
    main()
