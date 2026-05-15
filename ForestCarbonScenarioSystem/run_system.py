import importlib.util
import socket
import subprocess
import sys
from pathlib import Path


ROOT_FOLDER = Path(__file__).resolve().parent
APP_CODE = ROOT_FOLDER / "app.py"
SRC_CODE = ROOT_FOLDER / "src"
DEFAULT_OUTPUT = (ROOT_FOLDER.parent / "ForestCarbonScenarioSystem_outputs").resolve()
HOST = "localhost"
PORT = 8501
FULL_NAME = "森林损失情景控制与蒙特卡洛碳储量模拟系统V1.0"

def show_name():
    print("")
    print(FULL_NAME)
    print("=" * 48)
    print("系统服务终端，运行时不要关闭！")
    print("")


def check_code_files():
    if not APP_CODE.exists():
        print("未找到app.py，请确认脚本放在项目根目录。")
        quit(1)

    if not SRC_CODE.exists():
        print("未找到src目录，请确认项目文件完整。")
        quit(1)


def check_python_version():
    major = sys.version_info.major
    minor = sys.version_info.minor
    print(f"Python版本:{major}.{minor}")
    if major < 3:
        print("Python版本过低，请使用Python3。")
        quit(1)


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
        print("请先安装依赖，在当前目录运行：")
        print("python -m pip install -r requirements.txt")
        quit(1)

    print("依赖检查:通过")


def print_run_explain(port, url):
    print("")
    print("系统运行说明：")
    print("1.终端同时负责网页界面服务和碳储量后台计算。")
    print("2.浏览器显示用户界面，事件生成、机器学习和蒙特卡洛计算在这个终端进程里执行。")
    print("")
    if port != PORT:
        print(f"默认端口{PORT}已被占用，改用端口{port}。")
    print(f"正在启动系统:{url}")
    print(f"默认输出目录:{DEFAULT_OUTPUT}")
    print("")


def is_port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def find_port(target_port):
    port = int(target_port)
    while port < target_port + 100:
        if not is_port_open(HOST, port):
            return port
        port += 1
    print("8501-8600端口都已被占用，请关闭占用程序后重试。")
    quit(1)


def construct_url(port):
    return f"http://{HOST}:{int(port)}"


def run_streamlit(port):
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_CODE),
        "--server.port",
        str(port),
        "--server.address",
        HOST,
    ]

    try:
        subprocess.run(command, cwd=str(ROOT_FOLDER))
    except KeyboardInterrupt:
        print("")
        print("系统已停止。")
    except Exception as error:
        print("")
        print("启动失败：")
        print(error)
        quit(1)


#代码src目录加入py导入路径，确保正常读取项目代码
def add_code_path():
    src_text = str(SRC_CODE)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


def quit(code):
    print("")
    try:
        input("按回车键退出...")
    except EOFError:
        pass
    raise SystemExit(code)

def main():
    show_name()
    check_code_files()
    check_python_version()
    check_required_packages()
    add_code_path()
    port = find_port(PORT)
    url = construct_url(port)
    if "--check" in sys.argv:
        print(f"默认输出目录:{DEFAULT_OUTPUT}")
        print("启动检查完成。")
        return
    print_run_explain(port, url)
    run_streamlit(port)

if __name__ == "__main__":
    main()
