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

def notice():
    print("")
    print(FULL_NAME)
    print("=" * 48)
    print("系统服务终端，运行时不要关闭！！！")
    print("")


def check_files():
    if not APP_CODE.exists():
        print("未找到app.py，请检查项目核心文件是否完整")
        exit(1)

    if not SRC_CODE.exists():
        print("未找到src目录，请检查项目核心文件是否完整")
        exit(1)


def check_python_v():
    major = sys.version_info.major
    minor = sys.version_info.minor
    print(f"Python版本:{major}.{minor}")
    if major < 3:
        print("Python版本过低，请安装Python3以上版本")
        exit(1)


def check_requirements():
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
        exit(1)

    print("依赖检查:通过")


def display_system_start(port, url):
    print("")
    print("运行提示：")
    print("终端同时负责系统界面服务和碳储量后台计算，事件生成、机器学习和蒙特卡洛计算在终端进程调用\n")
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
    print("8501-8600端口都已占用，请关闭占用端口后重试")
    exit(1)


def construct_url(port):
    url = f"http://{HOST}:{int(port)}"
    return url


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
        print("系统已停止")
    except Exception as error:
        print("")
        print("启动失败：")
        print(error)
        exit(1)


#代码src目录加入py导入路径，确保正常读取项目代码
def add_code():
    src_text = str(SRC_CODE)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


def exit(code):
    print("")
    try:
        input("按回车键退出...")
    except EOFError:
        pass
    raise SystemExit(code)

def main():
    notice()
    check_files()
    check_python_v()
    check_requirements()
    add_code()
    port = find_port(PORT)
    url = construct_url(port)
    if "--check" in sys.argv:
        print(f"默认输出目录:{DEFAULT_OUTPUT}")
        print("启动检查完成")
        return
    display_system_start(port, url)
    run_streamlit(port)

if __name__ == "__main__":
    main()
