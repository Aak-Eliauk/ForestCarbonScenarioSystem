from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _is_streamlit_runtime():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx(suppress_warning=True) is not None


if __name__ == "__main__":
    if _is_streamlit_runtime():
        from fcscs.ui.pages import run_app

        run_app()
    else:
        print("请使用以下任一方式启动系统：")
        print("  python run_system.py")
        print("  streamlit run app.py")
        print("")
        print("直接执行python app.py不会启动Streamlit网页服务。")
