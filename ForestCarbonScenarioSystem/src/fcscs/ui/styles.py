import html

import streamlit as st


def apply_base_style():
    st.markdown('\n        <style>\n        #MainMenu, footer, header, [data-testid="stToolbar"],\n        [data-testid="stDecoration"], [data-testid="stStatusWidget"],\n        [data-testid="manage-app-button"], [data-testid="stToolbarActions"],\n        [data-testid="stHeader"], .viewerBadge_container__1QSob,\n        a[href*="google.com/search"], a[href*="chat.openai.com"],\n        a[href*="chatgpt.com"] {\n            display: none !important;\n            visibility: hidden !important;\n        }\n\n        .stApp {\n            background:\n                radial-gradient(circle at 12% 8%, rgba(58, 122, 87, 0.18), transparent 28%),\n                radial-gradient(circle at 88% 0%, rgba(229, 168, 84, 0.18), transparent 24%),\n                linear-gradient(135deg, #f7f4ec 0%, #edf3ec 48%, #e7efe2 100%);\n            color: #1f2f28;\n        }\n\n        .block-container {\n            max-width: 1240px;\n            padding-top: 1.6rem;\n            padding-bottom: 2.2rem;\n        }\n\n        .app-hero {\n            position: relative;\n            overflow: hidden;\n            padding: 1.05rem 1.35rem;\n            border-radius: 18px;\n            background:\n                linear-gradient(120deg, rgba(23, 65, 47, 0.96), rgba(47, 103, 75, 0.94)),\n                radial-gradient(circle at 100% 10%, rgba(246, 199, 117, 0.32), transparent 26%);\n            color: #fffaf0;\n            margin-bottom: 1rem;\n            border: 1px solid rgba(255, 255, 255, 0.22);\n            box-shadow: 0 18px 45px rgba(34, 66, 48, 0.18);\n        }\n\n        .app-hero h1 {\n            margin: 0;\n            font-size: 1.78rem;\n            letter-spacing: 0.02em;\n            font-weight: 760;\n        }\n\n        .app-hero p {\n            margin: 0.35rem 0 0 0;\n            max-width: 900px;\n            font-size: 0.98rem;\n            line-height: 1.7;\n            color: rgba(255, 250, 240, 0.9);\n        }\n\n        .soft-card {\n            background: rgba(255, 255, 255, 0.78);\n            border: 1px solid rgba(38, 83, 59, 0.12);\n            border-radius: 10px;\n            padding: 1rem 1.1rem;\n            box-shadow: 0 12px 28px rgba(68, 88, 73, 0.08);\n            margin-bottom: 1rem;\n        }\n\n        .soft-card h3 {\n            margin: 0 0 0.5rem 0;\n            color: #1f4a37;\n            font-size: 1.02rem;\n            font-weight: 720;\n        }\n\n        .soft-card p {\n            color: #2b4338;\n            line-height: 1.62;\n            font-size: 0.94rem;\n            margin: 0 0 0.35rem 0;\n        }\n\n        .metric-strip {\n            display: grid;\n            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));\n            gap: 0.75rem;\n            margin-bottom: 1rem;\n        }\n\n        .metric-box {\n            background: rgba(255, 255, 255, 0.82);\n            border-radius: 10px;\n            padding: 0.85rem 0.95rem;\n            border: 1px solid rgba(38, 83, 59, 0.12);\n            box-shadow: 0 8px 20px rgba(68, 88, 73, 0.06);\n        }\n\n        .metric-box .label {\n            font-size: 0.78rem;\n            color: #637268;\n            margin-bottom: 0.16rem;\n        }\n\n        .metric-box .value {\n            font-size: 1.28rem;\n            color: #183d2d;\n            font-weight: 760;\n        }\n\n        [data-testid="stSidebar"] {\n            background: linear-gradient(180deg, #183d2d 0%, #28563f 100%);\n            border-right: 1px solid rgba(255, 255, 255, 0.12);\n        }\n\n        [data-testid="stSidebar"] * {\n            color: #fffaf0;\n        }\n\n        [data-testid="stSidebar"] .stRadio label {\n            padding: 0.2rem 0;\n        }\n\n        div[data-testid="stForm"] {\n            background: rgba(255, 255, 255, 0.8);\n            border-radius: 16px;\n            padding: 1rem 1rem 0.6rem 1rem;\n            border: 1px solid rgba(38, 83, 59, 0.12);\n            box-shadow: 0 12px 28px rgba(68, 88, 73, 0.07);\n        }\n\n        div.stButton > button {\n            border-radius: 999px;\n            font-weight: 680;\n            border: 1px solid rgba(31, 74, 55, 0.22);\n        }\n\n        div[data-testid="stDataFrame"] {\n            border-radius: 14px;\n            overflow: hidden;\n        }\n\n        .section-card {\n            background: rgba(255, 255, 255, 0.84);\n            border: 1px solid rgba(38, 83, 59, 0.12);\n            border-radius: 18px;\n            padding: 1rem;\n            box-shadow: 0 14px 34px rgba(68, 88, 73, 0.09);\n            margin-bottom: 1rem;\n        }\n\n        .status-pill {\n            display: inline-block;\n            padding: 0.22rem 0.62rem;\n            border-radius: 999px;\n            background: rgba(39, 105, 73, 0.12);\n            color: #173f2e;\n            font-weight: 700;\n            font-size: 0.82rem;\n        }\n        </style>\n        ', unsafe_allow_html=True)


def render_page_banner(title, description=""):
    safe_title = html.escape(str(title))
    safe_description = html.escape(str(description))
    if safe_description:
        html_text = '\n        <div class="app-hero">\n            <h1>' + safe_title + '</h1>\n            <p>' + safe_description + '</p>\n        </div>\n        '
    else:
        html_text = '\n        <div class="app-hero">\n            <h1>' + safe_title + '</h1>\n        </div>\n        '
    st.markdown(html_text, unsafe_allow_html=True)


def render_soft_card(title, lines):
    html_lines = []
    for line in lines:
        html_lines.append(f"<p>{html.escape(str(line))}</p>")
    html_text = '\n    <div class="soft-card">\n        <h3>' + html.escape(str(title)) + '</h3>\n        ' + "".join(html_lines) + '\n    </div>\n    '
    st.markdown(html_text, unsafe_allow_html=True)


def render_metric_strip(items):
    if not items:
        return
    columns = st.columns(min(len(items), 4))
    for index, item in enumerate(items):
        label, value = item
        column = columns[index % len(columns)]
        column.metric(str(label), value)
