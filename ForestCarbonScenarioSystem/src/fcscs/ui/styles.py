import html

import streamlit as st


GLOBAL_STYLE = """
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="manage-app-button"], [data-testid="stToolbarActions"],
[data-testid="stHeader"], .viewerBadge_container__1QSob,
a[href*="google.com/search"], a[href*="chat.openai.com"],
a[href*="chatgpt.com"] {
    display: none !important;
    visibility: hidden !important;
}

.stApp {
    background:
        radial-gradient(circle at 12% 8%, rgba(58, 122, 87, 0.16), transparent 28%),
        radial-gradient(circle at 88% 0%, rgba(217, 157, 73, 0.14), transparent 24%),
        linear-gradient(135deg, #f7f4ec 0%, #edf3ec 48%, #e7efe2 100%);
    color: #1f2f28;
}

.block-container {
    max-width: 1240px;
    padding-top: 1.45rem;
    padding-bottom: 2.2rem;
}

.app-hero {
    position: relative;
    overflow: hidden;
    padding: 1rem 1.25rem;
    border-radius: 8px;
    background: linear-gradient(120deg, rgba(23, 65, 47, 0.96), rgba(47, 103, 75, 0.94));
    color: #fffaf0;
    margin-bottom: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.22);
    box-shadow: 0 14px 34px rgba(34, 66, 48, 0.16);
}

.app-hero h1 {
    margin: 0;
    font-size: 1.54rem;
    letter-spacing: 0;
    font-weight: 760;
}

.app-hero p {
    margin: 0.32rem 0 0 0;
    max-width: 900px;
    font-size: 0.94rem;
    line-height: 1.65;
    color: rgba(255, 250, 240, 0.9);
}

.soft-card {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(38, 83, 59, 0.12);
    border-radius: 8px;
    padding: 1rem 1.1rem;
    box-shadow: 0 10px 24px rgba(68, 88, 73, 0.08);
    margin-bottom: 1rem;
}

.soft-card h3 {
    margin: 0 0 0.5rem 0;
    color: #1f4a37;
    font-size: 1.02rem;
    font-weight: 720;
}

.soft-card p {
    color: #2b4338;
    line-height: 1.62;
    font-size: 0.94rem;
    margin: 0 0 0.35rem 0;
}

.metric-box {
    background: rgba(255, 255, 255, 0.82);
    border-radius: 8px;
    padding: 0.85rem 0.95rem;
    border: 1px solid rgba(38, 83, 59, 0.12);
    box-shadow: 0 8px 20px rgba(68, 88, 73, 0.06);
}
"""

SIDEBAR_STYLE = """
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(18, 52, 38, 0.98) 0%, rgba(28, 70, 50, 0.98) 55%, rgba(35, 82, 58, 0.98) 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.12);
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 0.18rem 1rem 1.25rem 1rem;
}

[data-testid="stSidebar"] * {
    color: #fffaf0;
}

[data-testid="stSidebar"] h1 {
    font-size: 0.95rem;
    line-height: 1.32;
    letter-spacing: 0;
    margin: 0;
    padding: 0;
    overflow-wrap: anywhere;
}

.sidebar-brand {
    padding: 0 0 0.85rem 0;
    border-bottom: 1px solid rgba(255, 250, 240, 0.16);
    margin: -0.16rem 0 0.9rem 0;
}

.brand-lockup {
    display: flex;
    align-items: flex-start;
    gap: 0.72rem;
}

.brand-mark {
    width: 4.1rem;
    height: 3.15rem;
    flex: 0 0 4.1rem;
    display: grid;
    place-items: center;
    border-radius: 8px;
    background: linear-gradient(145deg, rgba(255, 250, 240, 0.96), rgba(212, 232, 211, 0.94));
    color: #163a2a !important;
    font-size: 0.7rem;
    line-height: 1.05;
    text-align: center;
    font-weight: 820;
    letter-spacing: 0;
    box-shadow: 0 12px 28px rgba(8, 28, 18, 0.24);
}

.brand-copy {
    min-width: 0;
}

.sidebar-brand .brand-kicker {
    color: rgba(255, 250, 240, 0.72);
    font-size: 0.72rem;
    font-weight: 720;
    margin-bottom: 0.3rem;
}

[data-testid="stSidebar"] .stRadio > label {
    color: rgba(255, 250, 240, 0.72) !important;
    font-size: 0.82rem;
    font-weight: 720;
    margin-bottom: 0.5rem;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 0.48rem;
}

[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
    align-items: center;
    min-height: 3.08rem;
    padding: 0.68rem 0.72rem;
    border: 1px solid rgba(255, 250, 240, 0.13);
    border-radius: 8px;
    background: rgba(255, 250, 240, 0.07);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
    transition: background 160ms ease, border-color 160ms ease;
}

[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
    background: rgba(255, 250, 240, 0.12);
    border-color: rgba(255, 250, 240, 0.24);
}

[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) {
    background: rgba(255, 250, 240, 0.96);
    border-color: rgba(255, 250, 240, 0.96);
    box-shadow: 0 10px 24px rgba(10, 31, 22, 0.2);
}

[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) p,
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) span {
    color: #163a2a !important;
    font-weight: 760;
}

[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] > div:first-child {
    margin-right: 0.58rem;
    transform: scale(0.86);
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255, 250, 240, 0.14);
    margin: 1rem 0 0.85rem 0;
}

.sidebar-section-label {
    color: rgba(255, 250, 240, 0.68);
    font-size: 0.76rem;
    font-weight: 720;
    margin: 0 0 0.5rem 0;
}

.sidebar-nav-item {
    width: 100%;
    min-height: 2.85rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 250, 240, 0.2);
    background: rgba(255, 250, 240, 0.1);
    color: #fffaf0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 720;
    padding: 0.65rem 0.8rem;
    margin-bottom: 0.48rem;
    text-align: center;
}

.sidebar-nav-item.active {
    background: rgba(255, 250, 240, 0.96);
    border-color: rgba(255, 250, 240, 0.96);
    color: #163a2a;
    box-shadow: 0 10px 24px rgba(10, 31, 22, 0.2);
}

[data-testid="stSidebar"] div.stButton > button {
    min-height: 2.85rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 250, 240, 0.2);
    background: rgba(255, 250, 240, 0.1);
    color: #fffaf0 !important;
    box-shadow: none;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    border-color: rgba(255, 250, 240, 0.34);
    background: rgba(255, 250, 240, 0.16);
    color: #fffaf0 !important;
}
"""

FORM_STYLE = """
div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.8);
    border-radius: 8px;
    padding: 1rem 1rem 0.6rem 1rem;
    border: 1px solid rgba(38, 83, 59, 0.12);
    box-shadow: 0 12px 28px rgba(68, 88, 73, 0.07);
}

div.stButton > button {
    border-radius: 8px;
    font-weight: 680;
    border: 1px solid rgba(31, 74, 55, 0.22);
}

div[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

div[data-testid="stTextArea"] textarea {
    border-radius: 8px;
    border-color: rgba(38, 83, 59, 0.12);
    background: rgba(255, 255, 255, 0.72);
    font-family: Consolas, "Courier New", monospace;
    line-height: 1.55;
}

.section-card {
    background: rgba(255, 255, 255, 0.84);
    border: 1px solid rgba(38, 83, 59, 0.12);
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 14px 34px rgba(68, 88, 73, 0.09);
    margin-bottom: 1rem;
}

.status-pill {
    display: inline-block;
    padding: 0.22rem 0.62rem;
    border-radius: 8px;
    background: rgba(39, 105, 73, 0.12);
    color: #173f2e;
    font-weight: 700;
    font-size: 0.82rem;
}

.data-section-title {
    margin: 1.25rem 0 0.62rem 0;
    padding: 0.72rem 0.9rem;
    border-left: 5px solid #245f43;
    border-radius: 8px;
    background: linear-gradient(90deg, rgba(36, 95, 67, 0.12), rgba(255, 255, 255, 0.42));
    color: #173d2d;
    font-size: 1.02rem;
    font-weight: 800;
    box-shadow: 0 8px 18px rgba(68, 88, 73, 0.05);
}

.data-section-title span {
    display: block;
    margin-top: 0.18rem;
    color: #64756a;
    font-size: 0.78rem;
    font-weight: 520;
    line-height: 1.55;
}

.history-group-title {
    margin: 0.85rem 0 0.4rem 0;
    color: #173d2d;
    font-size: 0.95rem;
    font-weight: 800;
}

.data-check-card {
    margin: 1rem 0 0.85rem 0;
    padding: 0.85rem 1rem;
    border-radius: 8px;
    border: 1px solid rgba(38, 83, 59, 0.14);
    background: rgba(255, 255, 255, 0.72);
    color: #213b30;
    box-shadow: 0 10px 24px rgba(68, 88, 73, 0.06);
}

.data-check-card.ok {
    border-color: rgba(37, 118, 76, 0.28);
    background: rgba(232, 246, 234, 0.76);
}

.data-check-card.warn {
    border-color: rgba(197, 97, 40, 0.25);
    background: rgba(255, 244, 226, 0.78);
}

.data-check-card strong {
    display: block;
    margin-bottom: 0.25rem;
    font-size: 0.96rem;
    color: #173d2d;
}

.data-check-card p {
    margin: 0.18rem 0;
    line-height: 1.56;
    color: #43554a;
}
"""

RESULT_STYLE = """
.uncertainty-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 0.75rem;
    margin: 0.2rem 0 1.05rem 0;
}

.uncertainty-card {
    display: flex;
    align-items: center;
    gap: 0.72rem;
    min-height: 4.2rem;
    padding: 0.85rem 0.95rem;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.84);
    border: 1px solid rgba(38, 83, 59, 0.12);
    box-shadow: 0 10px 24px rgba(68, 88, 73, 0.07);
}

.uncertainty-icon {
    width: 3.1rem;
    height: 2.45rem;
    flex: 0 0 3.1rem;
    display: grid;
    place-items: center;
    border-radius: 8px;
    background: rgba(31, 74, 55, 0.1);
    color: #1f4a37;
    font-size: 0.68rem;
    letter-spacing: -0.01rem;
    font-weight: 820;
}

.uncertainty-label {
    color: #637268;
    font-size: 0.78rem;
    font-weight: 680;
    margin-bottom: 0.16rem;
}

.uncertainty-value {
    color: #183d2d;
    font-size: 1.08rem;
    font-weight: 780;
    line-height: 1.25;
    overflow-wrap: anywhere;
}
"""

RASTER_STYLE = """
.raster-zoom-frame {
    width: 100%;
    max-height: 76vh;
    overflow: auto;
    border: 1px solid rgba(38, 83, 59, 0.16);
    border-radius: 8px;
    background: #f4f5f0;
    padding: 0.45rem;
}

.raster-zoom-frame img {
    display: block;
    height: auto;
}
"""

def apply_base_style():
    style_text = "\n".join([GLOBAL_STYLE, SIDEBAR_STYLE, FORM_STYLE, RESULT_STYLE, RASTER_STYLE])
    st.markdown("<style>\n" + style_text + "\n</style>", unsafe_allow_html=True)


def render_page_banner(title, description=""):
    safe_title = html.escape(str(title))
    safe_description = html.escape(str(description))
    if safe_description:
        html_text = f"""
        <div class="app-hero">
            <h1>{safe_title}</h1>
            <p>{safe_description}</p>
        </div>
        """
    else:
        html_text = f"""
        <div class="app-hero">
            <h1>{safe_title}</h1>
        </div>
        """
    st.markdown(html_text, unsafe_allow_html=True)
