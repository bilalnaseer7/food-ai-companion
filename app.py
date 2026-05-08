import os
import math
import hashlib
import html as html_module
import inspect
import json
import re
import base64
import requests
from datetime import datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

from src.data_loader import load_reviews
from src.recommend import rag_recommend
from src.retrieval import find_static_cuisine_matches
from src.taste_profile import load_profile, save_profile, update_profile
from src.places import PRICE_LABEL

load_dotenv()

st.set_page_config(
    page_title="FoodAI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html("<style>[data-testid='stHeaderActionElements'] {display: none;}</style>")

EASTERN_TZ = ZoneInfo("America/New_York")
now = datetime.now(EASTERN_TZ)
day = now.strftime("%a")
hour = now.hour
time_now = "morning" if hour < 12 and hour >= 4 else "afternoon" if hour < 18 and hour >= 12 else "evening" if hour < 23 and hour >= 18 else "late"

CUISINE_GRADIENTS = {
    "italian": "warm-amber", "pizza": "wine-rust",
    "japanese": "scallion-jade", "ramen": "broth-green",
    "korean": "bean-paprika", "chinese": "smoke-olive",
    "mexican": "char-orange", "american": "rice-gold",
    "seafood": "salmon-glaze", "mediterranean": "garden-jade",
    "cocktail": "spritz-rust", "bar": "rye-sage",
    "breakfast": "pepper-cream", "thai": "broth-green",
    "indian": "amber-gold", "french": "pepper-cream",
    "greek": "garden-jade", "vietnamese": "scallion-jade",
}

BUDGET_LABELS = {1: "Thrifty", 2: "Moderate", 3: "Foodie", 4: "Int'l Student"}
BUDGET_MAP = {"budget": 1, "moderate": 2, "premium": 3, "premium+": 4}
BUDGET_REVERSE = {1: "budget", 2: "moderate", 3: "premium", 4: "premium+"}
PROFILE_PATH = "data/taste_profile.json"
PROFILE_RESET_MARKER = "data/.profile_reset"

QUICK_STARTS = {
    "eat":  ["something cozy", "michelin star","wood-fired anything", "walking distance"],
    "cook": ["25 min weeknight", "use up the salmon", "one-pot", "something to impress"],
    "drink": ["rainy night", "pre-dinner aperitivo", "smoky and bitter", "low-ABV refresher"],
}

EMPTY_COPY = {
    "eat": {
        "glyph": "✦",
        "title": "Where shall we eat?",
        "body": "Tell me what you're craving. Be vague, be specific, anything goes. I'll help you narrow it down.",
    },
    "cook": {
        "glyph": "◐",
        "title": "What's in the kitchen tonight?",
        "body": "Drop your craving and what's actually in the fridge. We'll work backwards from there.",
    },
    "drink": {
        "glyph": "◑",
        "title": "Pick a vibe.",
        "body": "A mood, a season, a song — we'll match it to what you have on the bar cart.",
    },
}

THINKING_MSG = {
    "eat": "Reading the room…",
    "cook": "Browsing your shelf…",
    "drink": "Mixing ideas…",
}

TAB_HEADING = {
    "eat": "Tonight, near you" if time_now == "evening" or time_now == "late" else "Right here, right now", 
    "cook": "In your kitchen",
    "drink": "On the bar cart",
}

TAB_ICON = {"eat": "◖", "cook": "◐", "drink": "◗"}
MAX_PREFERENCE_TAGS = 8


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@st.cache_data
def get_df():
    return load_reviews(path="data/restaurants.csv", max_rows=3000)


@st.cache_data
def get_full_restaurant_df():
    return load_reviews(path="data/restaurants.csv", max_rows=None)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def image_url_to_data_uri(url: str) -> str:
    if not url:
        return ""
    try:
        response = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "FoodAI/1.0 (+https://streamlit.app)"},
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "image/jpeg").split(";")[0]
        if not content_type.startswith("image/"):
            return ""
        encoded = base64.b64encode(response.content).decode("ascii")
        return f"data:{content_type};base64,{encoded}"
    except Exception:
        return ""


def compatible_form(*, key: str, border: bool = False, enter_to_submit: bool = True):
    """Use Enter-to-submit when the installed Streamlit version supports it."""
    kwargs = {"key": key, "border": border}
    if "enter_to_submit" in inspect.signature(st.form).parameters:
        kwargs["enter_to_submit"] = enter_to_submit
    return st.form(**kwargs)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #FAF7F2;
    --bg-deep: #F2EDE3;
    --card: #FFFFFF;
    --ink: #1A1A1A;
    --ink-2: #6B6B6B;
    --ink-3: #9A968D;
    --input-text: #5F5D58;
    --placeholder-text: #74716A;
    --line: rgba(26,26,26,0.06);
    --line-2: rgba(26,26,26,0.10);
    --terracotta: #C96A3A;
    --terracotta-2: #B25A2C;
    --button-terracotta: #C96A3A;
    --button-terracotta-2: #B25A2C;
    --sage: #7A9E7E;
    --sage-2: #688C6D;
    --gold: #C9A227;
    --tag-terracotta: rgba(201,106,58,0.13);
    --tag-sage: rgba(122,158,126,0.15);
    --tag-gold: rgba(201,162,39,0.16);
    --tag-ink: rgba(26,26,26,0.05);
    --shadow-card: 0 1px 0 rgba(26,26,26,0.02), 0 6px 18px rgba(60,40,20,0.06), 0 24px 60px -30px rgba(60,40,20,0.12);
    --shadow-pop: 0 1px 0 rgba(26,26,26,0.02), 0 12px 32px rgba(60,40,20,0.10), 0 32px 80px -40px rgba(60,40,20,0.18);
    --shadow-input: inset 0 0 0 1px rgba(26,26,26,0.06);
    --radius-sm: 8px; --radius: 14px; --radius-lg: 20px; --radius-pill: 999px;
    --result-card-height: 300px;
    --result-card-action-height: 150px;
    --drink-card-image-width: 220px;
    --serif: 'DM Serif Display', Georgia, serif;
    --sans: 'DM Sans', system-ui, sans-serif;
    --mono: 'IBM Plex Mono', ui-monospace, monospace;
    /* Override Streamlit's internal theme variables so its own component CSS uses our colors */
    --background-color: #FAF7F2;
    --secondary-background-color: #F2EDE3;
}

* { box-sizing: border-box; }

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: var(--sans) !important;
    font-weight: 350 !important;
    font-size: 15px !important;
    line-height: 1.55 !important;
    overscroll-behavior: none !important;
}
            

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"] { background: #FAF7F2 !important; }
[data-testid="stHeader"] {
    background: var(--bg) !important;
    box-shadow: none !important;
    display: block !important;
    visibility: visible !important;
}
[data-testid="stMain"] {
    margin-left: 0 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}
.main .block-container,
[data-testid="stMainBlockContainer"],
div[class*="block-container"] {
    background: transparent !important;
    width: 100% !important;
    max-width: 1120px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding: 36px 48px 220px !important;
    box-sizing: border-box !important;
}

#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}

a { color: inherit; text-decoration: none; }
a:hover { text-decoration: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg) !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebarCollapsedControl"] {
    background: #FAF7F2 !important;
    border-right: 1px solid rgba(26,26,26,0.06) !important;
}
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] span,
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {
    color: #1A1A1A !important;
    opacity: 1 !important;
}
[data-testid="stExpandSidebarButton"] {
    margin-top: 50px !important;
    margin-left: 12px !important;
}

[data-testid="stSidebar"][aria-expanded="true"] {
    width: 355px !important;
    min-width: 355px !important;
    max-width: 355px !important;
    margin-right: -175px !important;
    position: relative !important;
    transition: all 0.3s ease !important;
}
[data-testid="stSidebar"] > div { padding: 24px 20px !important; }
[data-testid="stSidebar"] * { color: var(--ink) !important; }

/* ── Brand ── */
.brand { display: flex; align-items: center; gap: 10px; margin-top: -30px; padding-bottom: 18px; }
.brand-mark {
    width: 13px; height: 13px; border-radius: 50%; margin-left: 6px;
    background: #FFD8A8;
    box-shadow:
        0 0 0 3px rgba(201,106,58,0.10),
        0 0 12px 4px rgba(201,106,58,0.32),
        0 0 24px 8px rgba(255,216,168,0.20);
    flex-shrink: 0;
}
.brand-name { font-family: var(--serif); font-size: 32px; letter-spacing: -0.01em; margin-left: 10px; }
.brand-name em { font-style: italic; color: var(--terracotta); }

/* ── Side sections ── */
.side-section { margin-bottom: 24px; }
.side-label {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--ink-3);
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px;
}
.side-label .count { font-variant-numeric: tabular-nums; }

/* Cuisine pulse */
.cuisine-list { display: flex; flex-direction: column; gap: 8px; }
.cuisine-row {
    display: grid; grid-template-columns: 80px 1fr 28px;
    align-items: center; gap: 10px; font-size: 13px;
}
.cuisine-row .name { color: var(--ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cuisine-row .pct { font-family: var(--mono); font-size: 10.5px; color: var(--ink-3); text-align: right; }
.cuisine-bar { height: 6px; background: var(--bg-deep); border-radius: var(--radius-pill); overflow: hidden; }
.cuisine-bar > i {
    display: block; height: 100%;
    background: linear-gradient(90deg, var(--terracotta) 0%, #E08A5D 100%);
    border-radius: inherit;
    transition: width 0.6s cubic-bezier(.2,.8,.2,1);
}

/* ── Pills ── */
.tag-cluster { display: flex; flex-wrap: wrap; gap: 6px; }
.pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 11px; border-radius: var(--radius-pill);
    font-size: 12.5px; font-weight: 400;
    background: var(--tag-ink); color: var(--ink);
    border: 1px solid transparent;
}
.pill.terracotta { background: var(--tag-terracotta); color: var(--terracotta-2); }
.pill.sage { background: var(--tag-sage); color: var(--sage-2); }
.pill.outline { background: transparent; border-color: var(--line-2); color: var(--ink-2); }
.pill .pill-bar { width: 4px; height: 4px; border-radius: 50%; background: currentColor; opacity: 0.5; }

/* ── Budget ── */
.budget { display: flex; gap: 4px; align-items: center; }
.budget .dot {
    width: 28px; height: 6px; border-radius: var(--radius-pill);
    background: var(--bg-deep);
    transition: background 0.15s ease;
    display: inline-block;
}
.budget .dot.on { background: var(--gold); }
.budget .dot:hover { background: rgba(201, 162, 39, 0.5); }
.budget-label { font-family: var(--mono); font-size: 10.5px; color: var(--ink-3); margin-left: 4px; }
[data-testid="stSidebar"] [data-testid="stButton"] button {
    background: transparent !important;
    border: 1px solid var(--line-2) !important;
    border-radius: var(--radius) !important;
    color: var(--ink-2) !important;
    font-family: var(--sans) !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    box-shadow: none !important;
    min-height: 28px !important;
    padding: 4px 8px !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
    color: var(--ink) !important;
    border-color: var(--ink) !important;
    background: var(--card) !important;
}

/* ── Insights / donut ── */
.insights {
    display: flex; flex-direction: column; gap: 8px;
    padding: 12px 14px;
    background: var(--card); border: 1px solid var(--line);
    border-radius: var(--radius);
}
.insights-head {
    display: flex; justify-content: space-between; align-items: baseline;
    font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--ink-3);
}
.insights-head .pct { font-size: 10.5px; color: var(--sage-2); }
.donut-wrap { display: flex; align-items: center; gap: 12px; padding: 4px 0 2px; }
.donut { width: 64px; height: 64px; flex-shrink: 0; }
.donut-text { font-family: var(--serif); font-size: 18px; fill: var(--ink); }
.legend { flex: 1; display: flex; flex-direction: column; gap: 3px; font-size: 11.5px; }
.legend-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.legend-row .lbl { display: flex; align-items: center; gap: 6px; color: var(--ink-2); }
.legend-row .lbl .sw { width: 8px; height: 8px; border-radius: 2px; }
.legend-row .num { font-family: var(--mono); font-size: 10.5px; color: var(--ink-3); }

/* ── History ── */
.history { display: flex; flex-direction: column; gap: 4px; }
.history-row { display: flex; align-items: center; gap: 8px; font-size: 12.5px; padding: 4px 0; color: var(--ink-2); }
.history-row .dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.history-row.acc .dot { background: var(--sage); }
.history-row.rej .dot { background: var(--terracotta); }
.history-row .name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--ink); }
.history-row .src { font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); }
.history-empty { color: var(--ink-3); font-style: italic; font-size: 12.5px; padding: 4px 0; }

/* ── Reset link ── */
.reset-link {
    display: flex; align-items: center; justify-content: space-between;
    padding: 9px 12px; background: transparent;
    border: 1px solid var(--line-2); border-radius: var(--radius);
    color: var(--ink-2); font-size: 13px;
    transition: all 0.15s ease;
    cursor: pointer;
}
.reset-link:hover { color: var(--ink); border-color: var(--ink); background: var(--card); }
.reset-link .glyph { font-family: var(--mono); font-size: 11px; opacity: 0.6; }

/* ── Greeting ── */
.greeting {
    display: flex; align-items: baseline; justify-content: space-between;
    gap: 24px; margin: 15px 0 8px;
}
.greeting h1 {
    font-family: var(--serif); font-size: 38px; font-weight: 400;
    letter-spacing: -0.015em; margin: 0; line-height: 1.1;
}
.greeting-title {
    font-family: var(--serif); font-size: 38px; font-weight: 400;
    letter-spacing: -0.015em; margin: 0; line-height: 1.1;
}
.greeting h1 em, .greeting-title em { color: var(--terracotta); font-style: italic; }
.greeting a { display: none !important; }
.greeting .meta {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em;
    color: var(--ink-3); text-transform: uppercase;
}
.subline {
    color: var(--ink-2); font-size: 15px; margin: 0 0 28px;
    max-width: 56ch;
}

/* ── Hint ── */
.hint {
    position: relative; display: flex; gap: 14px; align-items: flex-start;
    padding: 14px 16px 14px 18px;
    background: linear-gradient(180deg, #FFFBF5 0%, #FAF3E8 100%);
    border: 1px solid rgba(201, 162, 39, 0.25);
    border-radius: var(--radius);
    margin-bottom: 24px;
}
.hint .glyph {
    width: 32px; height: 32px; border-radius: 50%;
    background: var(--tag-gold); color: #8C7016;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--serif); font-size: 16px; flex-shrink: 0;
}
.hint .body { flex: 1; font-size: 13.5px; }
.hint .body b { font-weight: 500; }
.hint .body p { margin: 2px 0 0; color: var(--ink-2); font-size: 13px; }
.hint .x {
    width: 22px; height: 22px; border-radius: 6px;
    color: var(--ink-3); font-size: 14px; line-height: 1;
    display: flex; align-items: center; justify-content: center;
}
.hint .x:hover { background: rgba(0,0,0,0.05); color: var(--ink); }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border-bottom: none !important;
    border-radius: 0 !important; padding: 0 !important; gap: 0 !important;
    margin-bottom: 0 px !important;
}
[data-testid="stTabs"] [role="tabpanel"],
[data-testid="stTabs"] [data-baseweb="tab-panel"],
[data-testid="stTabs"] [role="tabpanel"] > div {
    border-top: none !important;
    box-shadow: none !important;
}
[data-testid="stTabs"] [role="tab"] {
    color: var(--ink-2) !important;
    font-family: var(--sans) !important;
    font-size: 14.5px !important;
    font-weight: 400 !important;
    border-radius: 0 !important;
    padding: 14px 20px 16px !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    margin: 0 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--ink) !important;
    font-weight: 500 !important;
    border-bottom: 2px solid var(--terracotta) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: var(--terracotta) !important;
}

/* ── Search card ── */
.search-card {
    background: var(--card); border: 1px solid var(--line);
    border-radius: var(--radius-lg); padding: 22px 24px 20px;
    box-shadow: var(--shadow-card); margin-bottom: 28px;
}
.search-grid { display: grid; gap: 14px 16px; margin-bottom: 16px; }
.search-grid.eat { grid-template-columns: 1fr 200px; }
.search-grid.cook, .search-grid.drink { grid-template-columns: 1fr; }
.field { display: flex; flex-direction: column; gap: 6px; }
.field-label {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--ink-3);
    margin-bottom: 5px;
}

[data-testid="stBottom"],
[data-testid="stMainBlockContainer"],
section[data-testid="stSidebar"] ~ div,
.stMainBlockContainer,
div[class*="block-container"] {
    background: #FAF7F2 !important;
}
[data-testid="stMainBlockContainer"],
.stMainBlockContainer,
div[class*="block-container"] {
    max-width: 1120px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}            
        
/* Streamlit input overrides */
.stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div,
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="base-input"] {
    background: var(--bg) !important;
    border: 1px solid rgba(26,26,26,0.10) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: none !important;
    overflow: hidden !important;
}
.stTextInput input, .stTextArea textarea {
    caret-color: var(--ink) !important;
    color: var(--input-text) !important;
    background: transparent !important;
    font-family: var(--sans) !important;
    font-size: 14.5px !important;
    border: none !important;
    border-radius: inherit !important;
    padding: 12px 14px !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="base-input"] input {
    caret-color: var(--ink) !important;
    color: var(--input-text) !important;
    -webkit-text-fill-color: var(--input-text) !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
[data-baseweb="input"] input::placeholder,
[data-baseweb="textarea"] textarea::placeholder,
[data-baseweb="base-input"] input::placeholder {
    color: var(--placeholder-text) !important;
    opacity: 1 !important;
}
.stTextInput > div > div:focus-within, .stTextArea > div > div:focus-within {
    background: var(--card) !important;
    border-color: var(--terracotta) !important;
    box-shadow: 0 0 0 3px rgba(201, 106, 58, 0.12) !important;
}

/* Streamlit form labels — hide them, we use custom .field-label HTML */
.stTextInput label, .stTextArea label, .stSelectbox label, .stCheckbox label {
    display: none !important;
}

/* ── Suggest chips (anchor links) ── */
.suggest-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 12px;
}
.suggest-chip {
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    background: transparent;
    border: 1px dashed var(--line-2);
    border-radius: var(--radius-pill);
    padding: 4px 11px;
    font-size: 12.5px;
    font-weight: 400;
    line-height: 1.2;
    color: var(--ink-2);
    transition: all 0.14s ease;
    cursor: pointer;
    white-space: nowrap;
}
.suggest-chip:hover {
    border-color: var(--terracotta); color: var(--terracotta-2); border-style: solid;
}

/* ── Submit button (primary) ── */
[data-testid="stFormSubmitButton"] button,
[data-testid="stFormSubmitButton"] button[kind="primary"],
[data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] {
    background: var(--button-terracotta) !important; color: #fff !important;
    border: none !important; 
    font-family: var(--sans) !important; font-weight: 500 !important;
    font-size: 14px !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.2) inset, 0 4px 14px rgba(169,101,67,0.24) !important;
    transition: all 0.12s ease !important;
    width: 100% !important;
}
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]:hover {
    background: var(--button-terracotta-2) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(145,86,58,0.30) !important;
}
[data-testid="stFormSubmitButton"] button p {
    color: #fff !important;
}

.suggest-chip { color: var(--ink-2) !important; }

/* ── Recently saved strip ── */
.recent {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px; background: var(--bg-deep);
    border-radius: var(--radius); margin-bottom: 12px;
    font-size: 13px; color: var(--ink-2);
    margin-bottom: 12px;
}
.recent-label {
    font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--ink-3); flex-shrink: 0;
}
.recent-track { display: flex; gap: 6px; flex-wrap: nowrap; overflow-x: scroll; flex: 1; scrollbar-width: none; }
.recent-track::-webkit-scrollbar { display: none; }
.recent-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; background: var(--card);
    border-radius: var(--radius-pill); font-size: 12px;
    white-space: nowrap; border: 1px solid var(--line); color: var(--ink);
}
.recent-chip .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--sage); }
.recent-chip.rej .dot { background: var(--terracotta); }

/* ── Results head ── */
.results-head {
    display: flex; align-items: baseline; justify-content: space-between;
    margin: -15px 0 5px;
}
.results-head h2 {
    font-family: var(--serif); font-weight: 400; font-size: 22px;
    letter-spacing: -0.01em; margin: 0;
}
.results-head .count {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--ink-3);
}

/* ── Cards ── */
.cards { display: flex; flex-direction: column; gap: 16px; }
.card {
    background: var(--card); border-radius: var(--radius-lg);
    border: 1px solid var(--line); box-shadow: var(--shadow-card);
    display: grid; grid-template-columns: 168px 1fr;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.25s ease;
    margin-bottom: 16px;
}
.card:hover {box-shadow: var(--shadow-pop); }
.card.accepted { border-color: rgba(122,158,126,0.5); }
.card.rejected { opacity: 0.5; }

.card-img { position: relative; min-height: 180px; overflow: hidden; }
.card-img img {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    z-index: 1;
}
.card-img::after {
    content: ""; position: absolute; inset: 0;
    background: repeating-linear-gradient(45deg, rgba(255,255,255,0.04) 0 8px, transparent 8px 16px);
    pointer-events: none;
    z-index: 2;
}
.card-img .ph {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.14em;
    color: rgba(0,0,0,0.35); text-transform: uppercase;
}
.card-img.has-photo .ph { display: none; }
.card-img .label, .card-img .photo-attr {
    position: absolute; bottom: 10px; left: 10px;
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(8px);
    padding: 4px 9px; border-radius: var(--radius-pill);
    font-family: var(--mono); font-size: 10px;
    letter-spacing: 0.08em; color: var(--ink); text-transform: uppercase;
    z-index: 3;
}
.card-img .photo-attr {
    left: auto; right: 10px;
    max-width: calc(100% - 20px);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    text-transform: none;
    color: var(--ink-2);
}

.ph-warm-amber  { background: linear-gradient(135deg, #E8B07A 0%, #C97A3D 100%); }
.ph-smoke-olive { background: linear-gradient(135deg, #B5A77E 0%, #6E7A56 100%); }
.ph-rice-gold   { background: linear-gradient(135deg, #F0DDA0 0%, #C9A227 100%); }
.ph-wine-rust   { background: linear-gradient(135deg, #B05546 0%, #6A2A2C 100%); }
.ph-broth-green { background: linear-gradient(135deg, #C9D4A0 0%, #6F8B5C 100%); }
.ph-pepper-cream{ background: linear-gradient(135deg, #F2EAD6 0%, #C9B07A 100%); }
.ph-char-orange { background: linear-gradient(135deg, #E89868 0%, #A14826 100%); }
.ph-salmon-glaze{ background: linear-gradient(135deg, #F0A788 0%, #B05A3F 100%); }
.ph-bean-paprika{ background: linear-gradient(135deg, #E0C290 0%, #A85F30 100%); }
.ph-scallion-jade{ background: linear-gradient(135deg, #C4D6A8 0%, #5F8268 100%); }
.ph-amber-gold  { background: linear-gradient(135deg, #E8B870 0%, #A6722A 100%); }
.ph-garden-jade { background: linear-gradient(135deg, #B8CFA0 0%, #6A8E70 100%); }
.ph-spritz-rust { background: linear-gradient(135deg, #E89570 0%, #B0552E 100%); }
.ph-rye-sage    { background: linear-gradient(135deg, #C4B580 0%, #6E7E55 100%); }

.card-body { padding: 14px 14px 12px; display: flex; flex-direction: column; gap: 5px; min-width: 0; }
.card-meta-top {
    display: flex; align-items: center; gap: 10px;
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--ink-3);
}
.card-meta-top .sep { opacity: 0.6; }
.card-meta-top .open { color: var(--sage-2); }

.card-title {
    font-family: var(--serif); font-weight: 400; font-size: 22px;
    line-height: 1.15; letter-spacing: -0.01em;
    margin: 0; color: var(--ink);
}
.card-rating {
    display: flex; align-items: center; gap: 8px;
    font-size: 13px; color: var(--ink); flex-wrap: wrap;
}
.card-rating .stars { color: var(--gold); letter-spacing: -1px; font-size: 13px; }
.card-rating .num { font-variant-numeric: tabular-nums; }
.card-rating .sep { color: var(--ink-3); }
.card-rating .reviews { color: var(--ink-3); font-size: 12.5px; }

.card-blurb {
    font-size: 14px; line-height: 1.55; color: var(--ink);
    margin: 4px 0 6px; max-width: 80ch;
}
.card-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 2px; max-height: 32px; overflow: hidden; }

.card-actions {
    display: flex; gap: 8px; margin-top: auto;
    padding-top: 14px; border-top: 1px solid var(--line);
    align-items: center;
    justify-content: space-between;
}
.card-extra {
    display: flex; align-items: center; gap: 8px;
    margin-right: 0; font-size: 12px;
    color: var(--sage-2); font-family: var(--mono); letter-spacing: 0.04em;
}
.card-extra .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--sage); }
.card-extra.warn { color: var(--gold); }
.card-extra.warn .dot { background: var(--gold); }
.card-extra.closed { color: var(--terracotta); }
.card-extra.closed .dot { background: var(--terracotta); }

.btn-accept, .btn-reject {
    display: inline-flex; align-items: center; gap: 6px;
    border: 1px solid; background: var(--card);
    padding: 8px 14px; border-radius: var(--radius);
    font-size: 13px; font-weight: 500;
    transition: all 0.15s ease; cursor: pointer;
}
.btn-accept { color: var(--sage-2); border-color: rgba(122,158,126,0.4); }
.btn-accept:hover { background: var(--sage); color: #fff; border-color: var(--sage); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(122,158,126,0.30); }
.btn-reject { color: var(--terracotta-2); border-color: rgba(201,106,58,0.4); }
.btn-reject:hover { background: var(--terracotta); color: #fff; border-color: var(--terracotta); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(201,106,58,0.30); }
.btn-accept .glyph, .btn-reject .glyph { font-family: var(--mono); font-size: 11px; font-weight: 600; }

[class*="st-key-cook_pass"] button, [class*="st-key-drink_pass"] button,
[class*="st-key-cook_option_pass_"] button, [class*="st-key-drink_option_pass_"] button {
    color: var(--terracotta-2) !important; border-color: rgba(201,106,58,0.4) !important;
    background: var(--card) !important; font-weight: 500 !important;
}
[class*="st-key-cook_pass"] button:hover, [class*="st-key-drink_pass"] button:hover,
[class*="st-key-cook_option_pass_"] button:hover, [class*="st-key-drink_option_pass_"] button:hover {
    background: var(--terracotta) !important; color: #fff !important; border-color: var(--terracotta) !important;
}
[class*="st-key-cook_save"] button, [class*="st-key-drink_save"] button,
[class*="st-key-cook_option_save_"] button, [class*="st-key-drink_option_save_"] button {
    color: var(--sage-2) !important; border-color: rgba(122,158,126,0.4) !important;
    background: var(--card) !important; font-weight: 500 !important;
}
[class*="st-key-cook_save"] button:hover, [class*="st-key-drink_save"] button:hover,
[class*="st-key-cook_option_save_"] button:hover, [class*="st-key-drink_option_save_"] button:hover {
    background: var(--sage) !important; color: #fff !important; border-color: var(--sage) !important;
}
[class*="st-key-cook_remix_toggle"] button, [class*="st-key-drink_remix_toggle"] button,
[class*="st-key-cook_undo_save"] button, [class*="st-key-drink_undo_save"] button,
[class*="st-key-cook_option_undo_pass_"] button, [class*="st-key-drink_option_undo_pass_"] button,
[class*="st-key-cook_option_undo_save_"] button, [class*="st-key-drink_option_undo_save_"] button {
    color: var(--ink-2) !important; border-color: var(--line-2) !important;
    background: var(--card) !important; font-weight: 500 !important;
}
[class*="st-key-cook_remix_toggle"] button:hover, [class*="st-key-drink_remix_toggle"] button:hover,
[class*="st-key-cook_undo_save"] button:hover, [class*="st-key-drink_undo_save"] button:hover,
[class*="st-key-cook_option_remix_"] button:hover,
[class*="st-key-drink_option_remix_"] button:hover {
    background: var(--bg-deep) !important; color: var(--ink) !important; border-color: var(--line) !important;
}
[class*="st-key-cook_option_remix_"] button,
[class*="st-key-drink_option_remix_"] button {
    color: var(--ink-2) !important; border-color: var(--line-2) !important;
    background: var(--card) !important; font-weight: 500 !important;
}

/* ── Cook recipe cards ── */
[class*="st-key-cook_recipe_card_"] {
    background: var(--card) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-lg) !important;
    padding: 18px 20px 14px !important;
    margin-bottom: 0 !important;
    overflow: hidden !important;
}

div:has(> [class*="st-key-cook_recipe_card_"]) {
    gap: 16px !important;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
[class*="st-key-cook_recipe_card_"] p,
[class*="st-key-cook_recipe_card_"] li { font-size: 14px !important; line-height: 1.65 !important; }
[class*="st-key-cook_recipe_card_"] h1,
[class*="st-key-cook_recipe_card_"] h2,
[class*="st-key-cook_recipe_card_"] h3 { margin-top: 0 !important; }
.cook-card-title {
    font-family: var(--serif);
    font-weight: 400;
    font-size: 22px;
    line-height: 1.15;
    letter-spacing: -0.01em;
    margin: 0;
    color: var(--ink);
}
.cook-card-status {
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-3);
}
.cook-card-status.saved { color: var(--sage-2); }
.cook-card-status.passed { color: var(--terracotta); }
.cook-card-why {
    color: var(--ink-2);
    font-size: 13.5px;
    line-height: 1.55;
    margin: 4px 0 16px;
}
[class*="st-key-cook_recipe_card_"] .stExpander,
[class*="st-key-cook_recipe_card_"] .stExpander *,
[class*="st-key-cook_recipe_card_"] .stExpander details,
[class*="st-key-cook_recipe_card_"] .stExpander details *,
[class*="st-key-cook_recipe_card_"] .stExpander summary,
[class*="st-key-cook_recipe_card_"] .stExpander [data-testid="stExpanderDetails"],
[class*="st-key-cook_recipe_card_"] .stExpander [data-testid="stExpanderDetails"] > div {
    border: none !important;
    background: transparent !important;
}
[class*="st-key-cook_recipe_card_"] .stExpander summary {
    background: var(--bg) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--line) !important;
    padding: 8px 12px !important;
}
[class*="st-key-cook_recipe_card_"] .stExpander summary,
[class*="st-key-cook_recipe_card_"] .stExpander [data-testid="stExpanderToggleIcon"] {
    color: var(--ink-2) !important;
}
[class*="st-key-cook_recipe_card_"] .stExpander summary:hover {
    color: var(--ink) !important;
}
[class*="st-key-cook_recipe_card_"] .stExpander details[open] [data-testid="stExpanderDetails"] {
    margin: 0 !important;
    padding: 16px 0 0 0 !important;
}
[class*="st-key-cook_recipe_card_"] .stExpander details[open] [data-testid="stElementContainer"] {
    margin-bottom: 0 !important;
    padding-bottom: 20px !important;
}
[class*="st-key-cook_recipe_card_"] .stExpander details[open] div {
    gap: 0 !important;
}
.cook-recipe-details {
    display: flex;
    flex-direction: column;
    gap: 10px;
    color: var(--ink);
    font-size: 14px;
    line-height: 1.45;
}
.cook-recipe-row { margin: 0; }
.cook-recipe-row b,
.cook-recipe-section b { font-weight: 600; }
.cook-recipe-section { margin: 0; }
.cook-recipe-section-title { margin: 0 0 5px; }
.cook-recipe-details ul,
.cook-recipe-details ol {
    margin: 0;
    padding-left: 20px;
}
.cook-recipe-details li {
    margin: 0 0 4px !important;
    padding-left: 2px;
    line-height: 1.45 !important;
}
.cook-recipe-details li:last-child { margin-bottom: 0 !important; }
[class*="st-key-cook_recipe_card_"] .trace-panel {
    margin-top: 0 !important;
}

/* ── Drink recipe cards (same treatment as cook) ── */
[class*="st-key-drink_recipe_card_"] {
    background: var(--card) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0 0 16px 0 !important;
    margin-bottom: 0 !important;
    overflow: hidden !important;
    position: relative !important;
}
div:has(> [class*="st-key-drink_recipe_card_"]) {
    gap: 16px !important;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
[class*="st-key-drink_recipe_card_"] p,
[class*="st-key-drink_recipe_card_"] li { font-size: 14px !important; line-height: 1.65 !important; }
[class*="st-key-drink_recipe_card_"] h1,
[class*="st-key-drink_recipe_card_"] h2,
[class*="st-key-drink_recipe_card_"] h3 { margin-top: 0 !important; }
[class*="st-key-drink_recipe_card_"] > div,
[class*="st-key-drink_recipe_card_"] [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}
[class*="st-key-drink_recipe_card_"] > div,
[class*="st-key-drink_recipe_card_"] [data-testid="stVerticalBlock"],
[class*="st-key-drink_recipe_card_"] [data-testid="stElementContainer"] {
    background: transparent !important;
    position: relative !important;
    z-index: 2 !important;
}
[class*="st-key-drink_recipe_card_"] [data-testid="stElementContainer"]:has(.drink-card-layout) {
    position: static !important;
    z-index: auto !important;
}
.drink-card-layout {
    min-height: 128px;
    background: transparent;
}
.drink-card-image {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    width: var(--drink-card-image-width);
    overflow: hidden;
    background: linear-gradient(135deg, #E89570 0%, #B0552E 100%);
    z-index: 1;
}
.drink-card-image img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center center;
}
.drink-card-image .ph {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(0,0,0,0.35);
}
.drink-card-image.has-photo .ph { display: none; }
.drink-card-body {
    position: relative;
    z-index: 2;
    min-width: 0;
    margin-left: var(--drink-card-image-width);
    padding: 12px 18px 18px;
    display: flex;
    flex-direction: column;
    gap: 7px;
}
.drink-card-title-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
}
.drink-card-title {
    font-family: var(--serif);
    font-weight: 400;
    font-size: 22px;
    line-height: 1.15;
    letter-spacing: -0.01em;
    margin: 0;
    color: var(--ink);
}
.drink-card-status {
    flex-shrink: 0;
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-3);
}
.drink-card-status.saved { color: var(--sage-2); }
.drink-card-status.passed { color: var(--terracotta); }
.drink-card-why {
    color: var(--ink-2);
    font-size: 13.5px;
    line-height: 1.55;
    margin: 0;
}
[class*="st-key-drink_recipe_card_"] .stExpander,
[class*="st-key-drink_recipe_card_"] .trace-panel {
    margin-left: calc(var(--drink-card-image-width) + 18px) !important;
    margin-right: 18px !important;
    width: calc(100% - var(--drink-card-image-width) - 36px) !important;
    max-width: calc(100% - var(--drink-card-image-width) - 36px) !important;
    box-sizing: border-box !important;
}
[class*="st-key-drink_recipe_card_"] [data-testid="stHorizontalBlock"] {
    margin-left: calc(var(--drink-card-image-width) + 18px) !important;
    margin-right: 18px !important;
    width: calc(100% - var(--drink-card-image-width) - 36px) !important;
    max-width: calc(100% - var(--drink-card-image-width) - 36px) !important;
    box-sizing: border-box !important;
}
[class*="st-key-drink_recipe_card_"] .stExpander,
[class*="st-key-drink_recipe_card_"] .stExpander *,
[class*="st-key-drink_recipe_card_"] .stExpander details,
[class*="st-key-drink_recipe_card_"] .stExpander details *,
[class*="st-key-drink_recipe_card_"] .stExpander summary,
[class*="st-key-drink_recipe_card_"] .stExpander [data-testid="stExpanderDetails"],
[class*="st-key-drink_recipe_card_"] .stExpander [data-testid="stExpanderDetails"] > div {
    border: none !important;
    background: transparent !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}
[class*="st-key-drink_recipe_card_"] .stExpander details,
[class*="st-key-drink_recipe_card_"] .stExpander summary,
[class*="st-key-drink_recipe_card_"] .stExpander [data-testid="stExpanderDetails"],
[class*="st-key-drink_recipe_card_"] .stExpander [data-testid="stExpanderDetails"] > div {
    width: 100% !important;
}
[class*="st-key-drink_recipe_card_"] .stExpander summary {
    background: var(--bg) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--line) !important;
    padding: 8px 12px !important;
}
[class*="st-key-drink_recipe_card_"] .stExpander summary,
[class*="st-key-drink_recipe_card_"] .stExpander [data-testid="stExpanderToggleIcon"] {
    color: var(--ink-2) !important;
}
[class*="st-key-drink_recipe_card_"] .stExpander summary:hover { color: var(--ink) !important; }
[class*="st-key-drink_recipe_card_"] .stExpander details[open] [data-testid="stExpanderDetails"] {
    margin: 0 !important;
    padding: 16px 0 0 0 !important;
}
[class*="st-key-drink_recipe_card_"] .stExpander details[open] [data-testid="stElementContainer"] {
    margin-bottom: 0 !important;
    padding-bottom: 20px !important;
}
[class*="st-key-drink_recipe_card_"] .stExpander details[open] div { gap: 0 !important; }
.cocktail-recipe-details {
    display: flex;
    flex-direction: column;
    gap: 10px;
    color: var(--ink);
    font-size: 14px;
    line-height: 1.45;
}
.cocktail-recipe-row {
    margin: 0;
}
.cocktail-recipe-row b,
.cocktail-recipe-section b {
    font-weight: 600;
}
.cocktail-recipe-section {
    margin: 0;
}
.cocktail-recipe-section-title {
    margin: 0 0 5px;
}
.cocktail-recipe-details ul,
.cocktail-recipe-details ol {
    margin: 0;
    padding-left: 20px;
}
.cocktail-recipe-details li {
    margin: 0 0 4px !important;
    padding-left: 2px;
    line-height: 1.45 !important;
}
.cocktail-recipe-details li:last-child {
    margin-bottom: 0 !important;
}
            
[class*="st-key-drink_recipe_card_"] .trace-panel {
    margin-top: 0 !important;
    margin-bottom: 16px !important;
}

.card.combo {
    height: var(--result-card-height) !important;
    min-height: var(--result-card-height) !important;
    margin-bottom: 16px !important;
    border-top-right-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
    border-right: none !important;
}

.card.combo .card-img {
    min-height: var(--result-card-height) !important;
}

.card.combo .card-body {
    overflow: hidden !important;
}

.card.combo .card-blurb {
    display: -webkit-box !important;
    -webkit-line-clamp: 3 !important;
    -webkit-box-orient: vertical !important;
    overflow: hidden !important;
}

[class*="st-key-card_rail_"] {
    background: var(--card) !important;
    border: 1px solid var(--line) !important;
    border-left: none !important;
    border-radius: 0 var(--radius-lg) var(--radius-lg) 0 !important;
    box-shadow: var(--shadow-card) !important;
    overflow: hidden !important;
    height: var(--result-card-height) !important;
    min-height: var(--result-card-height) !important;
    margin: 0 0 16px !important;
}

[class*="st-key-card_rail_accept_"] {
    border-color: rgba(122,158,126,0.5) !important;
}

[class*="st-key-card_rail_"] > div,
[class*="st-key-card_rail_"] [data-testid="stVerticalBlock"],
[class*="st-key-card_rail_"] [data-testid="stVerticalBlockBorderWrapper"] {
    height: var(--result-card-height) !important;
    min-height: var(--result-card-height) !important;
}

[class*="st-key-card_rail_"] [data-testid="stVerticalBlock"] {
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    gap: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    align-items: stretch !important;
}

[class*="st-key-card_rail_"] [data-testid="stElementContainer"] {
    margin: 0 !important;
    padding: 0 !important;
}

[class*="st-key-card_rail_"] [class*="st-key-card_pass_"],
[class*="st-key-card_rail_"] [class*="st-key-card_save_"],
[class*="st-key-card_rail_"] [class*="st-key-card_undo_"],
[class*="st-key-card_rail_"] [data-testid="stButton"] {
    flex: 0 0 var(--result-card-action-height) !important;
    width: 100% !important;
    height: var(--result-card-action-height) !important;
    min-height: var(--result-card-action-height) !important;
    margin: 0 !important;
    padding: 0 !important;
}

[class*="st-key-card_rail_"] [data-testid="stButton"] button {
    width: 100% !important;
    height: var(--result-card-action-height) !important;
    min-height: var(--result-card-action-height) !important;
    max-height: var(--result-card-action-height) !important;
    border-radius: 0 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    outline: none !important;
    border: 0 !important;
    background: transparent !important;
    margin: 0 !important;
    padding: 0 !important;
}

[class*="st-key-card_rail_"] [data-testid="stButton"] button,
[class*="st-key-card_rail_"] [data-testid="stButton"] button > div,
[class*="st-key-card_rail_"] [data-testid="stButton"] button p {
    background: transparent !important;
    box-shadow: none !important;
}

[class*="st-key-card_pass_"] button {
    color: var(--terracotta-2) !important;
    background: transparent !important;
    border-top-right-radius: var(--radius-lg) !important;
}

[class*="st-key-card_pass_"] {
    border-bottom: 1px solid var(--line) !important;
}

[class*="st-key-card_pass_"] button:hover {
    background: var(--terracotta) !important;
    border-color: var(--terracotta) !important;
    color: #fff !important;
}

[class*="st-key-card_save_"] button {
    color: var(--sage-2) !important;
    background: transparent !important;
    border-bottom-right-radius: var(--radius-lg) !important;
}

[class*="st-key-card_save_"] button:hover {
    background: var(--sage) !important;
    border-color: var(--sage) !important;
    color: #fff !important;
}

[class*="st-key-card_rail_"] [class*="st-key-card_undo_"],
[class*="st-key-card_rail_"] [class*="st-key-card_undo_"] [data-testid="stButton"],
[class*="st-key-card_rail_"] [class*="st-key-card_undo_"] button,
[class*="st-key-card_rail_"] [data-testid="stButton"][class*="st-key-card_undo_"],
[class*="st-key-card_rail_"] [data-testid="stButton"][class*="st-key-card_undo_"] button {
    flex-basis: var(--result-card-height) !important;
    height: var(--result-card-height) !important;
    min-height: var(--result-card-height) !important;
    max-height: var(--result-card-height) !important;
}

[class*="st-key-card_rail_"] [class*="st-key-card_undo_"] button,
[class*="st-key-card_rail_"] [data-testid="stButton"][class*="st-key-card_undo_"] button {
    color: var(--ink-2) !important;
    background: transparent !important;
    border-color: var(--line) !important;
    border-top-right-radius: var(--radius-lg) !important;
    border-bottom-right-radius: var(--radius-lg) !important;
}

[class*="st-key-card_undo_accept_"] button:hover {
    background: var(--sage) !important;
    color: #fff !important;
}

[class*="st-key-card_undo_reject_"] button:hover {
    background: var(--terracotta) !important;
    color: #fff !important;
}

.card-feedback-done {
    display: inline-flex; align-items: center; gap: 6px;
    margin-left: auto;
    text-align: right;
    justify-content: flex-end;
    font-family: var(--mono); font-size: 12px;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.card-feedback-done.acc { color: var(--sage-2); }
.card-feedback-done.rej { color: var(--terracotta); }

/* ── Recommendation trace ── */
.trace-panel {
    margin: -8px 0 18px;
    padding: 10px 14px 12px;
    border: 1px dashed var(--line-2);
    border-radius: var(--radius-sm);
    background: rgba(255,255,255,0.50);
    color: var(--ink-2);
}
.trace-panel summary {
    cursor: pointer;
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-3);
    list-style-position: inside;
}
.trace-panel dl {
    display: grid;
    grid-template-columns: minmax(128px, 0.28fr) 1fr;
    gap: 7px 14px;
    margin: 12px 0 0;
}
.trace-panel dt {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-3);
}
.trace-panel dd {
    margin: 0;
    font-size: 13px;
    line-height: 1.45;
    color: var(--ink-2);
}

/* ── Empty state ── */
.empty {
    background: var(--card); border: 1px dashed var(--line-2);
    border-radius: var(--radius-lg); padding: 48px 32px;
    text-align: center;
    display: flex; flex-direction: column; align-items: center; gap: 12px;
}
.empty .glyph {
    width: 56px; height: 56px; border-radius: 50%;
    background: var(--tag-terracotta); color: var(--terracotta);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--serif); font-size: 28px; margin-bottom: 4px;
}
.empty h3 {
    font-family: var(--serif); font-weight: 400; font-size: 22px;
    letter-spacing: -0.01em; margin: 0;
}
.empty p { color: var(--ink-2); margin: 0; max-width: 38ch; font-size: 14px; }
.empty .quick-row { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-top: 6px; }

/* ── Skeleton ── */
.skeleton {
    background: var(--card); border-radius: var(--radius-lg);
    border: 1px solid var(--line);
    display: grid; grid-template-columns: 168px 1fr;
    overflow: hidden; min-height: 180px; margin-bottom: 16px;
}
.skeleton .img {
    background: linear-gradient(90deg, var(--bg-deep) 0%, #EFE8DA 50%, var(--bg-deep) 100%);
    background-size: 200% 100%; animation: shimmer 1.4s linear infinite;
}
.skeleton .body { padding: 20px; display: flex; flex-direction: column; gap: 10px; }
.skeleton .bar {
    height: 12px; border-radius: 4px;
    background: linear-gradient(90deg, var(--bg-deep) 0%, #EFE8DA 50%, var(--bg-deep) 100%);
    background-size: 200% 100%; animation: shimmer 1.4s linear infinite;
}
.skeleton .bar.title { height: 22px; width: 60%; }
.skeleton .bar.short { width: 40%; }
.skeleton .bar.medium { width: 75%; }
@keyframes shimmer { 0% { background-position: 100% 0; } 100% { background-position: -100% 0; } }

/* ── LLM response ── */
[class*="st-key-llm_response_"] > div {
    background: linear-gradient(180deg, #FFFBF5 0%, #FAF3E8 100%);
    border: 1px solid rgba(201,162,39,0.2);
    border-radius: var(--radius); padding: 18px 22px;
    margin: 8px 0 8px;
    font-size: 14.5px; line-height: 1.65; color: var(--ink);
}

.stSpinner > div { border-top-color: var(--terracotta) !important; }

/* Search card — applied directly to the form container */
[data-testid="stForm"] {
    background: var(--card) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-lg) !important;
    padding: 22px 24px 20px !important;
    box-shadow: var(--shadow-card) !important;
    margin-bottom: 12px !important;
    margin-top: 12px !important;
}
[data-testid="stForm"] [data-testid="stVerticalBlock"],
[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
    background: transparent !important;
}

/* Hide form submit's default appearance when used as a chip */
.stForm [data-testid="stFormSubmitButton"] { background: transparent !important; padding: 0 !important; border: none !important; box-shadow: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_gradient_class(categories):
    if not categories:
        return "amber-gold"
    for cat in categories[:2]:
        for key, grad in CUISINE_GRADIENTS.items():
            if key in cat.lower():
                return grad
    return "amber-gold"


def stars_html(rating):
    if not rating:
        return ""
    full = max(0, min(5, round(float(rating))))
    empty = 5 - full
    return "★" * full + "☆" * empty


def stable_widget_key(*parts):
    raw = "::".join(str(part or "") for part in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]


def normalized_lookup_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def donut_svg(eat, cook, drink, total):
    if total == 0:
        return '<svg class="donut" viewBox="0 0 64 64"><circle cx="32" cy="32" r="28" fill="none" stroke="#F2EDE3" stroke-width="8"/><text x="32" y="36" text-anchor="middle" class="donut-text">0</text></svg>'
    r = 28
    C = 2 * math.pi * r
    s = max(eat + cook + drink, 1)
    el = C * eat / s
    cl = C * cook / s
    dl = C * drink / s

    def arc(color, dl_arc, offset):
        return f'<circle cx="32" cy="32" r="{r}" fill="none" stroke="{color}" stroke-width="8" stroke-dasharray="{dl_arc:.1f} {C - dl_arc:.1f}" stroke-dashoffset="{-offset:.1f}"/>'

    return f'''<svg class="donut" viewBox="0 0 64 64">
<circle cx="32" cy="32" r="{r}" fill="none" stroke="#F2EDE3" stroke-width="8"/>
<g transform="rotate(-90 32 32)">
{arc("#C96A3A", el, 0)}
{arc("#7A9E7E", cl, el)}
{arc("#C9A227", dl, el + cl)}
</g>
<text x="32" y="36" text-anchor="middle" class="donut-text">{total}</text>
</svg>'''


def _split_cook_recipes(response_text):
    blocks = re.split(r'\n(?=\*{0,2}RECIPE\*{0,2}\s*:)', response_text.strip(), flags=re.IGNORECASE)
    results = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        name_m = re.match(r'\*{0,2}RECIPE\*{0,2}\s*:\s*\*{0,2}(.+?)\*{0,2}\s*$', block, re.IGNORECASE | re.MULTILINE)
        if not name_m:
            continue
        name = name_m.group(1).strip()
        why_m = re.search(r'\*{0,2}WHY IT FITS\*{0,2}\s*:\s*(.+?)(?=\n\s*\*{0,2}[A-Z]|\Z)', block, re.IGNORECASE | re.DOTALL)
        why = why_m.group(1).strip() if why_m else ""
        results.append((name, why, block))
    if not results:
        return [(extract_generated_item_name(response_text, "cook"), "", response_text)]
    return results


def _format_cook_recipe_for_expander(recipe_block):
    text = str(recipe_block or "").strip()
    if not text:
        return ""

    text = re.sub(r'(?im)^\s*\*{0,2}RECIPE\*{0,2}\s*:\s*.+\n?', '', text)
    text = re.sub(
        r'(?ims)^\s*\*{0,2}WHY IT FITS\*{0,2}\s*:\s*.*?(?=^\s*\*{0,2}(?:USES FROM PANTRY|MISSING OR SUBSTITUTE INGREDIENTS|INGREDIENTS|STEPS|CAUTION)\*{0,2}\s*:|\Z)',
        '',
        text,
    )
    text = re.sub(r'(?im)^\s*\*{0,2}CAUTION\*{0,2}\s*:\s*none\.?\s*$', '', text)

    labels = ["USES FROM PANTRY", "MISSING OR SUBSTITUTE INGREDIENTS", "INGREDIENTS", "STEPS", "CAUTION"]
    label_pattern = "|".join(re.escape(label) for label in labels)
    pattern = (
        r'(?ims)^\s*\*{0,2}(' + label_pattern + r')\*{0,2}\s*:\s*'
        r'(.*?)(?=^\s*\*{0,2}(?:' + label_pattern + r')\*{0,2}\s*:|\Z)'
    )
    sections = {
        label.upper(): " ".join(value.strip().split()) if label.upper() not in {"INGREDIENTS", "STEPS"} else value.strip()
        for label, value in re.findall(pattern, text)
    }

    def esc(value):
        cleaned = re.sub(r'[*_`]+', '', str(value or ""))
        cleaned = re.sub(r'^\s*[-–—]+\s*$', '', cleaned)
        return html_module.escape(cleaned.strip())

    def clean_scalar(value):
        cleaned = re.sub(r'[*_`]+', '', str(value or ""))
        cleaned = re.sub(r'^\s*[:\-–—]+\s*', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def list_items(value, ordered=False):
        lines = []
        for line in str(value or "").splitlines():
            line = re.sub(r'^\s*(?:[-*]\s*|\d+[.)]\s*)', '', line).strip()
            line = re.sub(r'[*_`]+', '', line).strip()
            if re.fullmatch(r'[-–—]+', line):
                continue
            if line:
                lines.append(f"<li>{esc(line)}</li>")
        if not lines:
            return ""
        tag = "ol" if ordered else "ul"
        return f"<{tag}>{''.join(lines)}</{tag}>"

    parts = ['<div class="cook-recipe-details">']
    for label, pretty in [
        ("USES FROM PANTRY", "Uses from pantry"),
        ("MISSING OR SUBSTITUTE INGREDIENTS", "Missing or substitute ingredients"),
    ]:
        value = clean_scalar(sections.get(label, ""))
        if value and value.lower() not in {"none", "n/a", "na"}:
            parts.append(f'<p class="cook-recipe-row"><b>{pretty}:</b> {esc(value)}</p>')

    ingredients_html = list_items(sections.get("INGREDIENTS", ""))
    if ingredients_html:
        parts.append(f'<div class="cook-recipe-section"><p class="cook-recipe-section-title"><b>Ingredients</b></p>{ingredients_html}</div>')

    steps_html = list_items(sections.get("STEPS", ""), ordered=True)
    if steps_html:
        parts.append(f'<div class="cook-recipe-section"><p class="cook-recipe-section-title"><b>Steps</b></p>{steps_html}</div>')

    caution = clean_scalar(sections.get("CAUTION", ""))
    if caution and caution.lower() not in {"none", "n/a", "na"}:
        parts.append(f'<p class="cook-recipe-row"><b>Caution:</b> {esc(caution)}</p>')

    parts.append("</div>")
    return "".join(parts)


def _split_cocktail_recipes(response_text):
    blocks = re.split(r'\n(?=\*{0,2}COCKTAIL\*{0,2}\s*:)', response_text.strip(), flags=re.IGNORECASE)
    results = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        name_m = re.match(r'\*{0,2}COCKTAIL\*{0,2}\s*:\s*\*{0,2}(.+?)\*{0,2}\s*$', block, re.IGNORECASE | re.MULTILINE)
        if not name_m:
            continue
        name = name_m.group(1).strip()
        why_m = re.search(r'\*{0,2}WHY IT FITS\*{0,2}\s*:\s*(.+?)(?=\n\s*\*{0,2}[A-Z]|\Z)', block, re.IGNORECASE | re.DOTALL)
        why = why_m.group(1).strip() if why_m else ""
        results.append((name, why, block))
    if not results:
        return [(extract_generated_item_name(response_text, "drink"), "", response_text)]
    return results


def _format_cocktail_recipe_for_expander(cocktail_block):
    text = str(cocktail_block or "").strip()
    if not text:
        return ""

    # The model returns parser labels for card extraction; hide those in the expander.
    text = re.sub(r'(?im)^\s*\*{0,2}COCKTAIL\*{0,2}\s*:\s*.+\n?', '', text)
    text = re.sub(
        r'(?ims)^\s*\*{0,2}WHY IT FITS\*{0,2}\s*:\s*.*?(?=^\s*\*{0,2}(?:GLASS|ICE|INGREDIENTS|METHOD|GARNISH|SUBSTITUTIONS|NOTE)\*{0,2}\s*:|\Z)',
        '',
        text,
    )

    labels = ["GLASS", "ICE", "INGREDIENTS", "METHOD", "GARNISH", "SUBSTITUTIONS", "NOTE"]
    pattern = (
        r'(?ims)^\s*\*{0,2}(' + "|".join(labels) + r')\*{0,2}\s*:\s*'
        r'(.*?)(?=^\s*\*{0,2}(?:' + "|".join(labels) + r')\*{0,2}\s*:|\Z)'
    )
    sections = {
        label.upper(): " ".join(value.strip().split()) if label.upper() not in {"INGREDIENTS", "METHOD"} else value.strip()
        for label, value in re.findall(pattern, text)
    }

    def esc(value):
        return html_module.escape(str(value or "").strip())

    def list_items(value, ordered=False):
        lines = []
        for line in str(value or "").splitlines():
            line = re.sub(r'^\s*(?:[-*]\s*|\d+[.)]\s*)', '', line).strip()
            if line:
                lines.append(f"<li>{esc(line)}</li>")
        if not lines:
            return ""
        tag = "ol" if ordered else "ul"
        return f"<{tag}>{''.join(lines)}</{tag}>"

    parts = ['<div class="cocktail-recipe-details">']
    for label, pretty in [("GLASS", "Glass"), ("ICE", "Ice")]:
        value = sections.get(label, "")
        if value and value.lower() not in {"none", "n/a", "na"}:
            parts.append(f'<p class="cocktail-recipe-row"><b>{pretty}:</b> {esc(value)}</p>')

    ingredients_html = list_items(sections.get("INGREDIENTS", ""))
    if ingredients_html:
        parts.append(f'<div class="cocktail-recipe-section"><p class="cocktail-recipe-section-title"><b>Ingredients</b></p>{ingredients_html}</div>')

    method_html = list_items(sections.get("METHOD", ""), ordered=True)
    if method_html:
        parts.append(f'<div class="cocktail-recipe-section"><p class="cocktail-recipe-section-title"><b>Method</b></p>{method_html}</div>')

    for label, pretty in [("GARNISH", "Garnish"), ("SUBSTITUTIONS", "Substitutions"), ("NOTE", "Note")]:
        value = sections.get(label, "")
        if value and value.lower() not in {"none", "n/a", "na"}:
            parts.append(f'<p class="cocktail-recipe-row"><b>{pretty}:</b> {esc(value)}</p>')

    parts.append("</div>")
    return "".join(parts)


def match_indicator(inventory, response_text):
    if not inventory or not response_text:
        return None, None
    matched = sum(1 for item in inventory if item.lower() in response_text.lower())
    total = len(inventory)
    if matched == total:
        return f"All {total} items on hand", "good"
    elif matched > 0:
        return f"{matched} of {total} items on hand", "warn"
    return None, None


def _clean_trace_list(values, limit=5):
    cleaned = []
    seen = set()
    for value in values or []:
        text = " ".join(str(value or "").strip().split())
        key = text.lower()
        if text and key not in seen:
            cleaned.append(text)
            seen.add(key)
        if len(cleaned) >= limit:
            break
    return cleaned


def _trace_join(values, empty="none"):
    values = _clean_trace_list(values)
    return ", ".join(values) if values else empty


def _clip_trace(value, limit=180):
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _strip_trace_markdown(value, blank_if_none=False):
    text = re.sub(r"[*_`]+", "", str(value or ""))
    text = re.sub(r"^\s*[:\-–]+\s*", "", text)
    text = " ".join(text.split())
    if blank_if_none and text.lower() in {"none", "n/a", "na", "not applicable", "no"}:
        return ""
    return text


def _profile_trace_matches(text):
    profile = st.session_state.get("profile", {})
    text_lower = str(text or "").lower()
    preferred = _clean_trace_list(profile.get("preferred_cuisines", []), limit=8)
    liked = _clean_trace_list(profile.get("liked_foods", []), limit=8)
    disliked = _clean_trace_list(profile.get("disliked_foods", []), limit=8)

    matched = []
    for item in preferred + liked:
        if item.lower() in text_lower:
            matched.append(item)
    conflicts = [item for item in disliked if item.lower() in text_lower]
    return _clean_trace_list(matched, limit=6), _clean_trace_list(conflicts, limit=4)


def _response_field_values(response_text, label):
    pattern = rf"^\s*(?:[-*]\s*)?\*{{0,2}}{re.escape(label)}\*{{0,2}}\s*:\s*(.+)$"
    values = re.findall(pattern, response_text or "", flags=re.IGNORECASE | re.MULTILINE)
    cleaned = [_strip_trace_markdown(value, blank_if_none=True) for value in values]
    return [_clip_trace(value, limit=140) for value in cleaned if value]


def _fallback_generated_name(request_text, suffix):
    base = _strip_trace_markdown(request_text)
    base = re.sub(r"\b(make|give me|i want|suggest|recommend|recipe|cocktail|drink)\b", "", base, flags=re.IGNORECASE)
    base = re.sub(r"[^a-zA-Z0-9 &,'-]+", " ", base)
    base = " ".join(base.split()).strip(" ,-")
    if not base:
        return f"Generated {suffix}"
    base = base[:52].strip(" ,-")
    if suffix.lower() not in base.lower():
        base = f"{base} {suffix}"
    return base[:64].strip()


def _clean_generated_name(value):
    text = _strip_trace_markdown(value)
    text = re.split(
        r"\s+(?:WHY IT FITS|USES FROM PANTRY|MISSING OR SUBSTITUTE INGREDIENTS|QUICK STEPS|CAUTION|INGREDIENTS|METHOD|FLAVOR LOGIC)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return text.strip(" .:-")[:64]


def extract_generated_item_name(response_text, kind, request_text=""):
    labels = ["RECIPE"] if kind == "cook" else ["COCKTAIL", "COCKTAIL NAME", "DRINK", "DRINK NAME"]
    for label in labels:
        values = _response_field_values(response_text, label)
        if values:
            return _clean_generated_name(values[0])

    heading_match = re.search(r"^#{1,3}\s+(.+)$", response_text or "", re.MULTILINE)
    if heading_match:
        return _clean_generated_name(heading_match.group(1))

    bold_match = re.search(r"^\s*\*\*(?:Cocktail Name|Recipe|Drink):?\*\*\s*(.+)$", response_text or "", re.IGNORECASE | re.MULTILINE)
    if bold_match:
        return _clean_generated_name(bold_match.group(1))

    suffix = "recipe" if kind == "cook" else "drink"
    return _fallback_generated_name(request_text, suffix)


def extract_generated_item_names(response_text, kind, request_text=""):
    text = response_text or ""
    if kind == "cook":
        labels = ["RECIPE"]
    else:
        labels = ["COCKTAIL", "COCKTAIL NAME", "DRINK", "DRINK NAME"]

    names = []
    for label in labels:
        pattern = (
            rf"^\s*(?:\d+[.)]\s*)?(?:[-*]\s*)?\*{{0,2}}{re.escape(label)}\*{{0,2}}\s*:\s*"
            rf"(.+?)(?=\s+\*{{0,2}}(?:WHY IT FITS|USES FROM PANTRY|MISSING OR SUBSTITUTE INGREDIENTS|"
            rf"QUICK STEPS|CAUTION|INGREDIENTS|METHOD|FLAVOR LOGIC)\*{{0,2}}\s*:|$)"
        )
        names.extend(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))

    if not names:
        names.extend(re.findall(r"^#{1,3}\s+(.+)$", text, flags=re.MULTILINE))

    cleaned = []
    seen = set()
    for name in names:
        value = _clean_generated_name(name)
        key = value.lower()
        if value and key not in seen:
            cleaned.append(value)
            seen.add(key)

    if not cleaned:
        cleaned = [extract_generated_item_name(text, kind, request_text)]
    return cleaned[:5]


def render_generated_option_feedback(kind, option_names):
    tab = "cook" if kind == "cook" else "drink"
    noun = "recipe" if kind == "cook" else "drink"
    names = _clean_trace_list(option_names, limit=5)
    if not names:
        return

    if len(names) > 1:
        st.markdown(
            f'<div class="field-label" style="margin-top:14px">Save a specific {noun}</div>',
            unsafe_allow_html=True,
        )

    for idx, name in enumerate(names, start=1):
        accepted = name in st.session_state.profile.get("accepted", [])
        rejected = name in st.session_state.profile.get("rejected", [])
        key_base = stable_widget_key(tab, name, idx)

        label_col, pass_col, save_col = st.columns([3.4, 1, 1])
        with label_col:
            status = "Saved" if accepted else "Passed" if rejected else ""
            status_html = f' <span style="color:#9A968D">· {html_module.escape(status)}</span>' if status else ""
            st.markdown(f"**{idx}. {html_module.escape(name)}**{status_html}", unsafe_allow_html=True)
        with pass_col:
            if accepted:
                st.write("")
            elif rejected:
                if st.button("Undo Pass", key=f"{tab}_option_undo_pass_{key_base}", use_container_width=True):
                    st.session_state.active_tab = tab
                    undo_card_feedback(name, False, tab=tab)
                    st.rerun()
            else:
                if st.button("Pass", key=f"{tab}_option_pass_{key_base}", use_container_width=True):
                    st.session_state.active_tab = tab
                    apply_card_feedback(name, False, tab=tab)
                    st.rerun()
        with save_col:
            if accepted:
                if st.button("Undo Save", key=f"{tab}_option_undo_save_{key_base}", use_container_width=True):
                    st.session_state.active_tab = tab
                    undo_card_feedback(name, True, tab=tab)
                    st.rerun()
            elif rejected:
                st.write("")
            else:
                if st.button("Save", key=f"{tab}_option_save_{key_base}", use_container_width=True):
                    st.session_state.active_tab = tab
                    apply_card_feedback(name, True, tab=tab)
                    st.rerun()


def render_trace_panel(rows, title="Why this recommendation?"):
    visible = []
    for label, value in rows:
        if isinstance(value, (list, tuple, set)):
            value = _trace_join(value)
        value = _clip_trace(value, limit=260)
        if value:
            visible.append((label, value))
    if not visible:
        return

    body = "".join(
        f"<dt>{html_module.escape(label)}</dt><dd>{html_module.escape(value)}</dd>"
        for label, value in visible
    )
    st.markdown(
        f'<details class="trace-panel">'
        f'<summary>{html_module.escape(title)}</summary>'
        f'<dl>{body}</dl>'
        f'</details>',
        unsafe_allow_html=True,
    )


def _restaurant_source_label(row):
    if row.get("match_source") == "static_exact_cuisine":
        return "Static restaurant dataset exact cuisine match"
    if row.get("match_source") == "static_sparse_cuisine_fallback":
        return "Static restaurant dataset closest cuisine fallback"
    if row.get("source") == "static_rag" or row.get("retrieval_score") is not None:
        return "Static restaurant dataset + embedding retrieval/RAG fallback"
    if row.get("fsq_id") or row.get("photo_url") or row.get("address"):
        return "Live Google Places result ranked with LLM + taste profile context"
    return "Recommendation result from the current app pipeline"


def render_restaurant_trace(row, blurb=""):
    categories = row.get("categories", []) or ([row.get("category")] if row.get("category") else [])
    evidence_parts = []
    if categories:
        evidence_parts.append(f"category: {_trace_join(categories[:2])}")
    if row.get("rating"):
        reviews = row.get("total_tips", 0)
        evidence_parts.append(f"rating: {row.get('rating')}" + (f" from {reviews} reviews" if reviews else ""))
    if row.get("photo_url"):
        evidence_parts.append("live photo available")
    if row.get("open_now") is not None:
        evidence_parts.append("live hours available")
    if row.get("retrieval_score") is not None:
        evidence_parts.append(f"retrieval score: {row.get('retrieval_score')}")
    if row.get("popular_food"):
        evidence_parts.append(f"popular food: {row.get('popular_food')}")
    if row.get("match_note"):
        evidence_parts.append(row.get("match_note"))

    trace_text = " ".join([
        row.get("name") or row.get("title", ""),
        " ".join(categories),
        str(row.get("popular_food", "")),
        str(row.get("attributes", "")),
        blurb,
    ])
    profile_matches, conflicts = _profile_trace_matches(trace_text)
    source = _restaurant_source_label(row)
    if "Google Places" in source:
        photo_status = "photo shown from Google Places" if row.get("photo_url") else "photo not returned for this place"
        hours_status = "hours available" if row.get("open_now") is not None else "hours not returned"
        limitation = f"{photo_status}; {hours_status}. Verify live details before visiting."
    elif "exact cuisine match" in source:
        limitation = "Local static match; live photo, hours, and current availability require Google Places."
    else:
        limitation = "Static fallback has no live photo, live hours, or current availability data."
    if conflicts:
        limitation = f"Potential profile conflict detected: {_trace_join(conflicts)}."

    render_trace_panel(
        [
            ("Source", source),
            ("Evidence used", "; ".join(evidence_parts) if evidence_parts else "restaurant metadata and current query"),
            ("Profile match", _trace_join(profile_matches, "no direct saved-preference tag matched")),
            ("LLM rationale", blurb or "No generated blurb was available for this result."),
            ("Limitation", limitation),
        ]
    )


def render_generation_trace(kind, response_text, inventory, request_text):
    inventory = _clean_trace_list(inventory, limit=12)
    used_from_inventory = [
        item for item in inventory
        if item.lower() in str(response_text or "").lower()
    ]
    _, conflicts = _profile_trace_matches(response_text)

    if kind == "cook":
        explicit_used = _response_field_values(response_text, "USES FROM PANTRY")
        missing = _response_field_values(response_text, "MISSING OR SUBSTITUTE INGREDIENTS")
        cautions = _response_field_values(response_text, "CAUTION")
        source = "LLM + taste profile generation; no recipe dataset/RAG grounding is claimed"
        limitation = (
            _trace_join(cautions[:2], "model-generated recipe idea; confirm allergens, doneness, and safety details")
        )
        rows = [
            ("Source", source),
            ("Request context", request_text or "not specified"),
            ("Pantry matched", _trace_join(used_from_inventory) if used_from_inventory else _trace_join(explicit_used, "not explicitly listed")),
            ("Missing/substitute", _trace_join(missing[:3], "none listed")),
            ("Profile role", "Request and pantry are primary; saved taste profile is secondary context."),
            ("Limitation", f"Potential conflict: {_trace_join(conflicts)}" if conflicts else limitation),
        ]
    else:
        grounding = st.session_state.get("drink_grounding", []) or []
        generated_names = extract_generated_item_names(response_text, "drink", request_text)
        generated_keys = {normalized_lookup_key(name) for name in generated_names}
        grounded_match = next(
            (item for item in grounding if normalized_lookup_key(item.get("name", "")) in generated_keys),
            None,
        )
        if grounded_match:
            source = "CocktailDB grounded recipe + LLM ranking/formatting from bar inventory"
            evidence = (
                f"{grounded_match.get('name')} from CocktailDB; "
                f"{grounded_match.get('have_count', 0)}/{grounded_match.get('total_ingredients', 0)} ingredients matched"
            )
            limitation = "CocktailDB provides recipe grounding; confirm measures and substitutions before mixing."
        elif grounding:
            source = "CocktailDB candidate retrieval + LLM selection from bar inventory"
            evidence = f"{len(grounding)} CocktailDB candidates retrieved; selected recipe may be adapted."
            limitation = "LLM may adapt from retrieved candidates; confirm measures and substitutions before mixing."
        else:
            source = "LLM + bar inventory generation; no CocktailDB record was retrieved"
            evidence = "No CocktailDB candidates were available for this bar inventory."
            limitation = "Model-generated drink idea; confirm measurements and avoid ingredients you cannot consume."
        rows = [
            ("Source", source),
            ("CocktailDB evidence", evidence),
            ("Request context", request_text or "not specified"),
            ("Inventory matched", _trace_join(used_from_inventory, "not explicitly listed")),
            ("Profile role", "Requested vibe and bar inventory are primary; saved taste profile is secondary context."),
            ("Limitation", f"Potential conflict: {_trace_join(conflicts)}" if conflicts else limitation),
        ]

    render_trace_panel(rows)


def refresh_preference_tags(profile):
    accepted = profile.get("accepted", [])
    rejected = profile.get("rejected", [])
    history = profile.get("history", [])
    if len(accepted) + len(rejected) == 0 and not history:
        profile["liked_foods"] = []
        profile["disliked_foods"] = []
        return profile

    payload = {
        "accepted": accepted[-24:],
        "rejected": rejected[-24:],
        "history": history[-24:],
        "preferred_cuisines": profile.get("preferred_cuisines", []),
        "cuisine_scores": profile.get("cuisine_scores", {}),
        "food_scores": profile.get("food_scores", {}),
        "budget": profile.get("budget", ""),
        "occasion": profile.get("occasion", ""),
    }
    system_prompt = (
        "You are updating a user's food preference tags based on their full interaction history. "
        "Return JSON only with keys liked_foods and disliked_foods. "
        "Generate a fresh, accurate list of up to 8 short tags per list, reflecting the user's current tastes. "
        "You do not have to use all 8 tags unless they are all relevant and informative. "
        "Prefer specific cuisines, dishes, ingredients, settings, and vibes. "
        "Weight recent history more heavily than older entries. "
        "Do not use restaurant names as tags. Do not pad with generic tags."
    )
    user_prompt = json.dumps(payload, ensure_ascii=False)

    try:
        response = get_client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        profile["liked_foods"] = clean_preference_tags(data.get("liked_foods", []))
        profile["disliked_foods"] = clean_preference_tags(data.get("disliked_foods", []))
    except Exception:
        fallback = fallback_preference_tags(profile)
        profile["liked_foods"] = fallback["liked_foods"]
        profile["disliked_foods"] = fallback["disliked_foods"]
    return profile


def clean_preference_tags(tags):
    cleaned = []
    seen = set()
    for tag in tags or []:
        text = str(tag).strip().strip("-•")
        if not text:
            continue
        text = " ".join(text.split())
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text[:36])
        if len(cleaned) >= MAX_PREFERENCE_TAGS:
            break
    return cleaned


def fallback_preference_tags(profile):
    likes = []
    dislikes = []
    for name, score in sorted(profile.get("cuisine_scores", {}).items(), key=lambda item: -item[1]):
        if score > 0:
            likes.append(name.replace(" restaurant", "").replace(" Restaurant", ""))
        elif score < 0:
            dislikes.append(name.replace(" restaurant", "").replace(" Restaurant", ""))
    likes.extend(profile.get("liked_foods", []))
    dislikes.extend(profile.get("disliked_foods", []))
    return {
        "liked_foods": clean_preference_tags(likes),
        "disliked_foods": clean_preference_tags(dislikes),
    }


def reset_taste_profile():
    if os.path.exists(PROFILE_PATH):
        os.remove(PROFILE_PATH)
    os.makedirs(os.path.dirname(PROFILE_RESET_MARKER), exist_ok=True)
    with open(PROFILE_RESET_MARKER, "w") as f:
        f.write("1")
    profile = load_profile()
    profile["liked_foods"] = []
    profile["disliked_foods"] = []
    save_profile(profile)
    st.session_state.profile = profile
    st.session_state.sample_profile_disabled = True
    st.session_state.eat_results = None
    st.session_state.eat_fsq_results = None
    st.session_state.eat_llm_response = None
    st.session_state.cook_response = None
    st.session_state.cocktail_response = None


def apply_card_feedback(name, accepted, cuisines=None, tab="eat", price=None):
    opposite_bucket = "rejected" if accepted else "accepted"
    if name in st.session_state.profile.get(opposite_bucket, []):
        st.session_state.profile[opposite_bucket].remove(name)
    preserved_liked_foods = st.session_state.profile.get("liked_foods", [])
    preserved_disliked_foods = st.session_state.profile.get("disliked_foods", [])
    st.session_state.profile = update_profile(
        st.session_state.profile,
        restaurant_name=name,
        accepted=accepted,
        cuisines=cuisines or None,
        price=price,
    )
    st.session_state.profile["liked_foods"] = preserved_liked_foods
    st.session_state.profile["disliked_foods"] = preserved_disliked_foods
    st.session_state.profile.setdefault("history", []).append({"name": name, "kind": "acc" if accepted else "rej", "tab": tab})
    if accepted:
        _tc = st.session_state.profile.setdefault("tab_counts", {"eat": 0, "cook": 0, "drink": 0})
        _tc[tab] = _tc.get(tab, 0) + 1
    save_profile(st.session_state.profile)


def undo_card_feedback(name, was_accepted, cuisines=None, tab="eat"):
    bucket = "accepted" if was_accepted else "rejected"
    kind = "acc" if was_accepted else "rej"
    if name in st.session_state.profile.get(bucket, []):
        st.session_state.profile[bucket].remove(name)

    delta = -0.15 if was_accepted else 0.15
    for cuisine in cuisines or []:
        current = st.session_state.profile.get("cuisine_scores", {}).get(cuisine, 0.0)
        st.session_state.profile["cuisine_scores"][cuisine] = round(max(-1.0, min(1.0, current + delta)), 3)

    st.session_state.profile["preferred_cuisines"] = [
        k for k, v in st.session_state.profile.get("cuisine_scores", {}).items() if v > 0.2
    ]
    st.session_state.profile["history"] = [
        h for h in st.session_state.profile.get("history", [])
        if not (h.get("name") == name and h.get("kind") == kind and h.get("tab") == tab)
    ]
    if was_accepted:
        tc = st.session_state.profile.setdefault("tab_counts", {"eat": 0, "cook": 0, "drink": 0})
        tc[tab] = max(0, tc.get(tab, 0) - 1)
    save_profile(st.session_state.profile)


def render_reset_button():
    st.button(
        "Reset taste profile  ↺",
        key="reset_taste_profile",
        on_click=reset_taste_profile,
        use_container_width=True,
    )


def remove_preference_tag(list_key, food):
    if food in st.session_state.profile.get(list_key, []):
        st.session_state.profile[list_key].remove(food)
        st.session_state.profile.get("food_scores", {}).pop(food, None)
        save_profile(st.session_state.profile)


# ── Session state ─────────────────────────────────────────────────────────────
def init_session():
    if "profile" not in st.session_state:
        profile = load_profile()
        sample_profile_disabled = os.path.exists(PROFILE_RESET_MARKER)
        st.session_state.profile = profile
        st.session_state.sample_profile_disabled = sample_profile_disabled

    defaults = {
        "eat_results": None, "eat_fsq_results": None, "eat_llm_response": None,
        "cook_response": None, "cocktail_response": None,
        "feedback_given": set(),
        "hint_dismissed": False,
        "eat_prefill": "", "cook_prefill": "", "drink_prefill": "",
        "active_tab": "eat",
        "cook_last_craving": "", "drink_last_vibe": "",
        "cook_remix_active": False, "drink_remix_active": False,
        "cook_remix_pending": None, "drink_remix_pending": None,
        "cook_remix_card": None, "cook_remix_previous": None,
        "drink_remix_card": None, "drink_grounding": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Query param handler ───────────────────────────────────────────────────────
def handle_query_params():
    qp = st.query_params

    if "action" in qp:
        action = qp["action"]
        name = qp.get("name", "")
        tab = qp.get("tab", "eat")
        cuisine = qp.get("cuisine", "")
        st.session_state.active_tab = tab

        if action in ("accept", "reject") and name:
            accepted = action == "accept"
            cuisines = [cuisine] if cuisine else None
            st.session_state.profile = update_profile(
                st.session_state.profile, restaurant_name=name,
                accepted=accepted, cuisines=cuisines,
            )
            st.session_state.profile.setdefault("history", []).append({"name": name, "kind": "acc" if accepted else "rej", "tab": tab})
            _tc = st.session_state.profile.setdefault("tab_counts", {"eat": 0, "cook": 0, "drink": 0})
            _tc[tab] = _tc.get(tab, 0) + 1
            save_profile(st.session_state.profile)

        elif action == "prefill":
            target = qp.get("tab", "eat")
            text = qp.get("text", "")
            st.session_state.active_tab = target
            st.session_state[f"{target}_prefill"] = text

        elif action == "dismiss_hint":
            st.session_state.hint_dismissed = True

        elif action == "reset":
            reset_taste_profile()

        st.query_params.clear()


def render_preference_tags(label, profile_key, tone):
    items = st.session_state.profile.get(profile_key, [])
    if not items:
        return

    pill_class = "sage" if tone == "liked" else "terracotta"
    pills = "".join([
        f'<span class="pill {pill_class}">'
        f'<span class="pill-bar"></span>{html_module.escape(item)}'
        f'</span>'
        for item in items
    ])

    st.markdown(
        f'<div class="side-section">'
        f'<div class="side-label"><span>{html_module.escape(label)}</span><span class="count">{len(items)}</span></div>'
        f'<div class="tag-cluster">{pills}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    profile = st.session_state.profile

    # Brand
    sidebar_html = '<div class="brand"><div class="brand-mark"></div><div class="brand-name">Food<em>AI</em></div></div>'

    # Cuisine pulse
    cs = profile.get("cuisine_scores", {})
    if cs:
        GENERIC_SKIP = {"restaurant", "food", "bar", "cafe", "bistro"}
        filtered_c = [
            (k, v) for k, v in cs.items()
            if k.lower().strip() not in GENERIC_SKIP
        ]
        sorted_c = sorted(filtered_c, key=lambda x: -x[1])[:5]
        max_score = max((v for _, v in sorted_c if v > 0), default=1)
        rows = ""
        for name, score in sorted_c:
            display_name = " ".join(
                part for part in name.split()
                if part.lower() != "restaurant"
            ).strip() or name
            pct = max(0, min(100, int((score / max_score) * 100)))
            rows += (
                f'<div class="cuisine-row">'
                f'<span class="name">{html_module.escape(display_name)}</span>'
                f'<span class="cuisine-bar"><i style="width:{pct}%"></i></span>'
                f'<span class="pct">{pct}</span>'
                f'</div>'
            )
        if sorted_c:
            sidebar_html += (
                f'<div class="side-section">'
                f'<div class="side-label"><span>Cuisine pulse</span><span class="count">Top {len(sorted_c)}</span></div>'
                f'<div class="cuisine-list">{rows}</div>'
                f'</div>'
            )
    sidebar_html += '<!--PREF_TAGS-->'

    # Inferred spending pattern
    inferred = profile.get("inferred_budget_level")
    inferred_count = profile.get("inferred_budget_count", 0)
    if inferred is not None and inferred_count >= 2:
        inferred_level = max(1, min(4, round(inferred)))
        inferred_dollar = "$" * inferred_level
        idots = "".join([
            f'<span class="dot{" on" if n <= inferred_level else ""}"></span>'
            for n in [1, 2, 3, 4]
        ])
        sidebar_html += (
            f'<div class="side-section">'
            f'<div class="side-label"><span>Inferred Spending Pattern</span></div>'
            f'<div class="budget">{idots}'
            f'<span class="budget-label">{BUDGET_LABELS[inferred_level]} · {inferred_count} signal{"s" if inferred_count != 1 else ""}</span>'
            f'</div></div>'
        )

    # Insights donut
    tc = profile.get("tab_counts", {"eat": 0, "cook": 0, "drink": 0})
    total = tc["eat"] + tc["cook"] + tc["drink"]
    accepted = len(profile.get("accepted", []))
    rejected = len(profile.get("rejected", []))
    all_dec = accepted + rejected
    pct = round((accepted / all_dec) * 100) if all_dec > 0 else 0

    legend = (
        '<div class="legend">'
        f'<div class="legend-row"><span class="lbl"><span class="sw" style="background:var(--terracotta)"></span>Eat out</span><span class="num">{tc["eat"]}</span></div>'
        f'<div class="legend-row"><span class="lbl"><span class="sw" style="background:var(--sage)"></span>Cook</span><span class="num">{tc["cook"]}</span></div>'
        f'<div class="legend-row"><span class="lbl"><span class="sw" style="background:var(--gold)"></span>Cocktails</span><span class="num">{tc["drink"]}</span></div>'
        '</div>'
    )
    sidebar_html += (
        f'<div class="side-section">'
        f'<div class="insights">'
        f'<div class="insights-head"><span>Profile insights</span><span class="pct">{pct}% Accepted</span></div>'
        f'<div class="donut-wrap">{donut_svg(tc["eat"], tc["cook"], tc["drink"], total)}{legend}</div>'
        f'</div></div>'
    )

    # History
    history = profile.get("history", [])[-6:][::-1]
    if not history:
        history_html = '<div class="history-empty">No history yet</div>'
    else:
        history_html = '<div class="history">'
        for h in history:
            src = {"eat": "Eat Out", "cook": "Cook", "drink": "Cocktails"}.get(h.get("tab", "eat"), "EAT")
            history_html += (
                f'<div class="history-row {h["kind"]}">'
                f'<span class="dot"></span>'
                f'<span class="name">{html_module.escape(h["name"])}</span>'
                f'<span class="src">{src}</span>'
                f'</div>'
            )
        history_html += '</div>'
    sidebar_html += (
        f'<div class="side-section">'
        f'<div class="side-label"><span>Recent Activity</span><span class="count">{all_dec}</span></div>'
        f'{history_html}'
        f'</div>'
    )

    sidebar_top, sidebar_bottom = sidebar_html.split("<!--PREF_TAGS-->", 1)
    st.markdown(sidebar_top, unsafe_allow_html=True)
    render_preference_tags("You like", "liked_foods", "liked")
    render_preference_tags("Not for you", "disliked_foods", "disliked")
    st.markdown(sidebar_bottom, unsafe_allow_html=True)
    render_reset_button()


# ── Greeting + hint + tabs ────────────────────────────────────────────────────

def render_greeting():
    greeting = "Good morning, " if time_now == "morning" else "Good afternoon, " if time_now == "afternoon" else "Good evening, " if time_now == "evening" else "Up for a midnight snack?"
    friend = "Early bird" if time_now == "morning" else "Foodie" if time_now == "afternoon" else "Foodie" if time_now == "evening" else ""
    time_str = now.strftime("%-I:%M %p")

    st.markdown(
        f'<div class="greeting">'
        f'<div class="greeting-title">{greeting}<em>{friend}</em></div>'
        f'</div>'
        f'<p class="subline">A few warm suggestions, narrowed by what you\'ve liked before.</p>',
        unsafe_allow_html=True
    )


def render_hint():
    if st.session_state.hint_dismissed:
        return
    components.html(
        """
        <style>
            html, body {
                margin: 0;
                background: transparent;
                font-family: 'DM Sans', system-ui, sans-serif;
                color: #1A1A1A;
            }
            .hint {
                position: relative;
                display: flex;
                gap: 14px;
                align-items: flex-start;
                padding: 14px 16px 14px 18px;
                background: linear-gradient(180deg, #FFFBF5 0%, #FAF3E8 100%);
                border: 1px solid rgba(201, 162, 39, 0.25);
                border-radius: 14px;
                box-sizing: border-box;
            }
            .glyph {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                background: rgba(201,162,39,0.16);
                color: #8C7016;
                display: flex;
                align-items: center;
                justify-content: center;
                font-family: Georgia, serif;
                font-size: 16px;
                flex-shrink: 0;
            }
            .body { flex: 1; font-size: 13.5px; }
            .body b { font-weight: 600; }
            .body p { margin: 2px 0 0; color: #6B6B6B; font-size: 13px; }
            button {
                width: 22px;
                height: 22px;
                border: 0;
                border-radius: 6px;
                background: transparent;
                color: #9A968D;
                font-size: 14px;
                line-height: 1;
                cursor: pointer;
            }
            button:hover { background: rgba(0,0,0,0.05); color: #1A1A1A; }
        </style>
        <div class="hint">
            <span class="glyph">✦</span>
            <div class="body">
                <b>Your taste profile learns as you go.</b>
                <p>Accept or pass on recommendations — every decision updates your cuisine scores and personalises future results.</p>
            </div>
            <button type="button" aria-label="Dismiss">×</button>
        </div>
        <script>
            const hide = () => {
                const parentDoc = window.parent && window.parent.document;
                if (parentDoc) {
                    parentDoc.body.classList.add("food-ai-hint-gone");
                    if (!parentDoc.getElementById("food-ai-hint-gap-fix")) {
                        const style = parentDoc.createElement("style");
                        style.id = "food-ai-hint-gap-fix";
                        style.textContent = `
                            body.food-ai-hint-gone [data-testid="stTabs"] {
                                margin-top: -80px !important;
                            }
                        `;
                        parentDoc.head.appendChild(style);
                    }
                }
                if (window.frameElement) window.frameElement.style.display = "none";
                document.documentElement.style.display = "none";
            };
            document.querySelector("button").addEventListener("click", () => {
                hide();
            });
        </script>
        """,
        height=74,
        scrolling=False,
    )


def render_recent_strip():
    accepted = st.session_state.profile.get("accepted", [])
    rejected = st.session_state.profile.get("rejected", [])
    if not accepted and not rejected:
        return
    tagged = [(n, "acc") for n in accepted[-5:]] + [(n, "rej") for n in rejected[-5:]]
    chips = "".join([
        f'<span class="recent-chip {kind}"><span class="dot"></span>{html_module.escape(n)}</span>'
        for n, kind in tagged[::-1]
    ])
    st.markdown(
        f'<div class="recent">'
        f'<span class="recent-label">Recent Activity</span>'
        f'<div class="recent-track" id="recent-track">{chips}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    components.html(
        '<script>'
        '(function(){'
        '  function attach(){'
        '    var t=window.parent.document.getElementById("recent-track");'
        '    if(t){'
        '      t.addEventListener("wheel",function(e){'
        '        if(e.deltaY!==0){e.preventDefault();t.scrollLeft+=e.deltaY;}'
        '      },{passive:false});'
        '    } else { setTimeout(attach,100); }'
        '  }'
        '  attach();'
        '})();'
        '</script>',
        height=0,
        scrolling=False,
    )


def _haversine_mi(lat1, lon1, lat2, lon2) -> float:
    from math import radians, sin, cos, sqrt, atan2
    R = 3958.8
    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ── Cards ─────────────────────────────────────────────────────────────────────
def render_card(r, tab="eat", blurb=""):
    name = r.get("name") or r.get("title", "")
    address = r.get("address", "")
    rating = r.get("rating")
    review_count = r.get("total_tips", 0)
    price_str = PRICE_LABEL.get(r.get("price"), "")
    cats = r.get("categories", []) or ([r.get("category")] if r.get("category") else [])
    cat_str = " · ".join(cats[:2]) if cats else ""
    gradient = get_gradient_class(cats)
    photo_url = r.get("photo_url", "")
    photo_attribution = r.get("photo_attribution", "")

    accepted = name in st.session_state.profile.get("accepted", [])
    rejected = name in st.session_state.profile.get("rejected", [])
    card_class = "card combo"
    if accepted: card_class += " accepted"
    elif rejected: card_class += " rejected"

    dist_str = ""
    origin = st.session_state.get("eat_search_origin")
    if origin and r.get("lat") and r.get("lng"):
        d = _haversine_mi(origin[0], origin[1], r["lat"], r["lng"])
        dist_str = f"{d:.1f} mi"

    # Build card HTML
    img_label = price_str
    rating_html = ""
    if rating:
        rating_html = (
            f'<div class="card-rating">'
            f'<span class="stars">{stars_html(rating)}</span>'
            f'<span class="num">{rating}</span>'
            + (f'<span class="sep">·</span><span class="reviews">{review_count} reviews</span>' if review_count else "")
            + (f'<span class="sep">·</span><span class="reviews">{html_module.escape(address)}</span>' if address else "")
            + '</div>'
        )
    elif address:
        rating_html = f'<div class="card-rating"><span class="reviews">{html_module.escape(address)}</span></div>'

    blurb_html = f'<p class="card-blurb">{html_module.escape(blurb)}</p>' if blurb else ""
    photo_html = (
        f'<img src="{html_module.escape(photo_url, quote=True)}" alt="{html_module.escape(name, quote=True)}" loading="lazy" '
        f'onerror="this.remove(); this.parentElement.classList.remove(\'has-photo\');">'
        if photo_url else ""
    )

    PRICE_CHIPS = {1: "Cheap Eats", 4: "Upscale"}
    tag_list = []
    if cats:
        tag_list.append(cats[0])
    price_chip = PRICE_CHIPS.get(r.get("price"), "")
    if price_chip:
        tag_list.append(price_chip)
    for attr in r.get("attributes", []):
        tag_list.append(attr)
    for c in cats[1:2]:
        tag_list.append(c)
    tags_html = "".join([
        f'<span class="pill outline">{html_module.escape(t)}</span>'
        for t in tag_list
    ])

    # Actions
    open_now = r.get("open_now")
    if open_now is True:
        closes_at = r.get("closes_at", "")
        open_label = f"{closes_at}" if closes_at else "Open now"
        open_html = f'<span class="card-extra"><span class="dot"></span>{html_module.escape(open_label)}</span>'
    elif open_now is False:
        next_open = r.get("next_open", "")
        closed_label = f"{next_open}" if next_open else "Closed"
        open_html = f'<span class="card-extra closed"><span class="dot"></span>{html_module.escape(closed_label)}</span>'
    else:
        open_html = ""
    feedback_html = ""
    if accepted:
        feedback_html = '<span class="card-feedback-done acc">✓ Saved</span>'
    elif rejected:
        feedback_html = '<span class="card-feedback-done rej">✕ Passed</span>'

    actions = (
        f'<div class="card-actions">{open_html}{feedback_html}</div>'
        if open_html or feedback_html else ''
    )

    label_html = f'<span class="label">{html_module.escape(img_label)}</span>' if img_label else ""
    html_block = (
        f'<article class="{card_class}">'
        f'<div class="card-img ph-{gradient}{" has-photo" if photo_url else ""}">'
        f'{photo_html}'
        f'<span class="ph">{html_module.escape((cats[0] if cats else "").lower().replace(" ", " / "))}</span>'
        f'{label_html}'
        f'</div>'
        f'<div class="card-body">'
        f'<div class="card-meta-top"><span>{html_module.escape(cat_str)}</span>'
        + (f'<span class="sep">·</span><span>{html_module.escape(dist_str)}</span>' if dist_str else "")
        + f'</div>'
        f'<h3 class="card-title">{html_module.escape(name)}</h3>'
        f'{rating_html}'
        f'{blurb_html}'
        + (f'<div class="card-tags">{tags_html}</div>' if tags_html else "")
        + actions
        + '</div></article>'
    )
    card_id = stable_widget_key(tab, name, address, r.get("id") or r.get("place_id") or r.get("fsq_id"))
    rail_state = "accept" if accepted else "reject" if rejected else "neutral"
    rail_key = f"card_rail_{rail_state}_{card_id}"
    pass_key = f"card_pass_{card_id}"
    save_key = f"card_save_{card_id}"
    undo_state = "accept" if accepted else "reject"
    undo_key = f"card_undo_{undo_state}_{card_id}"
    card_col, action_col = st.columns([8, 1.05], gap=None)
    cuisines = [cats[0]] if cats else None
    with card_col:
        st.markdown(html_block, unsafe_allow_html=True)
    with action_col:
        with st.container(key=rail_key, height=280, border=False, gap=None):
            if accepted:
                st.button(
                    "Undo Save",
                    key=undo_key,
                    on_click=undo_card_feedback,
                    args=(name, True, cuisines, tab),
                    use_container_width=True,
                )
            elif rejected:
                st.button(
                    "Undo Pass",
                    key=undo_key,
                    on_click=undo_card_feedback,
                    args=(name, False, cuisines, tab),
                    use_container_width=True,
                )
            else:
                price_val = r.get("price")
                st.button(
                    "Pass",
                    key=pass_key,
                    on_click=apply_card_feedback,
                    args=(name, False, cuisines, tab, price_val),
                    use_container_width=True,
                )
                st.button(
                    "Save",
                    key=save_key,
                    on_click=apply_card_feedback,
                    args=(name, True, cuisines, tab, price_val),
                    use_container_width=True,
                )


# ── Skeleton ──────────────────────────────────────────────────────────────────
def render_skeletons(n=3):
    blocks = "".join([
        '<div class="skeleton">'
        '<div class="img"></div>'
        '<div class="body">'
        '<div class="bar short"></div>'
        '<div class="bar title"></div>'
        '<div class="bar"></div>'
        '<div class="bar medium"></div>'
        '<div class="bar short"></div>'
        '</div></div>'
        for _ in range(n)
    ])
    st.markdown(f'<div class="cards">{blocks}</div>', unsafe_allow_html=True)


def curated_to_cards(rows):
    cards = []
    for r in rows[:5]:
        category = r.get("category", "")
        popular_food = " ".join(str(r.get("popular_food", "") or "").split())
        if popular_food:
            blurb = f"Curated match for {popular_food}."
        elif category:
            blurb = f"Curated {category} match from the local restaurant dataset."
        else:
            blurb = "Curated match from the local restaurant dataset."
        cards.append({
            "name": r.get("title", ""),
            "address": "",
            "categories": [category] if category else [],
            "price": None,
            "rating": None,
            "open_now": None,
            "total_tips": 0,
            "blurb": blurb,
            "source": "static_rag",
            "popular_food": popular_food,
            "retrieval_score": r.get("retrieval_score"),
            "review_snippets": r.get("review_snippets", ""),
            "match_source": r.get("match_source", ""),
            "match_note": r.get("match_note", ""),
        })
    return cards


def merge_curated_results(priority_rows, fallback_rows, top_k=5):
    merged = []
    seen = set()
    for row in list(priority_rows or []) + list(fallback_rows or []):
        title = str(row.get("title") or row.get("name") or "").strip()
        key = title.lower()
        if not key or key in seen:
            continue
        merged.append(row)
        seen.add(key)
        if len(merged) >= top_k:
            break
    return merged


# ── Empty state ───────────────────────────────────────────────────────────────
CHIP_TARGETS = {
    "eat": "hand-rolled pasta, candlelit, walking distance…",
    "cook": "something fast, something cozy, something to impress…",
    "drink": "rainy night, pre-dinner, after a long week…",
}

CHIP_SUBMIT_LABELS = {
    "eat": "Find restaurants",
    "cook": "Suggest recipes",
    "drink": "Suggest cocktails",
}


def suggest_chips_html(tab, centered=False):
    chips = "".join([
        f'<button type="button" class="suggest-chip" data-chip="{html_module.escape(c, quote=True)}">'
        f'{html_module.escape(c)}</button>'
        for c in QUICK_STARTS[tab]
    ])
    justify = "center" if centered else "flex-start"
    margin_top = "0" if centered else "12px"
    target = html_module.escape(CHIP_TARGETS[tab], quote=True)
    submit_label = html_module.escape(CHIP_SUBMIT_LABELS[tab], quote=True)
    return f"""
    <style>
        html, body {{
            margin: 0;
            background: transparent;
            font-family: 'DM Sans', system-ui, sans-serif;
        }}
        .suggest-row {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            justify-content: {justify};
            margin: {margin_top} 0 0;
        }}
        .suggest-chip {{
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            background: transparent;
            border: 1px dashed rgba(26,26,26,0.10);
            border-radius: 999px;
            padding: 4px 11px;
            color: #6B6B6B;
            font: 400 12.5px/1.2 'DM Sans', system-ui, sans-serif;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.14s ease;
        }}
        .suggest-chip:hover {{
            border-color: #C96A3A;
            border-style: solid;
            color: #B25A2C;
        }}
    </style>
    <div class="suggest-row" data-target-placeholder="{target}" data-submit-label="{submit_label}">{chips}</div>
    <script>
        const root = document.currentScript.previousElementSibling;
        const targetPlaceholder = root.dataset.targetPlaceholder;
        const submitLabel = root.dataset.submitLabel;
        const setNativeValue = (el, value) => {{
            const win = el.ownerDocument.defaultView;
            const proto = el.tagName === "TEXTAREA" ? win.HTMLTextAreaElement.prototype : win.HTMLInputElement.prototype;
            const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
            setter.call(el, value);
            el.dispatchEvent(new win.InputEvent("input", {{ bubbles: true, inputType: "insertText", data: value }}));
            el.dispatchEvent(new win.Event("change", {{ bubbles: true }}));
            el.focus();
        }};
        root.addEventListener("click", (event) => {{
            const chip = event.target.closest("[data-chip]");
            if (!chip) return;
            const doc = window.parent.document;
            const candidates = Array.from(doc.querySelectorAll("input, textarea"));
            const target = candidates.find((el) => el.placeholder === targetPlaceholder)
                || candidates.find((el) => el.placeholder && el.placeholder.startsWith(targetPlaceholder.slice(0, 18)));
            if (!target) return;
            setNativeValue(target, chip.dataset.chip);
            window.setTimeout(() => {{
                const buttons = Array.from(doc.querySelectorAll("button"));
                const submit = buttons.find((button) => button.innerText.includes(submitLabel));
                if (submit) submit.click();
            }}, 350);
        }});
    </script>
    """


def render_suggest_chips(tab, centered=False):
    components.html(suggest_chips_html(tab, centered=centered), height=46 if not centered else 34, scrolling=False)


def render_empty(tab):
    copy = EMPTY_COPY[tab]
    empty_html = f"""
    <style>
        html, body {{
            margin: 0;
            padding-bottom: 4px;
            background: transparent;
            color: #1A1A1A;
            font-family: 'DM Sans', system-ui, sans-serif;
        }}
        .empty {{
            background: #FFFFFF;
            border: 1px dashed rgba(26,26,26,0.10);
            border-radius: 20px;
            padding: 48px 32px 64px;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
            box-sizing: border-box;
        }}
        .glyph {{
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: rgba(201,106,58,0.13);
            color: #C96A3A;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: Georgia, serif;
            font-size: 28px;
            margin-bottom: 4px;
        }}
        h3 {{
            font-family: Georgia, serif;
            font-weight: 400;
            font-size: 22px;
            letter-spacing: -0.01em;
            margin: 0;
        }}
        p {{
            color: #6B6B6B;
            margin: 0;
            max-width: 38ch;
            font-size: 14px;
        }}
        .chip-wrap {{ margin-top: 6px; }}
    </style>
    <div class="empty">
        <div class="glyph">{html_module.escape(copy["glyph"])}</div>
        <h3>{html_module.escape(copy["title"])}</h3>
        <p>{html_module.escape(copy["body"])}</p>
        <div class="chip-wrap">{suggest_chips_html(tab, centered=True)}</div>
    </div>
    """
    components.html(empty_html, height=338, scrolling=False)


# ── Tabs ──────────────────────────────────────────────────────────────────────
def render_eat_tab(client, df):
    prefill = st.session_state.eat_prefill or ""
    st.session_state.eat_prefill = ""

    with compatible_form(key="eat_form", enter_to_submit=True, border=False):
        col1, col2 = st.columns([3, 1.2])
        with col1:
            st.markdown('<div class="field-label">What are you craving</div>', unsafe_allow_html=True)
            query = st.text_input("craving", value=prefill, placeholder="hand-rolled pasta, candlelit, walking distance…", label_visibility="collapsed")
            render_suggest_chips("eat")
        with col2:
            st.markdown('<div class="field-label">Zip Code</div>', unsafe_allow_html=True)
            zipcode = st.text_input("zip", placeholder="Searching all NYC", label_visibility="collapsed")
            run_search = st.form_submit_button("Find restaurants  →", type="primary", use_container_width=True)

    render_recent_strip()

    if run_search and query:
        st.session_state.active_tab = "eat"
        refresh_preference_tags(st.session_state.profile)
        save_profile(st.session_state.profile)

        st.markdown(f'<div class="results-head"><h2>{TAB_HEADING["eat"]}</h2><span class="count">{THINKING_MSG["eat"]}</span></div>', unsafe_allow_html=True)
        
        # Only show skeletons if there's no existing output
        skel_placeholder = None
        if not st.session_state.eat_llm_response:
            skel_placeholder = st.empty()
            with skel_placeholder.container():
                render_skeletons(5)

        from concurrent.futures import ThreadPoolExecutor
        from src.places import search_restaurants, geocode_location

        search_notes = []
        borough = zipcode if zipcode else "New York, NY"

        def _run_rag():
            _, results = rag_recommend(client, query, st.session_state.profile, df, top_k=5)
            return results

        def _run_places():
            origin = geocode_location(zipcode + " New York") if zipcode else (40.7128, -74.0060)
            restaurants = search_restaurants(query=query, borough=borough, limit=8)
            return origin, restaurants

        retrieved = []
        fsq_restaurants = []
        origin = (40.7128, -74.0060)

        with ThreadPoolExecutor(max_workers=2) as pool:
            rag_future = pool.submit(_run_rag)
            places_future = pool.submit(_run_places)

            try:
                retrieved = rag_future.result()
            except Exception as e:
                search_notes.append(f"Curated retrieval skipped: {e}")
            try:
                origin, fsq_restaurants = places_future.result()
            except Exception as e:
                search_notes.append(f"Live Places search skipped: {e}")

        try:
            exact_static_matches = find_static_cuisine_matches(query, get_full_restaurant_df(), top_k=5)
            if exact_static_matches:
                retrieved = merge_curated_results(exact_static_matches, retrieved, top_k=5)
        except Exception as e:
            search_notes.append(f"Exact cuisine fallback skipped: {e}")

        st.session_state.eat_results = retrieved
        st.session_state.eat_search_origin = origin
        st.session_state.eat_fsq_results = fsq_restaurants

        selected = []
        response = ""
        live_results = st.session_state.eat_fsq_results or []
        if live_results:
            try:
                from src.recommend import combined_recommend
                response, selected = combined_recommend(
                    client, query, st.session_state.profile,
                    retrieved, live_results
                )
            except Exception as e:
                search_notes.append(f"AI ranking skipped: {e}")
                selected = live_results[:5]
        if not selected and live_results:
            selected = live_results[:5]
        if not selected and retrieved:
            selected = curated_to_cards(retrieved)
        if selected and not response:
            response = f"Showing direct matches for {query}."
        if not selected:
            response = f"No matches came back for {query}. Try a more specific craving or location."
        if search_notes:
            st.session_state.eat_search_notes = search_notes
        else:
            st.session_state.eat_search_notes = []
        st.session_state.eat_llm_response = response
        st.session_state.eat_fsq_results = selected

        if skel_placeholder:
            skel_placeholder.empty()
        st.rerun()

    if st.session_state.eat_llm_response:
        results = st.session_state.eat_fsq_results or []
        st.markdown(
            f'<div class="results-head">'
            f'<h2>{TAB_HEADING["eat"]}</h2>'
            f'<span class="count">{len(results)} {"pick" if len(results)==1 else "picks"} for you</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        for r in results:
            render_card(r, tab="eat", blurb=r.get("blurb", ""))
    else:
        render_empty("eat")


def render_cook_tab(client):
    prefill = st.session_state.cook_prefill or ""
    st.session_state.cook_prefill = ""

    with compatible_form(key="cook_form", enter_to_submit=True, border=False):
        st.markdown('<div class="field-label">Tonight you want</div>', unsafe_allow_html=True)
        craving = st.text_input("craving", value=prefill, placeholder="something fast, something cozy, something to impress…", label_visibility="collapsed")
        st.markdown('<div class="field-label" style="margin-top:10px">In the pantry</div>', unsafe_allow_html=True)
        pantry_input = st.text_area(
            "pantry",
            value=", ".join(st.session_state.profile.get("pantry", [])),
            placeholder="just dump it all here",
            label_visibility="collapsed",
            height=88,
        )
        
        col1, col2 = st.columns([3, 1.2])
        with col1:
            render_suggest_chips("cook")
        with col2:
            run_cook = st.form_submit_button("Suggest recipes  →", type="primary", use_container_width=True)

    render_recent_strip()

    if run_cook and craving:
        st.session_state.active_tab = "cook"
        pantry = [p.strip() for p in pantry_input.split(",") if p.strip()]
        st.session_state.profile["pantry"] = pantry
        refresh_preference_tags(st.session_state.profile)
        save_profile(st.session_state.profile)

        st.markdown(f'<div class="results-head"><h2>{TAB_HEADING["cook"]}</h2><span class="count">{THINKING_MSG["cook"]}</span></div>', unsafe_allow_html=True)
        
        # Only show skeletons if there's no existing output
        skel_placeholder = None
        if not st.session_state.cook_response:
            skel_placeholder = st.empty()
            with skel_placeholder.container():
                render_skeletons(3)

        from src.recommend import recommend_recipe
        
        response = recommend_recipe(craving, st.session_state.profile, client=client)
        st.session_state.cook_response = response
        st.session_state.cook_last_craving = craving
        st.session_state.cook_remix_active = False

        if skel_placeholder:
            skel_placeholder.empty()
        st.rerun()

    if st.session_state.cook_remix_pending:
        st.session_state.active_tab = "cook"
        if not st.session_state.cook_response:
            render_skeletons(3)
        from src.recommend import recommend_recipe
        combined = st.session_state.cook_remix_pending
        st.session_state.cook_response = recommend_recipe(
            combined, st.session_state.profile, client=client,
            previous_response=st.session_state.cook_remix_previous,
        )
        st.session_state.cook_last_craving = combined
        st.session_state.cook_remix_pending = None
        st.session_state.cook_remix_previous = None
        st.rerun()

    if st.session_state.cook_response:
        pantry = st.session_state.profile.get("pantry", [])
        _is_empty_msg = "pantry is empty" in st.session_state.cook_response.lower()

        n_recipes = len(_split_cook_recipes(st.session_state.cook_response))
        recipe_label = f"{n_recipes} recipes" if n_recipes > 1 else "Your recipe"
        st.markdown(
            f'<div class="results-head">'
            f'<h2>{TAB_HEADING["cook"]}</h2>'
            f'<span class="count">{recipe_label}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        if _is_empty_msg:
            with st.container():
                st.markdown(st.session_state.cook_response)
        else:
            for idx, (recipe_name, recipe_why, recipe_block) in enumerate(_split_cook_recipes(st.session_state.cook_response)):
                key_base = stable_widget_key("cook", recipe_name, idx)
                accepted = recipe_name in st.session_state.profile.get("accepted", [])
                rejected = recipe_name in st.session_state.profile.get("rejected", [])

                with st.container(key=f"cook_recipe_card_{key_base}"):
                    status_class = "saved" if accepted else "passed" if rejected else ""
                    status_label = "Saved" if accepted else "Passed" if rejected else ""
                    status_html = (
                        f' <span class="cook-card-status {status_class}">{html_module.escape(status_label)}</span>'
                        if status_label else ""
                    )
                    why_clean = re.sub(r'\*+', '', recipe_why).strip()
                    st.markdown(
                        f'<h3 class="cook-card-title">{html_module.escape(recipe_name)}{status_html}</h3>',
                        unsafe_allow_html=True,
                    )
                    if why_clean:
                        st.markdown(f'<p class="cook-card-why">{html_module.escape(why_clean)}</p>', unsafe_allow_html=True)
                    with st.expander("Full recipe"):
                        st.markdown(_format_cook_recipe_for_expander(recipe_block), unsafe_allow_html=True)

                    if accepted:
                        _, _, c_save = st.columns([1, 1, 1])
                        with c_save:
                            if st.button("Undo Save", key=f"cook_option_undo_save_{key_base}", use_container_width=True):
                                st.session_state.active_tab = "cook"
                                undo_card_feedback(recipe_name, True, tab="cook")
                                st.rerun()
                    elif rejected:
                        c_pass, _, _ = st.columns([1, 1, 1])
                        with c_pass:
                            if st.button("Undo Pass", key=f"cook_option_undo_pass_{key_base}", use_container_width=True):
                                st.session_state.active_tab = "cook"
                                undo_card_feedback(recipe_name, False, tab="cook")
                                st.rerun()
                    else:
                        c_pass, c_remix, c_save = st.columns([1, 1, 1])
                        with c_pass:
                            if st.button("Pass", key=f"cook_option_pass_{key_base}", use_container_width=True):
                                st.session_state.active_tab = "cook"
                                apply_card_feedback(recipe_name, False, tab="cook")
                                st.rerun()
                        with c_remix:
                            is_remixing = (
                                st.session_state.cook_remix_active
                                and st.session_state.cook_remix_card == recipe_name
                            )
                            if st.button("Remix ↩" if is_remixing else "Remix", key=f"cook_option_remix_{key_base}", use_container_width=True):
                                st.session_state.active_tab = "cook"
                                if is_remixing:
                                    st.session_state.cook_remix_active = False
                                    st.session_state.cook_remix_card = None
                                else:
                                    st.session_state.cook_remix_active = True
                                    st.session_state.cook_remix_card = recipe_name
                                st.rerun()
                        with c_save:
                            if st.button("Save", key=f"cook_option_save_{key_base}", use_container_width=True):
                                st.session_state.active_tab = "cook"
                                apply_card_feedback(recipe_name, True, tab="cook")
                                st.rerun()

                    render_generation_trace("cook", recipe_block, pantry, st.session_state.cook_last_craving)

                if (
                    st.session_state.cook_remix_active
                    and st.session_state.cook_remix_card == recipe_name
                ):
                    with compatible_form(key=f"cook_remix_form_{key_base}", enter_to_submit=True, border=False):
                        col1, col2 = st.columns([3, 1.2])
                        with col1:
                            remix_input = st.text_input("Add context", placeholder="make it spicier, fewer steps, vegetarian…", label_visibility="collapsed")
                        with col2:
                            if st.form_submit_button("Remix →", type="primary", use_container_width=True) and remix_input:
                                st.session_state.active_tab = "cook"
                                st.session_state.cook_remix_pending = f"{st.session_state.cook_last_craving}. Remix '{recipe_name}': {remix_input}"
                                st.session_state.cook_remix_previous = recipe_block
                                st.session_state.cook_remix_active = False
                                st.session_state.cook_remix_card = None
                                st.rerun()

    else:
        render_empty("cook")


def render_cocktail_tab(client):
    prefill = st.session_state.drink_prefill or ""
    st.session_state.drink_prefill = ""

    with compatible_form(key="cocktail_form", enter_to_submit=True, border=False):
        st.markdown('<div class="field-label">The vibe</div>', unsafe_allow_html=True)
        vibe = st.text_input("vibe", value=prefill, placeholder="rainy night, pre-dinner, after a long week…", label_visibility="collapsed")
        st.markdown('<div class="field-label" style="margin-top:10px">Bar inventory</div>', unsafe_allow_html=True)
        bar_input = st.text_area(
            "bar",
            value=", ".join(st.session_state.profile.get("bar_inventory", [])),
            placeholder="tell me what you've got, or try 'the basics'",
            label_visibility="collapsed",
            height=88,
        )
        col1, col2 = st.columns([3, 1.2])
        
        with col1:
            render_suggest_chips("drink")
        with col2:
            run_cocktail = st.form_submit_button("Suggest cocktails  →", type="primary", use_container_width=True)

    render_recent_strip()

    if run_cocktail and vibe:
        st.session_state.active_tab = "drink"
        bar = [b.strip() for b in bar_input.split(",") if b.strip()]
        st.session_state.profile["bar_inventory"] = bar
        refresh_preference_tags(st.session_state.profile)
        save_profile(st.session_state.profile)

        st.markdown(f'<div class="results-head"><h2>{TAB_HEADING["drink"]}</h2><span class="count">{THINKING_MSG["drink"]}</span></div>', unsafe_allow_html=True)
        
        # Only show skeletons if there's no existing output
        skel_placeholder = None
        if not st.session_state.cocktail_response:
            skel_placeholder = st.empty()
            with skel_placeholder.container():
                render_skeletons(3)

        from src.recommend import recommend_cocktail

        response, grounding = recommend_cocktail(vibe, st.session_state.profile)
        st.session_state.cocktail_response = response
        st.session_state.drink_grounding = grounding
        st.session_state.drink_last_vibe = vibe
        st.session_state.drink_remix_active = False
        st.session_state.drink_remix_card = None

        if skel_placeholder:
            skel_placeholder.empty()
        st.rerun()

    if st.session_state.drink_remix_pending:
        st.session_state.active_tab = "drink"
        if not st.session_state.cocktail_response:
            render_skeletons(3)
        from src.recommend import recommend_cocktail
        combined = st.session_state.drink_remix_pending
        response, grounding = recommend_cocktail(combined, st.session_state.profile, previous_response=st.session_state.cocktail_response)
        st.session_state.cocktail_response = response
        st.session_state.drink_grounding = grounding
        st.session_state.drink_last_vibe = combined
        st.session_state.drink_remix_pending = None
        st.rerun()

    if st.session_state.cocktail_response:
        bar = st.session_state.profile.get("bar_inventory", [])
        _is_empty_bar = "bar is empty" in st.session_state.cocktail_response.lower()

        n_drinks = len(_split_cocktail_recipes(st.session_state.cocktail_response))
        drink_label = f"{n_drinks} cocktails" if n_drinks > 1 else "Your cocktail"
        st.markdown(
            f'<div class="results-head">'
            f'<h2>{TAB_HEADING["drink"]}</h2>'
            f'<span class="count">{drink_label}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        if _is_empty_bar:
            with st.container():
                st.markdown(st.session_state.cocktail_response)
        else:
            grounding = st.session_state.get("drink_grounding", [])
            thumb_candidates = [c["thumbnail"] for c in grounding if c.get("thumbnail")]
            thumb_map = {
                normalized_lookup_key(c.get("name", "")): c["thumbnail"]
                for c in grounding
                if c.get("thumbnail")
            }

            for idx, (cocktail_name, cocktail_why, cocktail_block) in enumerate(_split_cocktail_recipes(st.session_state.cocktail_response)):
                key_base = stable_widget_key("drink", cocktail_name, idx)
                accepted = cocktail_name in st.session_state.profile.get("accepted", [])
                rejected = cocktail_name in st.session_state.profile.get("rejected", [])
                thumb_url = thumb_map.get(normalized_lookup_key(cocktail_name), "")
                if not thumb_url and thumb_candidates:
                    thumb_url = thumb_candidates[idx % len(thumb_candidates)]
                thumb_src = image_url_to_data_uri(thumb_url) or thumb_url

                with st.container(key=f"drink_recipe_card_{key_base}"):
                    why_clean = re.sub(r'\*+', '', cocktail_why).strip()
                    status_class = "saved" if accepted else "passed" if rejected else ""
                    status_label = "Saved" if accepted else "Passed" if rejected else ""
                    thumb_html = (
                        f'<img src="{html_module.escape(thumb_src, quote=True)}" alt="{html_module.escape(cocktail_name, quote=True)}" loading="lazy" referrerpolicy="no-referrer" '
                        f'onerror="this.remove(); this.parentElement.classList.remove(\'has-photo\');">'
                        if thumb_src else ""
                    )
                    why_html = (
                        f'<p class="drink-card-why">{html_module.escape(why_clean)}</p>'
                        if why_clean else ""
                    )
                    status_html = (
                        f'<span class="drink-card-status {status_class}">{html_module.escape(status_label)}</span>'
                        if status_label else ""
                    )

                    st.markdown(
                        f'<div class="drink-card-layout">'
                        f'<div class="drink-card-image{" has-photo" if thumb_url else ""}">'
                        f'{thumb_html}'
                        f'<span class="ph">mixed / drink</span>'
                        f'</div>'
                        f'<div class="drink-card-body">'
                        f'<div class="drink-card-title-row">'
                        f'<h3 class="drink-card-title">{html_module.escape(cocktail_name)}</h3>'
                        f'{status_html}'
                        f'</div>'
                        f'{why_html}'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    with st.expander("Full recipe"):
                        st.markdown(_format_cocktail_recipe_for_expander(cocktail_block), unsafe_allow_html=True)

                    if accepted:
                        _, _, c_save = st.columns([1, 1, 1])
                        with c_save:
                            if st.button("Undo Save", key=f"drink_option_undo_save_{key_base}", use_container_width=True):
                                st.session_state.active_tab = "drink"
                                undo_card_feedback(cocktail_name, True, tab="drink")
                                st.rerun()
                    elif rejected:
                        c_pass, _, _ = st.columns([1, 1, 1])
                        with c_pass:
                            if st.button("Undo Pass", key=f"drink_option_undo_pass_{key_base}", use_container_width=True):
                                st.session_state.active_tab = "drink"
                                undo_card_feedback(cocktail_name, False, tab="drink")
                                st.rerun()
                    else:
                        c_pass, c_remix, c_save = st.columns([1, 1, 1])
                        with c_pass:
                            if st.button("Pass", key=f"drink_option_pass_{key_base}", use_container_width=True):
                                st.session_state.active_tab = "drink"
                                apply_card_feedback(cocktail_name, False, tab="drink")
                                st.rerun()
                        with c_remix:
                            is_remixing = (
                                st.session_state.drink_remix_active
                                and st.session_state.drink_remix_card == cocktail_name
                            )
                            if st.button("Remix ↩" if is_remixing else "Remix", key=f"drink_option_remix_{key_base}", use_container_width=True):
                                st.session_state.active_tab = "drink"
                                if is_remixing:
                                    st.session_state.drink_remix_active = False
                                    st.session_state.drink_remix_card = None
                                else:
                                    st.session_state.drink_remix_active = True
                                    st.session_state.drink_remix_card = cocktail_name
                                st.rerun()
                        with c_save:
                            if st.button("Save", key=f"drink_option_save_{key_base}", use_container_width=True):
                                st.session_state.active_tab = "drink"
                                apply_card_feedback(cocktail_name, True, tab="drink")
                                st.rerun()

                    render_generation_trace("drink", cocktail_block, bar, st.session_state.drink_last_vibe)

                if (
                    st.session_state.drink_remix_active
                    and st.session_state.drink_remix_card == cocktail_name
                ):
                    with compatible_form(key=f"drink_remix_form_{key_base}", enter_to_submit=True, border=False):
                        col1, col2 = st.columns([3, 1.2])
                        with col1:
                            remix_input = st.text_input("Add context", placeholder="make it sweeter, no citrus, more spirit-forward…", label_visibility="collapsed")
                        with col2:
                            if st.form_submit_button("Remix  →", type="primary", use_container_width=True) and remix_input:
                                st.session_state.active_tab = "drink"
                                st.session_state.drink_remix_pending = f"{st.session_state.drink_last_vibe}. {remix_input}"
                                st.session_state.cocktail_response = None
                                st.session_state.drink_remix_active = False
                                st.session_state.drink_remix_card = None
                                st.rerun()
    else:
        render_empty("drink")


def restore_active_tab():
    tab_index = {"eat": 0, "cook": 1, "drink": 2}.get(st.session_state.get("active_tab", "eat"), 0)
    if tab_index == 0:
        return

    components.html(
        f"""
        <script>
        (function() {{
            const target = {tab_index};
            function activateTab() {{
                const tabs = Array.from(window.parent.document.querySelectorAll('[data-baseweb="tab"]'));
                if (tabs.length > target) {{
                    const tab = tabs[target];
                    if (tab.getAttribute("aria-selected") !== "true") {{
                        tab.click();
                    }}
                    return;
                }}
                window.setTimeout(activateTab, 50);
            }}
            window.setTimeout(activateTab, 0);
        }})();
        </script>
        """,
        height=0,
        scrolling=False,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_session()
    handle_query_params()
    with st.sidebar:
        render_sidebar()

    render_greeting()
    render_hint()

    client = get_client()
    df = get_df()

    tab_eat, tab_cook, tab_drink = st.tabs(["🍽️  Eat Out", "🍳  Cook", "🍸  Cocktails"])
    restore_active_tab()

    with tab_eat:
        render_eat_tab(client, df)

    with tab_cook:
        render_cook_tab(client)

    with tab_drink:
        render_cocktail_tab(client)


if __name__ == "__main__":
    main()
