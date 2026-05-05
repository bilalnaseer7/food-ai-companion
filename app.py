import os
import math
import hashlib
import html as html_module
from urllib.parse import quote
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

from src.data_loader import load_reviews
from src.recommend import rag_recommend, map_recommend
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

import datetime
now = datetime.datetime.now()
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

BUDGET_LABELS = {1: "Cheap", 2: "Smart", 3: "Treat", 4: "Splurge"}
BUDGET_MAP = {"budget": 1, "moderate": 2, "premium": 3, "premium+": 4}
BUDGET_REVERSE = {1: "budget", 2: "moderate", 3: "premium", 4: "premium+"}
PROFILE_PATH = "data/taste_profile.json"
PROFILE_RESET_MARKER = "data/.profile_reset"

QUICK_STARTS = {
    "eat":  ["something cozy", "date night", "wood-fired anything", "walking distance"],
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
        "body": "Drop your craving and what's actually in the fridge. We'll work backward from there.",
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


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@st.cache_data
def get_df():
    return load_reviews(path="data/restaurants.csv", max_rows=3000)


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
    --line: rgba(26,26,26,0.06);
    --line-2: rgba(26,26,26,0.10);
    --terracotta: #C96A3A;
    --terracotta-2: #B25A2C;
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
    padding: 36px 48px 80px !important;
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

/* ── Sidebar (target Streamlit's sidebar to match design's 320px column) ── */
[data-testid="stSidebar"] {
    background: var(--bg) !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"][aria-expanded="true"] {
    width: 350px !important;
    min-width: 350px !important;
    max-width: 350px !important;
    margin-right: -175px !important;
    position: relative !important;
    transition: all 0.3s ease !important;
}
[data-testid="stSidebar"] > div { padding: 24px 20px !important; }
[data-testid="stSidebar"] * { color: var(--ink) !important; }

/* ── Brand ── */
.brand { display: flex; align-items: center; gap: 10px; margin-top: -12px; padding-bottom: 18px; }
.brand-mark {
    width: 8px; height: 8px; border-radius: 50%;
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
    transition: all 0.14s ease;
}
.pill:hover { transform: translateY(-1px); }
.pill.terracotta { background: var(--tag-terracotta); color: var(--terracotta-2); }
.pill.sage { background: var(--tag-sage); color: var(--sage-2); }
.pill.outline { background: transparent; border-color: var(--line-2); color: var(--ink-2); }
.pill .pill-bar { width: 4px; height: 4px; border-radius: 50%; background: currentColor; opacity: 0.5; }
.pill .x { font-size: 11px; opacity: 0.5; margin-left: 2px; }
.pill .x:hover { opacity: 1; }

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
    gap: 24px; margin: 10px 0 8px;
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
.stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div {
    background: var(--bg) !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-input) !important;
    overflow: hidden !important;
}
.stTextInput input, .stTextArea textarea {
    color: var(--ink) !important;
    background: transparent !important;
    font-family: var(--sans) !important;
    font-size: 14.5px !important;
    border: none !important;
    border-radius: inherit !important;
    padding: 12px 14px !important;
    box-shadow: none !important;
    outline: none !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--ink-3) !important; }
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
[data-testid="stFormSubmitButton"] button[kind="primary"] {
    background: var(--terracotta) !important; color: #fff !important;
    border: none !important; border-radius: var(--radius) !important;
    padding: 12px 22px !important;
    font-family: var(--sans) !important; font-weight: 500 !important;
    font-size: 14px !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.2) inset, 0 4px 14px rgba(201,106,58,0.30) !important;
    transition: all 0.12s ease !important;
    width: 100% !important;
}
[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
    background: var(--terracotta-2) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(201,106,58,0.36) !important;
}

.suggest-chip { color: var(--ink-2) !important; }

/* ── Recently saved strip ── */
.recent {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px; background: var(--bg-deep);
    border-radius: var(--radius); margin-bottom: 12px;
    font-size: 13px; color: var(--ink-2);
    margin-bottom: 20px;
}
.recent-label {
    font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--ink-3); flex-shrink: 0;
}
.recent-track { display: flex; gap: 6px; flex-wrap: nowrap; overflow: hidden; flex: 1; }
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
.llm-response {
    background: linear-gradient(180deg, #FFFBF5 0%, #FAF3E8 100%);
    border: 1px solid rgba(201,162,39,0.2);
    border-radius: var(--radius); padding: 18px 22px;
    margin: 8px 0 20px;
    font-size: 14.5px; line-height: 1.65; color: var(--ink);
    white-space: pre-wrap;
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


def reset_taste_profile():
    if os.path.exists(PROFILE_PATH):
        os.remove(PROFILE_PATH)
    os.makedirs(os.path.dirname(PROFILE_RESET_MARKER), exist_ok=True)
    with open(PROFILE_RESET_MARKER, "w") as f:
        f.write("1")
    profile = load_profile()
    save_profile(profile)
    st.session_state.profile = profile
    st.session_state.sample_profile_disabled = True
    st.session_state.history = []
    st.session_state.tab_counts = {"eat": 0, "cook": 0, "drink": 0}
    st.session_state.eat_results = None
    st.session_state.eat_fsq_results = None
    st.session_state.eat_llm_response = None
    st.session_state.cook_response = None
    st.session_state.cocktail_response = None


def apply_card_feedback(name, accepted, cuisines=None, tab="eat"):
    opposite_bucket = "rejected" if accepted else "accepted"
    if name in st.session_state.profile.get(opposite_bucket, []):
        st.session_state.profile[opposite_bucket].remove(name)
    st.session_state.profile = update_profile(
        st.session_state.profile,
        restaurant_name=name,
        accepted=accepted,
        cuisines=cuisines or None,
    )
    save_profile(st.session_state.profile)
    st.session_state.history.append({"name": name, "kind": "acc" if accepted else "rej", "tab": tab})
    st.session_state.tab_counts[tab] = st.session_state.tab_counts.get(tab, 0) + 1


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
    st.session_state.history = [
        h for h in st.session_state.history
        if not (h.get("name") == name and h.get("kind") == kind and h.get("tab") == tab)
    ]
    st.session_state.tab_counts[tab] = max(0, st.session_state.tab_counts.get(tab, 0) - 1)
    save_profile(st.session_state.profile)


def render_reset_button():
    st.button(
        "Reset taste profile  ↺",
        key="reset_taste_profile",
        on_click=reset_taste_profile,
        use_container_width=True,
    )


# ── Session state ─────────────────────────────────────────────────────────────
def init_session():
    if "profile" not in st.session_state:
        profile = load_profile()
        sample_profile_disabled = os.path.exists(PROFILE_RESET_MARKER)
        if not sample_profile_disabled and not profile["preferred_cuisines"] and not profile["cuisine_scores"]:
            profile.update({
                "preferred_cuisines": ["Italian", "Pizza"],
                "liked_foods": ["pasta", "pizza"],
                "disliked_foods": ["seafood"],
                "budget": "moderate",
                "online_order": "Yes",
                "occasion": "casual dinner",
                "cuisine_scores": {"Italian": 0.8, "Japanese": 0.6, "Mediterranean": 0.45, "American": 0.3},
                "food_scores": {"pasta": 0.7, "pizza": 0.6, "seafood": -0.5},
            })
            save_profile(profile)
        st.session_state.profile = profile
        st.session_state.sample_profile_disabled = sample_profile_disabled

    defaults = {
        "eat_results": None, "eat_fsq_results": None, "eat_llm_response": None,
        "cook_response": None, "cocktail_response": None,
        "feedback_given": set(),
        "hint_dismissed": False,
        "tab_counts": {"eat": 0, "cook": 0, "drink": 0},
        "history": [],
        "eat_prefill": "", "cook_prefill": "", "drink_prefill": "",
        "active_tab": "eat",
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

        if action in ("accept", "reject") and name:
            accepted = action == "accept"
            cuisines = [cuisine] if cuisine else None
            st.session_state.profile = update_profile(
                st.session_state.profile, restaurant_name=name,
                accepted=accepted, cuisines=cuisines,
            )
            save_profile(st.session_state.profile)
            st.session_state.history.append({"name": name, "kind": "acc" if accepted else "rej", "tab": tab})
            st.session_state.tab_counts[tab] = st.session_state.tab_counts.get(tab, 0) + 1

        elif action == "rm_like":
            food = qp.get("food", "")
            if food in st.session_state.profile.get("liked_foods", []):
                st.session_state.profile["liked_foods"].remove(food)
                st.session_state.profile.get("food_scores", {}).pop(food, None)
                save_profile(st.session_state.profile)

        elif action == "rm_dis":
            food = qp.get("food", "")
            if food in st.session_state.profile.get("disliked_foods", []):
                st.session_state.profile["disliked_foods"].remove(food)
                st.session_state.profile.get("food_scores", {}).pop(food, None)
                save_profile(st.session_state.profile)

        elif action == "budget":
            level = int(qp.get("level", "2"))
            st.session_state.profile["budget"] = BUDGET_REVERSE.get(level, "moderate")
            save_profile(st.session_state.profile)

        elif action == "prefill":
            target = qp.get("tab", "eat")
            text = qp.get("text", "")
            st.session_state[f"{target}_prefill"] = text

        elif action == "dismiss_hint":
            st.session_state.hint_dismissed = True

        elif action == "reset":
            reset_taste_profile()

        st.query_params.clear()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    profile = st.session_state.profile

    # Brand
    sidebar_html = '<div class="brand"><div class="brand-mark"></div><div class="brand-name">Food<em>AI</em></div></div>'

    # Cuisine pulse
    cs = profile.get("cuisine_scores", {})
    if cs:
        sorted_c = sorted(cs.items(), key=lambda x: -x[1])[:5]
        max_score = max(v for _, v in sorted_c) if sorted_c else 1
        rows = ""
        for name, score in sorted_c:
            display_name = " ".join(
                part for part in name.split()
                if part.lower() != "restaurant"
            ) or name
            pct = max(0, min(100, int((score / max(max_score, 0.01)) * 100)))
            rows += (
                f'<div class="cuisine-row">'
                f'<span class="name">{html_module.escape(display_name)}</span>'
                f'<span class="cuisine-bar"><i style="width:{pct}%"></i></span>'
                f'<span class="pct">{pct}</span>'
                f'</div>'
            )
        sidebar_html += (
            f'<div class="side-section">'
            f'<div class="side-label"><span>Cuisine pulse</span><span class="count">Top {len(sorted_c)}</span></div>'
            f'<div class="cuisine-list">{rows}</div>'
            f'</div>'
        )

    # Likes
    liked = profile.get("liked_foods", [])
    if liked:
        pills = "".join([
            f'<a href="?action=rm_like&food={quote(f)}" class="pill sage" target="_self">'
            f'<span class="pill-bar"></span>{html_module.escape(f)}<span class="x">×</span>'
            f'</a>'
            for f in liked
        ])
        sidebar_html += (
            f'<div class="side-section">'
            f'<div class="side-label"><span>You like</span><span class="count">{len(liked)}</span></div>'
            f'<div class="tag-cluster">{pills}</div>'
            f'</div>'
        )

    # Dislikes
    disliked = profile.get("disliked_foods", [])
    if disliked:
        pills = "".join([
            f'<a href="?action=rm_dis&food={quote(f)}" class="pill terracotta" target="_self">'
            f'<span class="pill-bar"></span>{html_module.escape(f)}<span class="x">×</span>'
            f'</a>'
            for f in disliked
        ])
        sidebar_html += (
            f'<div class="side-section">'
            f'<div class="side-label"><span>Not for you</span><span class="count">{len(disliked)}</span></div>'
            f'<div class="tag-cluster">{pills}</div>'
            f'</div>'
        )

    # Budget
    current_budget = BUDGET_MAP.get(profile.get("budget", "moderate"), 2)
    dots = "".join([
        f'<a href="?action=budget&level={n}" class="dot{" on" if n <= current_budget else ""}" target="_self"></a>'
        for n in [1, 2, 3, 4]
    ])
    sidebar_html += (
        f'<div class="side-section">'
        f'<div class="side-label"><span>Budget comfort</span><span class="count">{"$" * current_budget}</span></div>'
        f'<div class="budget">{dots}<span class="budget-label">{BUDGET_LABELS[current_budget]}</span></div>'
        f'</div>'
    )

    # Insights donut
    tc = st.session_state.tab_counts
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
        f'<div class="insights-head"><span>Profile insights</span><span class="pct">{pct}% positive</span></div>'
        f'<div class="donut-wrap">{donut_svg(tc["eat"], tc["cook"], tc["drink"], total)}{legend}</div>'
        f'</div></div>'
    )

    # History
    history = st.session_state.history[-6:][::-1]
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

    with st.sidebar:
        st.markdown(sidebar_html, unsafe_allow_html=True)
        render_reset_button()


# ── Greeting + hint + tabs ────────────────────────────────────────────────────

def render_greeting():
    greeting = "Good morning, " if time_now == "morning" else "Good afternoon, " if time_now == "afternoon" else "Good evening, " if time_now == "evening" else "Up for a midnight snack?"
    friend = "" if hour < 4 else "early bird" if hour < 12 else "Foodie"
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
        f'<div class="recent-track">{chips}</div>'
        f'</div>',
        unsafe_allow_html=True
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
    open_html = (
        f'<span class="card-extra"><span class="dot"></span>Open now</span>'
        if r.get("open_now") else ''
    )
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
                st.button(
                    "Pass",
                    key=pass_key,
                    on_click=apply_card_feedback,
                    args=(name, False, cuisines, tab),
                    use_container_width=True,
                )
                st.button(
                    "Save",
                    key=save_key,
                    on_click=apply_card_feedback,
                    args=(name, True, cuisines, tab),
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
        cards.append({
            "name": r.get("title", ""),
            "address": "",
            "categories": [category] if category else [],
            "price": None,
            "rating": None,
            "open_now": False,
            "total_tips": 0,
            "blurb": f"Curated match for {r.get('popular_food', 'this craving')}.",
        })
    return cards


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
    components.html(empty_html, height=288, scrolling=False)


# ── Tabs ──────────────────────────────────────────────────────────────────────
def render_eat_tab(client, df):
    prefill = st.session_state.eat_prefill or ""
    st.session_state.eat_prefill = ""

    with st.form(key="eat_form", enter_to_submit=True, border=False):
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
        st.markdown(f'<div class="results-head"><h2>{TAB_HEADING["eat"]}</h2><span class="count">{THINKING_MSG["eat"]}</span></div>', unsafe_allow_html=True)
        
        # Only show skeletons if there's no existing output
        skel_placeholder = None
        if not st.session_state.eat_llm_response:
            skel_placeholder = st.empty()
            with skel_placeholder.container():
                render_skeletons(5)

        search_notes = []
        try:
            _, retrieved = rag_recommend(client, query, st.session_state.profile, df, top_k=5)
        except Exception as e:
            retrieved = []
            search_notes.append(f"Curated retrieval skipped: {e}")
        st.session_state.eat_results = retrieved
        try:
            borough = zipcode if zipcode else "New York, NY"
            if zipcode:
                from src.places import geocode_location
                origin = geocode_location(zipcode + " New York")
            else:
                origin = (40.7128, -74.0060)
            st.session_state.eat_search_origin = origin
            _, fsq_restaurants = map_recommend(client, query, st.session_state.profile, borough=borough)
            st.session_state.eat_fsq_results = fsq_restaurants
        except Exception as e:
            st.session_state.eat_fsq_results = []
            search_notes.append(f"Live Places search skipped: {e}")

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

    with st.form(key="cook_form", enter_to_submit=True, border=False):
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
        pantry = [p.strip() for p in pantry_input.split(",") if p.strip()]
        st.session_state.profile["pantry"] = pantry
        save_profile(st.session_state.profile)

        st.markdown(f'<div class="results-head"><h2>{TAB_HEADING["cook"]}</h2><span class="count">{THINKING_MSG["cook"]}</span></div>', unsafe_allow_html=True)
        
        # Only show skeletons if there's no existing output
        skel_placeholder = None
        if not st.session_state.cook_response:
            skel_placeholder = st.empty()
            with skel_placeholder.container():
                render_skeletons(1)

        from src.recommend import recommend_recipe
        with st.spinner(""):
            response = recommend_recipe(craving, st.session_state.profile)
            st.session_state.cook_response = response
            st.session_state.tab_counts["cook"] += 1

        if skel_placeholder:
            skel_placeholder.empty()
        st.rerun()

    if st.session_state.cook_response:
        pantry = st.session_state.profile.get("pantry", [])
        match_text, match_kind = match_indicator(pantry, st.session_state.cook_response)
        match_html = ""
        if match_text:
            match_html = f'<span class="card-extra{" warn" if match_kind == "warn" else ""}"><span class="dot"></span>{match_text}</span>'

        st.markdown(
            f'<div class="results-head">'
            f'<h2>{TAB_HEADING["cook"]}</h2>'
            f'<span class="count">Your recipe</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        if match_html:
            st.markdown(f'<div style="margin-bottom:12px">{match_html}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="llm-response">{html_module.escape(st.session_state.cook_response)}</div>', unsafe_allow_html=True)
    else:
        render_empty("cook")


def render_cocktail_tab(client):
    prefill = st.session_state.drink_prefill or ""
    st.session_state.drink_prefill = ""

    with st.form(key="cocktail_form", enter_to_submit=True, border=False):
        st.markdown('<div class="field-label">The vibe</div>', unsafe_allow_html=True)
        vibe = st.text_input("vibe", value=prefill, placeholder="rainy night, pre-dinner, after a long week…", label_visibility="collapsed")
        st.markdown('<div class="field-label" style="margin-top:10px">Bar inventory</div>', unsafe_allow_html=True)
        bar_input = st.text_area(
            "bar",
            value=", ".join(st.session_state.profile.get("bar_inventory", [])),
            placeholder="bottles, mixers, fresh stuff",
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
        bar = [b.strip() for b in bar_input.split(",") if b.strip()]
        st.session_state.profile["bar_inventory"] = bar
        save_profile(st.session_state.profile)

        st.markdown(f'<div class="results-head"><h2>{TAB_HEADING["drink"]}</h2><span class="count">{THINKING_MSG["drink"]}</span></div>', unsafe_allow_html=True)
        
        # Only show skeletons if there's no existing output
        skel_placeholder = None
        if not st.session_state.cocktail_response:
            skel_placeholder = st.empty()
            with skel_placeholder.container():
                render_skeletons(1)

        from src.recommend import recommend_cocktail
        with st.spinner(""):
            response = recommend_cocktail(vibe, st.session_state.profile)
            st.session_state.cocktail_response = response
            st.session_state.tab_counts["drink"] += 1

        if skel_placeholder:
            skel_placeholder.empty()
        st.rerun()

    if st.session_state.cocktail_response:
        bar = st.session_state.profile.get("bar_inventory", [])
        match_text, match_kind = match_indicator(bar, st.session_state.cocktail_response)
        match_html = ""
        if match_text:
            match_html = f'<span class="card-extra{" warn" if match_kind == "warn" else ""}"><span class="dot"></span>{match_text}</span>'

        st.markdown(
            f'<div class="results-head">'
            f'<h2>{TAB_HEADING["drink"]}</h2>'
            f'<span class="count">Your drink</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        if match_html:
            st.markdown(f'<div style="margin-bottom:12px">{match_html}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="llm-response">{html_module.escape(st.session_state.cocktail_response)}</div>', unsafe_allow_html=True)
    else:
        render_empty("drink")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_session()
    handle_query_params()
    render_sidebar()

    render_greeting()
    render_hint()

    client = get_client()
    df = get_df()

    tab_eat, tab_cook, tab_drink = st.tabs(["🍽️  Eat Out", "🍳  Cook", "🍸  Cocktails"])

    with tab_eat:
        render_eat_tab(client, df)

    with tab_cook:
        render_cook_tab(client)

    with tab_drink:
        render_cocktail_tab(client)


if __name__ == "__main__":
    main()
