import os
import math
import html as html_module
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.data_loader import load_reviews
from src.recommend import rag_recommend, map_recommend
from src.taste_profile import load_profile, save_profile, update_profile
from src.places import PRICE_LABEL

load_dotenv()

st.set_page_config(
    page_title="Food AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Mono:wght@400&display=swap');

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
    --radius-sm: 8px; --radius: 14px; --radius-lg: 20px; --radius-pill: 999px;
    --serif: 'DM Serif Display', Georgia, serif;
    --sans: 'DM Sans', system-ui, sans-serif;
    --mono: 'IBM Plex Mono', ui-monospace, monospace;
}

html, body, .stApp { background-color: var(--bg) !important; color: var(--ink) !important; font-family: var(--sans) !important; font-weight: 350 !important; }
[data-testid="stAppViewContainer"] { background-color: var(--bg) !important; }
.main .block-container { background: transparent !important; padding-top: 1.5rem !important; padding-bottom: 6rem !important; max-width: 1060px !important; }
html, body { overscroll-behavior: none !important; }
.stApp { overscroll-behavior: none !important; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: var(--bg) !important; border-right: 1px solid var(--line) !important; }
[data-testid="stSidebar"] * { color: var(--ink) !important; }
[data-testid="collapsedControl"], [data-testid="collapsedControl"] button, [data-testid="collapsedControl"] svg { display: block !important; visibility: visible !important; opacity: 1 !important; pointer-events: auto !important; }

/* Brand */
.brand-wrap { display: flex; align-items: center; gap: 10px; padding-bottom: 2px; }
.brand-mark { width: 28px; height: 28px; border-radius: 50%; background: radial-gradient(circle at 35% 30%, #E89766 0%, var(--terracotta) 55%, #8A4521 100%); box-shadow: inset -2px -3px 6px rgba(0,0,0,0.15), 0 1px 2px rgba(0,0,0,0.08); flex-shrink: 0; }
.brand-name { font-family: var(--serif); font-size: 1.35rem; font-weight: 400; color: var(--ink); letter-spacing: -0.01em; }
.brand-name em { color: var(--terracotta); font-style: italic; }

/* Section labels */
.side-label { font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-3); display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.side-label .count { font-variant-numeric: tabular-nums; }

/* Cuisine bars */
.cuisine-list { display: flex; flex-direction: column; gap: 8px; margin-bottom: 4px; }
.cuisine-row { display: grid; grid-template-columns: 72px 1fr 28px; align-items: center; gap: 8px; font-size: 12.5px; }
.cuisine-bar { height: 5px; background: var(--bg-deep); border-radius: var(--radius-pill); overflow: hidden; }
.cuisine-fill { display: block; height: 100%; background: linear-gradient(90deg, var(--terracotta) 0%, #E08A5D 100%); border-radius: inherit; transition: width 0.6s cubic-bezier(.2,.8,.2,1); }
.cuisine-pct { font-family: var(--mono); font-size: 10px; color: var(--ink-3); text-align: right; }

/* Removable pills */
.remove-pill > button { background: var(--tag-sage) !important; color: var(--sage-2) !important; border: none !important; border-radius: var(--radius-pill) !important; padding: 3px 10px !important; font-size: 0.73rem !important; font-weight: 400 !important; box-shadow: none !important; transition: all 0.14s ease !important; min-height: 0 !important; height: auto !important; }
.remove-pill > button:hover { background: var(--sage) !important; color: #fff !important; transform: none !important; }
.remove-pill-dis > button { background: var(--tag-terracotta) !important; color: var(--terracotta-2) !important; border: none !important; border-radius: var(--radius-pill) !important; padding: 3px 10px !important; font-size: 0.73rem !important; font-weight: 400 !important; box-shadow: none !important; min-height: 0 !important; height: auto !important; }
.remove-pill-dis > button:hover { background: var(--terracotta) !important; color: #fff !important; transform: none !important; }
.pill-row { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 4px; }

/* Budget dots */
.budget-row { display: flex; align-items: center; gap: 6px; }
.budget-dot-active > button { background: var(--gold) !important; color: transparent !important; border: none !important; border-radius: var(--radius-pill) !important; width: 28px !important; height: 6px !important; min-height: 0 !important; padding: 0 !important; box-shadow: none !important; font-size: 0 !important; }
.budget-dot > button { background: var(--bg-deep) !important; color: transparent !important; border: none !important; border-radius: var(--radius-pill) !important; width: 28px !important; height: 6px !important; min-height: 0 !important; padding: 0 !important; box-shadow: none !important; font-size: 0 !important; }
.budget-dot > button:hover, .budget-dot-active > button:hover { background: rgba(201,162,39,0.5) !important; transform: none !important; }
.budget-label { font-family: var(--mono); font-size: 10px; color: var(--ink-3); margin-left: 2px; }

/* Insights / donut */
.insights-card { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius); padding: 12px 14px; margin-bottom: 4px; }
.insights-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
.insights-title { font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-3); }
.insights-pct { font-size: 10.5px; color: var(--sage-2); font-family: var(--mono); }
.donut-wrap { display: flex; align-items: center; gap: 12px; }
.legend { flex: 1; display: flex; flex-direction: column; gap: 3px; font-size: 11.5px; }
.legend-row { display: flex; justify-content: space-between; align-items: center; }
.legend-lbl { display: flex; align-items: center; gap: 5px; color: var(--ink-2); }
.legend-sw { width: 8px; height: 8px; border-radius: 2px; display: inline-block; }
.legend-num { font-family: var(--mono); font-size: 10.5px; color: var(--ink-3); }

/* History */
.history-list { display: flex; flex-direction: column; gap: 4px; }
.history-row { display: flex; align-items: center; gap: 8px; font-size: 12px; padding: 3px 0; }
.history-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.history-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--ink); font-size: 12px; }
.history-src { font-family: var(--mono); font-size: 9px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); }

/* Reset button */
.reset-wrap > button { background: transparent !important; border: 1px solid var(--line-2) !important; border-radius: var(--radius) !important; color: var(--ink-2) !important; font-size: 0.8rem !important; font-weight: 400 !important; box-shadow: none !important; width: 100% !important; }
.reset-wrap > button:hover { color: var(--ink) !important; border-color: var(--ink) !important; background: var(--card) !important; transform: none !important; }

/* Main header */
.foodai-header { font-family: var(--serif); font-size: 2.2rem; font-weight: 400; color: var(--ink); letter-spacing: -0.015em; margin-bottom: 0.2rem; line-height: 1.1; }
.foodai-header em { color: var(--terracotta); font-style: italic; }
.foodai-subtitle { color: var(--ink-2); font-size: 0.92rem; font-weight: 350; margin-bottom: 0.5rem; }

/* Onboarding hint */
.hint-box { display: flex; gap: 14px; align-items: flex-start; padding: 14px 16px; background: linear-gradient(180deg, #FFFBF5 0%, #FAF3E8 100%); border: 1px solid rgba(201,162,39,0.25); border-radius: var(--radius); margin-bottom: 18px; }
.hint-glyph { width: 30px; height: 30px; border-radius: 50%; background: var(--tag-gold); color: #8C7016; display: flex; align-items: center; justify-content: center; font-family: var(--serif); font-size: 15px; flex-shrink: 0; }
.hint-body { flex: 1; font-size: 13px; line-height: 1.5; }
.hint-body b { font-weight: 500; }
.hint-body p { margin: 3px 0 0; color: var(--ink-2); font-size: 12.5px; }
.hint-dismiss > button { background: transparent !important; border: none !important; color: var(--ink-3) !important; font-size: 1rem !important; box-shadow: none !important; padding: 0 !important; min-height: 0 !important; height: 22px !important; width: 22px !important; }

/* Recently saved strip */
.saved-strip { display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: var(--bg-deep); border-radius: var(--radius); margin-bottom: 18px; overflow: hidden; }
.saved-label { font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-3); flex-shrink: 0; }
.saved-track { display: flex; gap: 6px; flex-wrap: nowrap; overflow: hidden; flex: 1; }
.saved-chip { display: inline-flex; align-items: center; gap: 5px; padding: 3px 9px; background: var(--card); border-radius: var(--radius-pill); font-size: 11.5px; white-space: nowrap; border: 1px solid var(--line); color: var(--ink); }
.saved-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--sage); display: inline-block; }

/* Tabs */
[data-testid="stTabs"] [role="tablist"] { background: transparent !important; border-bottom: 1px solid var(--line) !important; border-radius: 0 !important; padding: 0 !important; }
[data-testid="stTabs"] [role="tab"] { color: var(--ink-2) !important; font-family: var(--sans) !important; font-weight: 400 !important; font-size: 0.88rem !important; border-radius: 0 !important; padding: 0.7rem 1.2rem !important; background: transparent !important; border-bottom: 2px solid transparent !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--ink) !important; font-weight: 500 !important; border-bottom: 2px solid var(--terracotta) !important; background: transparent !important; }

/* Inputs */
.stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div { background-color: var(--bg) !important; border: 1px solid var(--line-2) !important; border-radius: var(--radius-sm) !important; color: var(--ink) !important; font-family: var(--sans) !important; }
.stTextInput input, .stTextArea textarea { color: var(--ink) !important; background: var(--bg) !important; font-family: var(--sans) !important; font-size: 0.9rem !important; }
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--ink-3) !important; }

/* Suggest chips */
.suggest-chip-btn > button { background: transparent !important; border: 1px dashed var(--line-2) !important; border-radius: var(--radius-pill) !important; color: var(--ink-2) !important; font-size: 0.78rem !important; padding: 3px 10px !important; box-shadow: none !important; font-weight: 400 !important; min-height: 0 !important; }
.suggest-chip-btn > button:hover { border-color: var(--terracotta) !important; color: var(--terracotta-2) !important; border-style: solid !important; transform: none !important; }

/* Submit / primary button */
.stFormSubmitButton > button, .primary-btn > button { background: var(--terracotta) !important; color: #fff !important; border: none !important; border-radius: var(--radius-sm) !important; font-family: var(--sans) !important; font-weight: 500 !important; font-size: 0.875rem !important; box-shadow: 0 4px 14px rgba(201,106,58,0.30) !important; transition: all 0.15s ease !important; }
.stFormSubmitButton > button:hover, .primary-btn > button:hover { background: var(--terracotta-2) !important; transform: translateY(-1px) !important; box-shadow: 0 6px 18px rgba(201,106,58,0.36) !important; }

/* Match indicator */
.match-good { font-size: 12px; color: var(--sage-2); font-family: var(--mono); }
.match-warn { font-size: 12px; color: var(--gold); font-family: var(--mono); }

/* Skeleton */
.skeleton { background: var(--card); border-radius: var(--radius-lg); border: 1px solid var(--line); display: grid; grid-template-columns: 140px 1fr; overflow: hidden; min-height: 160px; margin-bottom: 14px; }
.skel-img { background: linear-gradient(90deg, var(--bg-deep) 0%, #EFE8DA 50%, var(--bg-deep) 100%); background-size: 200% 100%; animation: shimmer 1.4s linear infinite; }
.skel-body { padding: 18px; display: flex; flex-direction: column; gap: 10px; }
.skel-bar { height: 11px; border-radius: 4px; background: linear-gradient(90deg, var(--bg-deep) 0%, #EFE8DA 50%, var(--bg-deep) 100%); background-size: 200% 100%; animation: shimmer 1.4s linear infinite; }
.skel-bar.title { height: 20px; width: 55%; }
.skel-bar.short { width: 35%; }
.skel-bar.medium { width: 70%; }
@keyframes shimmer { 0% { background-position: 100% 0; } 100% { background-position: -100% 0; } }

/* Thinking label */
.thinking-label { font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-3); margin-bottom: 12px; }

/* Results header */
.results-head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 14px; margin-top: 6px; }
.results-head h3 { font-family: var(--serif); font-weight: 400; font-size: 1.3rem; letter-spacing: -0.01em; margin: 0; }
.results-count { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); }

/* Cards */
.restaurant-card { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius-lg); box-shadow: var(--shadow-card); display: grid; grid-template-columns: 140px 1fr; overflow: hidden; margin-bottom: 14px; transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.25s ease; }
.restaurant-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-pop); }
.restaurant-card.accepted { border-color: rgba(122,158,126,0.5) !important; }
.restaurant-card.rejected { opacity: 0.45; }
.card-img { position: relative; min-height: 160px; overflow: hidden; }
.card-img-inner { position: absolute; inset: 0; display: flex; align-items: flex-end; padding: 8px; }
.card-img::after { content: ""; position: absolute; inset: 0; background: repeating-linear-gradient(45deg, rgba(255,255,255,0.04) 0 8px, transparent 8px 16px); pointer-events: none; }
.card-img-label { background: rgba(255,255,255,0.88); backdrop-filter: blur(8px); padding: 3px 8px; border-radius: var(--radius-pill); font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.08em; color: var(--ink); text-transform: uppercase; position: relative; z-index: 1; }
.ph-warm-amber { background: linear-gradient(135deg, #E8B07A 0%, #C97A3D 100%); }
.ph-smoke-olive { background: linear-gradient(135deg, #B5A77E 0%, #6E7A56 100%); }
.ph-rice-gold   { background: linear-gradient(135deg, #F0DDA0 0%, #C9A227 100%); }
.ph-wine-rust   { background: linear-gradient(135deg, #B05546 0%, #6A2A2C 100%); }
.ph-broth-green { background: linear-gradient(135deg, #C9D4A0 0%, #6F8B5C 100%); }
.ph-pepper-cream{ background: linear-gradient(135deg, #F2EAD6 0%, #C9B07A 100%); }
.ph-char-orange { background: linear-gradient(135deg, #E89868 0%, #A14826 100%); }
.ph-salmon-glaze{ background: linear-gradient(135deg, #F0A788 0%, #B05A3F 100%); }
.ph-bean-paprika { background: linear-gradient(135deg, #E0C290 0%, #A85F30 100%); }
.ph-scallion-jade{ background: linear-gradient(135deg, #C4D6A8 0%, #5F8268 100%); }
.ph-amber-gold  { background: linear-gradient(135deg, #E8B870 0%, #A6722A 100%); }
.ph-garden-jade { background: linear-gradient(135deg, #B8CFA0 0%, #6A8E70 100%); }
.ph-spritz-rust { background: linear-gradient(135deg, #E89570 0%, #B0552E 100%); }
.ph-rye-sage    { background: linear-gradient(135deg, #C4B580 0%, #6E7E55 100%); }
.card-body { padding: 16px 18px 14px; display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.card-meta-top { font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-3); }
.card-title { font-family: var(--serif); font-weight: 400; font-size: 1.2rem; line-height: 1.15; letter-spacing: -0.01em; margin: 0; color: var(--ink); }
.card-rating { display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--ink); }
.card-stars { color: var(--gold); font-size: 11px; letter-spacing: -1px; }
.card-rating-num { font-variant-numeric: tabular-nums; }
.card-rating-sep { color: var(--ink-3); }
.card-address { color: var(--ink-3); font-size: 11.5px; }
.card-blurb { font-size: 13.5px; line-height: 1.55; color: var(--ink-2); margin: 3px 0 4px; }
.card-tags { display: flex; flex-wrap: wrap; gap: 4px; }
.card-tag { display: inline-block; background: var(--tag-ink); border-radius: var(--radius-pill); padding: 2px 9px; font-size: 11.5px; color: var(--ink-2); }
.card-actions { display: flex; gap: 7px; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--line); align-items: center; }
.card-feedback-done { font-family: var(--mono); font-size: 10.5px; color: var(--ink-3); padding-top: 10px; margin-top: 6px; border-top: 1px solid var(--line); }

/* Accept/reject card buttons */
.accept-btn > button { background: var(--card) !important; color: var(--sage-2) !important; border: 1px solid rgba(122,158,126,0.4) !important; border-radius: var(--radius-sm) !important; padding: 7px 13px !important; font-size: 0.8rem !important; font-weight: 500 !important; box-shadow: none !important; }
.accept-btn > button:hover { background: var(--sage) !important; color: #fff !important; border-color: var(--sage) !important; transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(122,158,126,0.30) !important; }
.reject-btn > button { background: var(--card) !important; color: var(--terracotta-2) !important; border: 1px solid rgba(201,106,58,0.4) !important; border-radius: var(--radius-sm) !important; padding: 7px 13px !important; font-size: 0.8rem !important; font-weight: 500 !important; box-shadow: none !important; }
.reject-btn > button:hover { background: var(--terracotta) !important; color: #fff !important; border-color: var(--terracotta) !important; transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(201,106,58,0.30) !important; }

/* LLM response */
.llm-response { background: linear-gradient(180deg, #FFFBF5 0%, #FAF3E8 100%); border: 1px solid rgba(201,162,39,0.2); border-radius: var(--radius); padding: 1.1rem 1.3rem; margin: 0.75rem 0 1.25rem; font-size: 0.875rem; line-height: 1.65; color: var(--ink); white-space: pre-wrap; }

/* Empty state */
.empty-state { background: var(--card); border: 1px dashed var(--line-2); border-radius: var(--radius-lg); padding: 48px 32px; text-align: center; display: flex; flex-direction: column; align-items: center; gap: 10px; }
.empty-glyph { width: 52px; height: 52px; border-radius: 50%; background: var(--tag-terracotta); display: flex; align-items: center; justify-content: center; color: var(--terracotta); font-family: var(--serif); font-size: 26px; margin-bottom: 4px; }
.empty-title { font-family: var(--serif); font-weight: 400; font-size: 1.3rem; letter-spacing: -0.01em; margin: 0; }
.empty-body { color: var(--ink-2); font-size: 13.5px; max-width: 38ch; margin: 0; line-height: 1.5; }
.empty-chips { display: flex; gap: 7px; flex-wrap: wrap; justify-content: center; margin-top: 6px; }

/* Misc */
label, .stSelectbox label, .stTextInput label, .stTextArea label { color: var(--ink-3) !important; font-size: 0.7rem !important; font-weight: 400 !important; font-family: var(--mono) !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }
[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--line) !important; border-radius: var(--radius-sm) !important; padding: 0.75rem !important; box-shadow: var(--shadow-card) !important; }
[data-testid="stMetricValue"] { color: var(--terracotta) !important; font-family: var(--serif) !important; font-size: 1.4rem !important; }
[data-testid="stMetricLabel"] { color: var(--ink-3) !important; font-size: 0.68rem !important; font-family: var(--mono) !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
hr { border-color: var(--line) !important; margin: 1rem 0 !important; }
.stSpinner > div { border-top-color: var(--terracotta) !important; }
.stCheckbox label { color: var(--ink-2) !important; font-size: 0.875rem !important; font-family: var(--sans) !important; letter-spacing: 0 !important; text-transform: none !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cuisine → gradient class mapping ─────────────────────────────────────────
CUISINE_GRADIENTS = {
    "italian": "ph-warm-amber", "pizza": "ph-wine-rust",
    "japanese": "ph-scallion-jade", "ramen": "ph-broth-green",
    "korean": "ph-bean-paprika", "chinese": "ph-smoke-olive",
    "mexican": "ph-char-orange", "american": "ph-rice-gold",
    "seafood": "ph-salmon-glaze", "mediterranean": "ph-garden-jade",
    "cocktail": "ph-spritz-rust", "bar": "ph-rye-sage",
    "breakfast": "ph-pepper-cream", "thai": "ph-broth-green",
    "indian": "ph-amber-gold", "french": "ph-pepper-cream",
    "greek": "ph-garden-jade", "vietnamese": "ph-scallion-jade",
}

BUDGET_LABELS = {1: "Cheap", 2: "Smart", 3: "Treat", 4: "Splurge"}
BUDGET_MAP = {"budget": 1, "moderate": 2, "premium": 3, "premium+": 4}
BUDGET_REVERSE = {1: "budget", 2: "moderate", 3: "premium", 4: "premium+"}

QUICK_STARTS = {
    "eat":  ["something cozy", "date night Italian", "wood-fired anything", "big portions"],
    "cook": ["quick weeknight", "use up the pasta", "something to impress", "one-pot"],
    "drink": ["rainy night in", "pre-dinner aperitivo", "something bitter", "low ABV"],
}

EMPTY_COPY = {
    "eat":   {"glyph": "✦", "title": "Where shall we eat?", "body": "Tell us what you're craving — vague or specific, anything goes."},
    "cook":  {"glyph": "◐", "title": "What's in the kitchen?", "body": "Drop your craving and what's in the fridge. We'll work backward from there."},
    "drink": {"glyph": "◑", "title": "Pick a vibe.", "body": "A mood, a season — we'll match it to your bar cart."},
}

THINKING_MSG = {
    "eat":   "Reading the room…",
    "cook":  "Browsing your shelf…",
    "drink": "Mixing ideas…",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@st.cache_data
def get_df():
    return load_reviews(path="data/restaurants.csv", max_rows=3000)


def get_gradient(categories):
    if not categories:
        return "ph-amber-gold"
    for cat in categories[:2]:
        for key, grad in CUISINE_GRADIENTS.items():
            if key in cat.lower():
                return grad
    return "ph-amber-gold"


def stars_html(rating):
    if not rating:
        return ""
    full = int(rating)
    half = (rating - full) >= 0.5
    empty = 5 - full - (1 if half else 0)
    return "★" * full + ("½" if half else "") + "☆" * empty


def donut_svg(eat, cook, drink, total):
    if total == 0:
        return '<svg viewBox="0 0 64 64" width="64" height="64"><circle cx="32" cy="32" r="24" fill="none" stroke="#F2EDE3" stroke-width="8"/><text x="32" y="37" text-anchor="middle" font-family="DM Serif Display,serif" font-size="16" fill="#1A1A1A">0</text></svg>'

    r = 24
    C = 2 * math.pi * r
    s = max(eat + cook + drink, 1)
    el = C * eat / s
    cl = C * cook / s
    dl = C * drink / s

    def arc(color, dashlen, offset):
        return f'<circle cx="32" cy="32" r="{r}" fill="none" stroke="{color}" stroke-width="8" stroke-dasharray="{dashlen:.1f} {C - dashlen:.1f}" stroke-dashoffset="{-offset:.1f}" transform="rotate(-90 32 32)"/>'

    return f'''<svg viewBox="0 0 64 64" width="64" height="64">
  <circle cx="32" cy="32" r="{r}" fill="none" stroke="#F2EDE3" stroke-width="8"/>
  {arc("#C96A3A", el, 0)}
  {arc("#7A9E7E", cl, el)}
  {arc("#C9A227", dl, el + cl)}
  <text x="32" y="37" text-anchor="middle" font-family="DM Serif Display,serif" font-size="16" fill="#1A1A1A">{total}</text>
</svg>'''


def match_indicator(inventory, response_text):
    if not inventory or not response_text:
        return None
    matched = [item for item in inventory if item.lower() in response_text.lower()]
    total = len(inventory)
    count = len(matched)
    if count == total:
        return f"✓ all {total} items on hand", "match-good"
    elif count > 0:
        return f"⚠ {count} / {total} items on hand", "match-warn"
    return None, None


# ── Session state init ────────────────────────────────────────────────────────
def init_session():
    if "profile" not in st.session_state:
        profile = load_profile()
        if not profile["preferred_cuisines"] and not profile["cuisine_scores"]:
            profile.update({
                "preferred_cuisines": ["Italian", "Pizza"],
                "liked_foods": ["pasta", "pizza"],
                "disliked_foods": ["seafood"],
                "budget": "moderate",
                "online_order": "Yes",
                "occasion": "casual dinner",
            })
            save_profile(profile)
        st.session_state.profile = profile

    defaults = {
        "eat_results": None, "eat_fsq_results": None,
        "eat_llm_response": None, "cook_response": None,
        "cocktail_response": None, "feedback_given": set(),
        "hint_dismissed": False,
        "tab_counts": {"eat": 0, "cook": 0, "drink": 0},
        "history": [],
        "eat_prefill": "", "cook_prefill": "", "drink_prefill": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="brand-wrap">'
            '<div class="brand-mark"></div>'
            '<div class="brand-name">Food <em>AI</em></div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")

        profile = st.session_state.profile

        # Cuisine pulse bars
        cuisine_scores = profile.get("cuisine_scores", {})
        if cuisine_scores:
            sorted_c = sorted(cuisine_scores.items(), key=lambda x: -x[1])[:5]
            max_score = max(v for _, v in sorted_c) if sorted_c else 1
            st.markdown(
                f'<div class="side-label"><span>Cuisine pulse</span><span class="count">top {len(sorted_c)}</span></div>',
                unsafe_allow_html=True
            )
            bars_html = '<div class="cuisine-list">'
            for name, score in sorted_c:
                pct = max(0, min(100, int((score / max(max_score, 0.01)) * 100)))
                bars_html += (
                    f'<div class="cuisine-row">'
                    f'<span style="font-size:12px;color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{name}</span>'
                    f'<span class="cuisine-bar"><span class="cuisine-fill" style="width:{pct}%"></span></span>'
                    f'<span class="cuisine-pct">{pct}</span>'
                    f'</div>'
                )
            bars_html += '</div>'
            st.markdown(bars_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # Removable likes
        liked = profile.get("liked_foods", [])
        if liked:
            st.markdown(
                f'<div class="side-label"><span>You like</span><span class="count">{len(liked)}</span></div>',
                unsafe_allow_html=True
            )
            pill_cols = st.columns(len(liked) if len(liked) <= 4 else 4)
            for i, food in enumerate(liked):
                with pill_cols[i % 4]:
                    st.markdown('<div class="remove-pill">', unsafe_allow_html=True)
                    if st.button(f"✕ {food}", key=f"rm_like_{food}"):
                        profile["liked_foods"].remove(food)
                        profile["food_scores"].pop(food, None)
                        save_profile(profile)
                        st.session_state.profile = profile
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # Removable dislikes
        disliked = profile.get("disliked_foods", [])
        if disliked:
            st.markdown(
                f'<div class="side-label"><span>Not for you</span><span class="count">{len(disliked)}</span></div>',
                unsafe_allow_html=True
            )
            d_cols = st.columns(len(disliked) if len(disliked) <= 4 else 4)
            for i, food in enumerate(disliked):
                with d_cols[i % 4]:
                    st.markdown('<div class="remove-pill-dis">', unsafe_allow_html=True)
                    if st.button(f"✕ {food}", key=f"rm_dis_{food}"):
                        profile["disliked_foods"].remove(food)
                        profile["food_scores"].pop(food, None)
                        save_profile(profile)
                        st.session_state.profile = profile
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # Budget dots
        current_budget_level = BUDGET_MAP.get(profile.get("budget", "moderate"), 2)
        st.markdown(
            f'<div class="side-label"><span>Budget comfort</span><span class="count">{"$" * current_budget_level}</span></div>',
            unsafe_allow_html=True
        )
        b_cols = st.columns([1, 1, 1, 1, 2])
        for i in range(4):
            level = i + 1
            active = level <= current_budget_level
            css_class = "budget-dot-active" if active else "budget-dot"
            with b_cols[i]:
                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                if st.button(" ", key=f"bdgt_{level}"):
                    profile["budget"] = BUDGET_REVERSE[level]
                    save_profile(profile)
                    st.session_state.profile = profile
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        with b_cols[4]:
            st.markdown(
                f'<span class="budget-label">{BUDGET_LABELS[current_budget_level]}</span>',
                unsafe_allow_html=True
            )
        st.markdown("<br>", unsafe_allow_html=True)

        # Donut insights
        tc = st.session_state.tab_counts
        total = tc["eat"] + tc["cook"] + tc["drink"]
        accepted = len(profile.get("accepted", []))
        rejected = len(profile.get("rejected", []))
        all_decisions = accepted + rejected
        pct = round((accepted / all_decisions) * 100) if all_decisions > 0 else 0

        donut = donut_svg(tc["eat"], tc["cook"], tc["drink"], total)
        legend = (
            '<div class="legend">'
            f'<div class="legend-row"><span class="legend-lbl"><span class="legend-sw" style="background:var(--terracotta)"></span>Eat out</span><span class="legend-num">{tc["eat"]}</span></div>'
            f'<div class="legend-row"><span class="legend-lbl"><span class="legend-sw" style="background:var(--sage)"></span>Cook</span><span class="legend-num">{tc["cook"]}</span></div>'
            f'<div class="legend-row"><span class="legend-lbl"><span class="legend-sw" style="background:var(--gold)"></span>Cocktails</span><span class="legend-num">{tc["drink"]}</span></div>'
            '</div>'
        )
        st.markdown(
            f'<div class="insights-card">'
            f'<div class="insights-head"><span class="insights-title">Profile insights</span><span class="insights-pct">{pct}% positive</span></div>'
            f'<div class="donut-wrap">{donut}{legend}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Recent decisions
        history = st.session_state.history[-6:][::-1]
        st.markdown(
            f'<div class="side-label"><span>Recent decisions</span><span class="count">{all_decisions}</span></div>',
            unsafe_allow_html=True
        )
        if not history:
            st.markdown('<div style="font-size:12px;color:var(--ink-3);font-style:italic;padding:4px 0">No history yet</div>', unsafe_allow_html=True)
        else:
            rows = '<div class="history-list">'
            for h in history:
                color = "var(--sage)" if h["kind"] == "acc" else "var(--terracotta)"
                src = {"eat": "EAT", "cook": "COOK", "drink": "BAR"}.get(h.get("tab", "eat"), "EAT")
                rows += (
                    f'<div class="history-row">'
                    f'<span class="history-dot" style="background:{color}"></span>'
                    f'<span class="history-name">{html_module.escape(h["name"])}</span>'
                    f'<span class="history-src">{src}</span>'
                    f'</div>'
                )
            rows += '</div>'
            st.markdown(rows, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="reset-wrap">', unsafe_allow_html=True)
        if st.button("↺  Reset taste profile", key="reset_profile"):
            if os.path.exists("data/taste_profile.json"):
                os.remove("data/taste_profile.json")
            st.session_state.profile = load_profile()
            st.session_state.history = []
            st.session_state.tab_counts = {"eat": 0, "cook": 0, "drink": 0}
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ── Feedback helper ───────────────────────────────────────────────────────────
def handle_feedback(restaurant_name, accepted, cuisines=None, foods=None, key=None, tab="eat"):
    if key and key in st.session_state.feedback_given:
        return
    st.session_state.profile = update_profile(
        st.session_state.profile,
        restaurant_name=restaurant_name,
        accepted=accepted,
        cuisines=cuisines,
        foods=foods,
    )
    save_profile(st.session_state.profile)
    if key:
        st.session_state.feedback_given.add(key)
    st.session_state.history.append({"name": restaurant_name, "kind": "acc" if accepted else "rej", "tab": tab})
    st.session_state.tab_counts[tab] += 1


# ── Skeleton loader ───────────────────────────────────────────────────────────
def render_skeletons(n=3):
    html_out = ""
    for _ in range(n):
        html_out += (
            '<div class="skeleton">'
            '<div class="skel-img"></div>'
            '<div class="skel-body">'
            '<div class="skel-bar short"></div>'
            '<div class="skel-bar title"></div>'
            '<div class="skel-bar medium"></div>'
            '<div class="skel-bar short"></div>'
            '</div></div>'
        )
    st.markdown(html_out, unsafe_allow_html=True)


# ── Empty state ───────────────────────────────────────────────────────────────
def render_empty_state(tab, on_suggest_key):
    copy = EMPTY_COPY[tab]
    chips = QUICK_STARTS[tab]

    st.markdown(
        f'<div class="empty-state">'
        f'<div class="empty-glyph">{copy["glyph"]}</div>'
        f'<div class="empty-title">{copy["title"]}</div>'
        f'<div class="empty-body">{copy["body"]}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;justify-content:center">', unsafe_allow_html=True)
    chip_cols = st.columns(len(chips))
    for i, chip in enumerate(chips):
        with chip_cols[i]:
            st.markdown('<div class="suggest-chip-btn">', unsafe_allow_html=True)
            if st.button(chip, key=f"chip_{tab}_{i}"):
                st.session_state[on_suggest_key] = chip
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Restaurant card ───────────────────────────────────────────────────────────
def render_restaurant_card(r, source="fsq", idx=0, blurb="", tab="eat"):
    key = f"{source}_{html_module.escape(str(r.get('name', idx)))}_{idx}"
    already_accepted = r.get("name", "") in st.session_state.profile.get("accepted", [])
    already_rejected = r.get("name", "") in st.session_state.profile.get("rejected", [])

    card_class = "restaurant-card"
    if already_accepted:
        card_class += " accepted"
    elif already_rejected:
        card_class += " rejected"

    price_str = PRICE_LABEL.get(r.get("price"), "")
    rating = r.get("rating")
    cats = ", ".join(r.get("categories", [])[:2]) if r.get("categories") else r.get("category", "")
    name = r.get("name") or r.get("title", "")
    address = r.get("address", "")
    gradient = get_gradient(r.get("categories", []))

    # Stars
    stars = f'<span class="card-stars">{stars_html(rating)}</span> <span class="card-rating-num">{rating}</span>' if rating else ""
    rating_html = (
        f'<div class="card-rating">{stars}'
        + (f'<span class="card-rating-sep">·</span><span class="card-address">{html_module.escape(address)}</span>' if address else "")
        + '</div>'
    ) if (rating or address) else ""

    # Tags
    tag_list = [t for t in r.get("categories", [])[:3] if t]
    if price_str:
        tag_list.insert(0, price_str)
    tags_html = "".join([f'<span class="card-tag">{html_module.escape(t)}</span>' for t in tag_list])

    blurb_html = f'<div class="card-blurb">{html_module.escape(blurb)}</div>' if blurb else ""

    st.markdown(
        f'<div class="{card_class}">'
        f'<div class="card-img {gradient}">'
        f'<div class="card-img-inner"><span class="card-img-label">{html_module.escape(cats[:20]) if cats else "restaurant"}</span></div>'
        f'</div>'
        f'<div class="card-body">'
        f'<div class="card-meta-top">{html_module.escape(cats)}</div>'
        f'<h3 class="card-title">{html_module.escape(name)}</h3>'
        f'{rating_html}'
        f'{blurb_html}'
        f'<div class="card-tags">{tags_html}</div>'
        f'</div></div>',
        unsafe_allow_html=True
    )

    if not already_accepted and not already_rejected:
        col1, col2, _ = st.columns([1, 1, 5])
        with col1:
            st.markdown('<div class="accept-btn">', unsafe_allow_html=True)
            if st.button("✓ Save", key=f"accept_{key}"):
                handle_feedback(name, accepted=True, cuisines=[cats.split(",")[0].strip()] if cats else None, key=f"fb_{key}", tab=tab)
                st.toast(f"Saved {name}", icon="✓")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="reject-btn">', unsafe_allow_html=True)
            if st.button("✕ Pass", key=f"reject_{key}"):
                handle_feedback(name, accepted=False, cuisines=[cats.split(",")[0].strip()] if cats else None, key=f"fb_{key}", tab=tab)
                st.toast(f"Passed on {name}", icon="✕")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    elif already_accepted:
        st.markdown('<div class="card-feedback-done" style="color:var(--sage-2)">✓ saved</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card-feedback-done" style="color:var(--terracotta)">✕ passed</div>', unsafe_allow_html=True)


# ── Onboarding hint ───────────────────────────────────────────────────────────
def render_hint():
    if st.session_state.hint_dismissed:
        return
    col1, col2 = st.columns([10, 1])
    with col1:
        st.markdown(
            '<div class="hint-box">'
            '<div class="hint-glyph">✦</div>'
            '<div class="hint-body">'
            '<b>Your taste profile learns as you go.</b>'
            '<p>Accept or pass on recommendations — every decision updates your cuisine scores and personalises future results.</p>'
            '</div></div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown('<div class="hint-dismiss">', unsafe_allow_html=True)
        if st.button("×", key="dismiss_hint"):
            st.session_state.hint_dismissed = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ── Recently saved strip ──────────────────────────────────────────────────────
def render_saved_strip():
    accepted = st.session_state.profile.get("accepted", [])
    if not accepted:
        return
    chips = "".join([
        f'<span class="saved-chip"><span class="saved-dot"></span>{html_module.escape(n)}</span>'
        for n in accepted[-5:]
    ])
    st.markdown(
        f'<div class="saved-strip">'
        f'<span class="saved-label">Saved</span>'
        f'<div class="saved-track">{chips}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ── Eat Out tab ───────────────────────────────────────────────────────────────
def render_eat_out_tab(client, df):
    with st.form(key="eat_form", enter_to_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            prefill = st.session_state.pop("eat_prefill", "") if st.session_state.get("eat_prefill") else ""
            query = st.text_input("What are you craving?", value=prefill, placeholder="hand-rolled pasta, candlelit, walking distance…")
        with col2:
            zipcode = st.text_input("Near", placeholder="ZIP code")

        # Suggest chips inside form
        chip_cols = st.columns(4)
        for i, chip in enumerate(QUICK_STARTS["eat"]):
            with chip_cols[i]:
                st.markdown('<div class="suggest-chip-btn">', unsafe_allow_html=True)
                st.form_submit_button(chip, on_click=lambda c=chip: st.session_state.update({"eat_prefill": c}))
                st.markdown('</div>', unsafe_allow_html=True)

        run_search = st.form_submit_button("Find restaurants →")

    if run_search and query:
        placeholder = st.empty()
        with placeholder.container():
            st.markdown(f'<div class="thinking-label">{THINKING_MSG["eat"]}</div>', unsafe_allow_html=True)
            render_skeletons(3)

        with st.spinner(""):
            _, retrieved = rag_recommend(client, query, st.session_state.profile, df, top_k=5)
            st.session_state.eat_results = retrieved
            try:
                borough = zipcode if zipcode else "New York, NY"
                _, fsq_restaurants = map_recommend(client, query, st.session_state.profile, borough=borough)
                st.session_state.eat_fsq_results = fsq_restaurants
            except Exception as e:
                st.session_state.eat_fsq_results = []

            from src.recommend import combined_recommend
            response, selected = combined_recommend(
                client, query, st.session_state.profile,
                retrieved, st.session_state.eat_fsq_results or []
            )
            st.session_state.eat_llm_response = response
            st.session_state.eat_fsq_results = selected

        placeholder.empty()

    if st.session_state.eat_llm_response:
        results = st.session_state.eat_fsq_results or []
        st.markdown(
            f'<div class="results-head">'
            f'<h3>Top picks for you</h3>'
            f'<span class="results-count">{len(results)} recommendations</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        for i, r in enumerate(results):
            render_restaurant_card(r, source="fsq", idx=i, blurb=r.get("blurb", ""), tab="eat")
    elif not st.session_state.eat_llm_response:
        render_empty_state("eat", "eat_prefill")


# ── Cook tab ──────────────────────────────────────────────────────────────────
def render_cook_tab(client):
    with st.form(key="cook_form", enter_to_submit=True):
        prefill = st.session_state.pop("cook_prefill", "") if st.session_state.get("cook_prefill") else ""
        craving = st.text_input("Tonight you want", value=prefill, placeholder="something fast, something cozy, something to impress…")
        pantry_input = st.text_area(
            "In the pantry",
            value=", ".join(st.session_state.profile.get("pantry", [])) if st.session_state.profile.get("pantry") else "",
            placeholder="comma-separated — eggs, pasta, garlic, lemon, olive oil…",
            height=88,
        )

        chip_cols = st.columns(4)
        for i, chip in enumerate(QUICK_STARTS["cook"]):
            with chip_cols[i]:
                st.markdown('<div class="suggest-chip-btn">', unsafe_allow_html=True)
                st.form_submit_button(chip, on_click=lambda c=chip: st.session_state.update({"cook_prefill": c}))
                st.markdown('</div>', unsafe_allow_html=True)

        run_cook = st.form_submit_button("Suggest recipes →")

    if run_cook and craving:
        pantry = [p.strip() for p in pantry_input.split(",") if p.strip()]
        st.session_state.profile["pantry"] = pantry
        save_profile(st.session_state.profile)

        placeholder = st.empty()
        with placeholder.container():
            st.markdown(f'<div class="thinking-label">{THINKING_MSG["cook"]}</div>', unsafe_allow_html=True)
            render_skeletons(1)

        from src.recommend import recommend_recipe
        with st.spinner(""):
            response = recommend_recipe(craving, st.session_state.profile)
            st.session_state.cook_response = response
            st.session_state.tab_counts["cook"] += 1

        placeholder.empty()

    if st.session_state.cook_response:
        pantry = st.session_state.profile.get("pantry", [])
        match_text, match_class = match_indicator(pantry, st.session_state.cook_response)
        if match_text:
            st.markdown(f'<div class="{match_class}" style="margin-bottom:8px">● {match_text}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="results-head"><h3>Your recipe</h3></div>',
            unsafe_allow_html=True
        )
        st.markdown(f'<div class="llm-response">{html_module.escape(st.session_state.cook_response)}</div>', unsafe_allow_html=True)
    else:
        render_empty_state("cook", "cook_prefill")


# ── Cocktail tab ──────────────────────────────────────────────────────────────
def render_cocktail_tab(client):
    with st.form(key="cocktail_form", enter_to_submit=True):
        prefill = st.session_state.pop("drink_prefill", "") if st.session_state.get("drink_prefill") else ""
        vibe = st.text_input("The vibe", value=prefill, placeholder="rainy night, pre-dinner, after a long week…")
        bar_input = st.text_area(
            "Bar inventory",
            value=", ".join(st.session_state.profile.get("bar_inventory", [])) if st.session_state.profile.get("bar_inventory") else "",
            placeholder="bottles, mixers, fresh stuff — rye, gin, lime, ginger, honey…",
            height=88,
        )
        mocktail = st.checkbox("Mocktail only")

        chip_cols = st.columns(4)
        for i, chip in enumerate(QUICK_STARTS["drink"]):
            with chip_cols[i]:
                st.markdown('<div class="suggest-chip-btn">', unsafe_allow_html=True)
                st.form_submit_button(chip, on_click=lambda c=chip: st.session_state.update({"drink_prefill": c}))
                st.markdown('</div>', unsafe_allow_html=True)

        run_cocktail = st.form_submit_button("Mix something →")

    if run_cocktail and vibe:
        bar = [b.strip() for b in bar_input.split(",") if b.strip()]
        st.session_state.profile["bar_inventory"] = bar
        save_profile(st.session_state.profile)

        placeholder = st.empty()
        with placeholder.container():
            st.markdown(f'<div class="thinking-label">{THINKING_MSG["drink"]}</div>', unsafe_allow_html=True)
            render_skeletons(1)

        from src.recommend import recommend_cocktail
        full_vibe = vibe + (" (mocktail, no alcohol)" if mocktail else "")
        with st.spinner(""):
            response = recommend_cocktail(full_vibe, st.session_state.profile)
            st.session_state.cocktail_response = response
            st.session_state.tab_counts["drink"] += 1

        placeholder.empty()

    if st.session_state.cocktail_response:
        bar = st.session_state.profile.get("bar_inventory", [])
        match_text, match_class = match_indicator(bar, st.session_state.cocktail_response)
        if match_text:
            st.markdown(f'<div class="{match_class}" style="margin-bottom:8px">● {match_text}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="results-head"><h3>Your drink</h3></div>',
            unsafe_allow_html=True
        )
        st.markdown(f'<div class="llm-response">{html_module.escape(st.session_state.cocktail_response)}</div>', unsafe_allow_html=True)
    else:
        render_empty_state("drink", "drink_prefill")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_session()
    render_sidebar()

    st.markdown(
        '<div class="foodai-header">What are you <em>hungry</em> for?</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="foodai-subtitle">Tell me what you want. I\'ll figure out the rest.</div>',
        unsafe_allow_html=True
    )

    render_hint()
    render_saved_strip()

    client = get_client()
    df = get_df()

    tab_eat, tab_cook, tab_cocktail = st.tabs(["🍜  Eat Out", "🍳  Cook", "🍹  Cocktails"])

    with tab_eat:
        render_eat_out_tab(client, df)

    with tab_cook:
        render_cook_tab(client)

    with tab_cocktail:
        render_cocktail_tab(client)


if __name__ == "__main__":
    main()