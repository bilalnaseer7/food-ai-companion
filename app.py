import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import html

from src.data_loader import load_reviews
from src.recommend import baseline_recommend, rag_recommend, map_recommend
from src.taste_profile import load_profile, save_profile, update_profile, profile_summary
from src.places import PRICE_LABEL

load_dotenv()

st.set_page_config(
    page_title="Food AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

    :root {
        --bg: #FAF7F2;
        --bg-deep: #F2EDE3;
        --card: #FFFFFF;
        --ink: #1A1A1A;
        --ink-2: #6B6B6B;
        --ink-3: #9A968D;
        --line: rgba(26, 26, 26, 0.06);
        --line-2: rgba(26, 26, 26, 0.10);

        --terracotta: #C96A3A;
        --terracotta-2: #B25A2C;
        --sage: #7A9E7E;
        --sage-2: #688C6D;
        --gold: #C9A227;

        --tag-terracotta: rgba(201, 106, 58, 0.13);
        --tag-sage: rgba(122, 158, 126, 0.15);
        --tag-gold: rgba(201, 162, 39, 0.16);
        --tag-ink: rgba(26, 26, 26, 0.05);

        --shadow-card: 0 1px 0 rgba(26,26,26,0.02), 0 6px 18px rgba(60,40,20,0.06), 0 24px 60px -30px rgba(60,40,20,0.12);
        --shadow-pop: 0 1px 0 rgba(26,26,26,0.02), 0 12px 32px rgba(60,40,20,0.10), 0 32px 80px -40px rgba(60,40,20,0.18);

        --radius-sm: 8px;
        --radius: 14px;
        --radius-lg: 20px;
        --radius-pill: 999px;

        --serif: 'DM Serif Display', Georgia, serif;
        --sans: 'DM Sans', ui-sans-serif, system-ui, sans-serif;
        --mono: 'IBM Plex Mono', ui-monospace, monospace;
    }

    html, body, .stApp {
        background-color: var(--bg) !important;
        color: var(--ink) !important;
        font-family: var(--sans) !important;
        font-weight: 350 !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important;
    }

    .main .block-container {
        background: transparent !important;
        padding-top: 2rem !important;
        padding-bottom: 6rem !important;
        max-width: 1100px !important;
    }

    html, body {
        overscroll-behavior: none !important;
        -webkit-overflow-scrolling: auto !important;
    }

    .stApp {
        overscroll-behavior: none !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--bg) !important;
        border-right: 1px solid var(--line) !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }

    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] button,
    [data-testid="collapsedControl"] svg {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
    }

    /* ── Header ── */
    .foodai-header {
        font-family: var(--serif) !important;
        font-size: 2.4rem !important;
        font-weight: 400 !important;
        color: var(--ink) !important;
        letter-spacing: -0.015em;
        margin-bottom: 0.2rem;
        line-height: 1.1;
    }

    .foodai-header em {
        color: var(--terracotta);
        font-style: italic;
    }

    .foodai-subtitle {
        color: var(--ink-2) !important;
        font-size: 0.95rem !important;
        font-weight: 350 !important;
        margin-bottom: 1.5rem !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [role="tablist"] {
        background: transparent !important;
        border-bottom: 1px solid var(--line) !important;
        border-radius: 0 !important;
        padding: 0 !important;
        gap: 0 !important;
    }

    [data-testid="stTabs"] [role="tab"] {
        color: var(--ink-2) !important;
        font-family: var(--sans) !important;
        font-weight: 400 !important;
        font-size: 0.9rem !important;
        border-radius: 0 !important;
        padding: 0.75rem 1.25rem !important;
        border-bottom: 2px solid transparent !important;
        background: transparent !important;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--ink) !important;
        font-weight: 500 !important;
        border-bottom: 2px solid var(--terracotta) !important;
        background: transparent !important;
    }

    /* ── Inputs ── */
    .stTextInput > div > div,
    .stTextArea > div > div,
    .stSelectbox > div > div {
        background-color: var(--bg) !important;
        border: 1px solid var(--line-2) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--ink) !important;
        font-family: var(--sans) !important;
    }

    .stTextInput input,
    .stTextArea textarea {
        color: var(--ink) !important;
        background: var(--bg) !important;
        font-family: var(--sans) !important;
        font-size: 0.9rem !important;
    }

    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: var(--terracotta) !important;
        box-shadow: 0 0 0 3px rgba(201, 106, 58, 0.12) !important;
        background: var(--card) !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: var(--ink-3) !important;
    }

    /* ── Buttons ── */
    .stButton > button,
    .stFormSubmitButton > button {
        background: var(--terracotta) !important;
        color: #fff !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-family: var(--sans) !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: 0 4px 14px rgba(201, 106, 58, 0.30) !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: var(--terracotta-2) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(201,106,58,0.36) !important;
    }

    /* Accept/reject buttons */
    .accept-btn > button {
        background: var(--card) !important;
        color: var(--sage-2) !important;
        border: 1px solid rgba(122, 158, 126, 0.4) !important;
        box-shadow: none !important;
        font-size: 0.8rem !important;
        padding: 0.35rem 0.9rem !important;
    }

    .accept-btn > button:hover {
        background: var(--sage) !important;
        color: #fff !important;
        border-color: var(--sage) !important;
        box-shadow: 0 4px 12px rgba(122,158,126,0.30) !important;
    }

    .reject-btn > button {
        background: var(--card) !important;
        color: var(--terracotta-2) !important;
        border: 1px solid rgba(201, 106, 58, 0.4) !important;
        box-shadow: none !important;
        font-size: 0.8rem !important;
        padding: 0.35rem 0.9rem !important;
    }

    .reject-btn > button:hover {
        background: var(--terracotta) !important;
        color: #fff !important;
        border-color: var(--terracotta) !important;
        box-shadow: 0 4px 12px rgba(201,106,58,0.30) !important;
    }

    /* ── Restaurant cards ── */
    .restaurant-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: var(--radius-lg);
        padding: 1.1rem 1.3rem 0.9rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-card);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .restaurant-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-pop);
    }

    .restaurant-card.accepted {
        border-color: rgba(122, 158, 126, 0.5) !important;
        background: rgba(122, 158, 126, 0.06) !important;
    }

    .restaurant-card.rejected {
        border-color: rgba(201, 106, 58, 0.3) !important;
        background: rgba(201, 106, 58, 0.04) !important;
        opacity: 0.6;
    }

    .restaurant-name {
        font-family: var(--serif);
        font-size: 1.25rem;
        font-weight: 400;
        color: var(--ink);
        margin-bottom: 0.2rem;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }

    .restaurant-meta {
        font-size: 0.78rem;
        color: var(--ink-3);
        margin-bottom: 0.5rem;
        letter-spacing: 0.02em;
        font-family: var(--mono);
    }

    .restaurant-blurb {
        font-size: 0.875rem;
        line-height: 1.55;
        color: var(--ink-2);
        margin-top: 0.4rem;
        margin-bottom: 0.5rem;
    }

    /* ── LLM response ── */
    .llm-response {
        background: linear-gradient(180deg, #FFFBF5 0%, #FAF3E8 100%);
        border: 1px solid rgba(201, 162, 39, 0.2);
        border-radius: var(--radius);
        padding: 1.1rem 1.3rem;
        margin: 0.75rem 0 1.25rem;
        font-size: 0.88rem;
        line-height: 1.65;
        color: var(--ink);
        white-space: pre-wrap;
    }

    /* ── Section / results labels ── */
    .section-label {
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--ink-3);
        margin-bottom: 0.5rem;
        font-family: var(--mono);
    }

    .results-label {
        font-family: var(--serif);
        font-size: 1.25rem;
        font-weight: 400;
        color: var(--ink);
        letter-spacing: -0.01em;
        margin-bottom: 1rem;
        margin-top: 0.25rem;
    }

    /* ── Profile tags ── */
    .profile-tag {
        display: inline-block;
        background: var(--tag-terracotta);
        color: var(--terracotta-2);
        border-radius: var(--radius-pill);
        padding: 3px 11px;
        font-size: 0.75rem;
        margin: 2px;
        font-weight: 400;
    }

    .profile-tag.dislike {
        background: var(--tag-ink);
        color: var(--ink-2);
    }

    .profile-tag.cuisine {
        background: var(--tag-sage);
        color: var(--sage-2);
    }

    /* ── Sidebar brand ── */
    .sidebar-brand {
        font-family: var(--serif);
        font-size: 1.4rem;
        font-weight: 400;
        color: var(--ink);
        letter-spacing: -0.01em;
        margin-bottom: 0.1rem;
    }

    .sidebar-brand em {
        color: var(--terracotta);
        font-style: italic;
    }

    .sidebar-subtitle {
        color: var(--ink-3);
        font-size: 0.78rem;
        margin-bottom: 0;
        font-family: var(--mono);
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: var(--card) !important;
        border: 1px solid var(--line) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.75rem !important;
        box-shadow: var(--shadow-card) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--terracotta) !important;
        font-family: var(--serif) !important;
        font-size: 1.5rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--ink-3) !important;
        font-size: 0.72rem !important;
        font-family: var(--mono) !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    /* ── Labels ── */
    label, .stSelectbox label, .stTextInput label, .stTextArea label {
        color: var(--ink-3) !important;
        font-size: 0.75rem !important;
        font-weight: 400 !important;
        font-family: var(--mono) !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
    }

    /* ── Divider ── */
    hr {
        border-color: var(--line) !important;
        margin: 1.25rem 0 !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: var(--terracotta) !important;
    }

    /* ── Caption ── */
    .stCaption {
        color: var(--ink-3) !important;
        font-family: var(--mono) !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.06em !important;
    }

    /* ── Selectbox ── */
    .stSelectbox [data-baseweb="select"] {
        background-color: var(--bg) !important;
        border: 1px solid var(--line-2) !important;
        color: var(--ink) !important;
    }

    /* ── Checkbox ── */
    .stCheckbox label {
        color: var(--ink-2) !important;
        font-size: 0.875rem !important;
        font-family: var(--sans) !important;
        letter-spacing: 0 !important;
        text-transform: none !important;
    }

    /* ── Warning/success ── */
    .stWarning {
        background: var(--tag-gold) !important;
        color: var(--ink) !important;
        border-radius: var(--radius-sm) !important;
    }

    .stSuccess {
        background: var(--tag-sage) !important;
        color: var(--ink) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* ── Markdown ── */
    .stMarkdown h3 {
        font-family: var(--serif) !important;
        color: var(--ink) !important;
        font-size: 1.1rem !important;
        font-weight: 400 !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@st.cache_data
def get_df():
    return load_reviews(path="data/restaurants.csv", max_rows=3000)


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

    for key in ["eat_results", "eat_fsq_results", "eat_llm_response",
                "eat_fsq_response", "cook_response", "cocktail_response"]:
        if key not in st.session_state:
            st.session_state[key] = None

    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-brand"><em>Food AI</em></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">Taste Profile</div>', unsafe_allow_html=True)
        st.markdown("---")

        profile = st.session_state.profile

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accepted", len(profile.get("accepted", [])))
        with col2:
            st.metric("Rejected", len(profile.get("rejected", [])))

        st.markdown("<br>", unsafe_allow_html=True)

        cuisines = profile.get("preferred_cuisines", [])
        if cuisines:
            tags = " ".join([f'<span class="profile-tag cuisine">{c}</span>' for c in cuisines])
            st.markdown(f'<div style="margin-bottom:0.75rem"><div class="section-label">Cuisines</div>{tags}</div>', unsafe_allow_html=True)

        liked = profile.get("liked_foods", [])
        if liked:
            tags = " ".join([f'<span class="profile-tag">{f}</span>' for f in liked])
            st.markdown(f'<div style="margin-bottom:0.75rem"><div class="section-label">Likes</div>{tags}</div>', unsafe_allow_html=True)

        disliked = profile.get("disliked_foods", [])
        if disliked:
            tags = " ".join([f'<span class="profile-tag dislike">{f}</span>' for f in disliked])
            st.markdown(f'<div style="margin-bottom:0.75rem"><div class="section-label">Dislikes</div>{tags}</div>', unsafe_allow_html=True)

        accepted_list = profile.get("accepted", [])
        if accepted_list:
            st.markdown('<div class="section-label">Recently Accepted</div>', unsafe_allow_html=True)
            for name in accepted_list[-3:]:
                st.markdown(f'<div style="font-size:0.8rem;color:var(--sage-2);padding:2px 0">✓ {name}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)

        budget = st.selectbox(
            "Budget",
            ["budget", "moderate", "premium"],
            index=["budget", "moderate", "premium"].index(profile.get("budget", "moderate")),
            key="budget_select",
        )
        occasion = st.text_input("Occasion", value=profile.get("occasion", "casual dinner"), key="occasion_input")

        if st.button("Save Settings"):
            st.session_state.profile["budget"] = budget
            st.session_state.profile["occasion"] = occasion
            save_profile(st.session_state.profile)
            st.success("Saved!")

        st.markdown("---")
        if st.button("Reset Profile", key="reset_profile"):
            if os.path.exists("data/taste_profile.json"):
                os.remove("data/taste_profile.json")
            st.session_state.profile = load_profile()
            st.rerun()


def handle_feedback(restaurant_name, accepted, cuisines=None, foods=None, key=None):
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


def render_restaurant_card(r, source="fsq", idx=0, blurb=""):
    key = f"{source}_{r.get('name', idx)}_{idx}"
    already_accepted = r.get("name", "") in st.session_state.profile.get("accepted", [])
    already_rejected = r.get("name", "") in st.session_state.profile.get("rejected", [])

    card_class = "restaurant-card"
    if already_accepted:
        card_class += " accepted"
    elif already_rejected:
        card_class += " rejected"

    price_str  = PRICE_LABEL.get(r.get("price"), "")
    rating_str = f"{r['rating']} ⭐" if r.get("rating") else ""
    cats       = ", ".join(r.get("categories", [])[:2]) if r.get("categories") else r.get("category", "")
    name       = r.get("name") or r.get("title", "")
    address    = r.get("address", "")

    meta_parts = [p for p in [cats, price_str, rating_str, address] if p]
    meta_str   = "  ·  ".join(meta_parts)

    blurb_html = f'<div class="restaurant-blurb">{html.escape(blurb)}</div>' if blurb else ""

    st.markdown(
        f'<div class="{card_class}">'
        f'<div class="restaurant-name">{html.escape(name)}</div>'
        f'<div class="restaurant-meta">{html.escape(meta_str)}</div>'
        f'{blurb_html}'
        f'</div>',
        unsafe_allow_html=True
    )

    if not already_accepted and not already_rejected:
        col1, col2, _ = st.columns([1, 1, 5])
        with col1:
            st.markdown('<div class="accept-btn">', unsafe_allow_html=True)
            if st.button("✓ Accept", key=f"accept_{key}"):
                handle_feedback(name, accepted=True, cuisines=[cats.split(",")[0].strip()] if cats else None, key=f"fb_{key}")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="reject-btn">', unsafe_allow_html=True)
            if st.button("✗ Pass", key=f"reject_{key}"):
                handle_feedback(name, accepted=False, cuisines=[cats.split(",")[0].strip()] if cats else None, key=f"fb_{key}")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    elif already_accepted:
        st.markdown('<span style="color:var(--sage-2);font-size:0.8rem;font-family:var(--mono)">✓ accepted</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:var(--terracotta);font-size:0.8rem;font-family:var(--mono)">✗ passed</span>', unsafe_allow_html=True)


def render_eat_out_tab(client, df):
    st.markdown('<div class="section-label">Where do you want to eat?</div>', unsafe_allow_html=True)

    with st.form(key="eat_form", enter_to_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("What are you craving?", placeholder="e.g. cozy ramen, cheap tacos, date night Italian")
        with col2:
            zipcode = st.text_input("Zip Code", placeholder="All NYC")
        run_search = st.form_submit_button("Find Restaurants")

    if run_search:
        if not query:
            st.warning("Enter a craving first.")
        else:
            with st.spinner("Finding the best spots..."):
                _, retrieved = rag_recommend(client, query, st.session_state.profile, df, top_k=5)
                st.session_state.eat_results = retrieved

                try:
                    borough = zipcode if zipcode else "New York, NY"
                    _, fsq_restaurants = map_recommend(client, query, st.session_state.profile, borough=borough)
                    st.session_state.eat_fsq_results = fsq_restaurants
                except Exception as e:
                    st.session_state.eat_fsq_results = []
                    st.error(f"Places error: {e}")

                from src.recommend import combined_recommend
                response, selected = combined_recommend(
                    client, query, st.session_state.profile,
                    retrieved,
                    st.session_state.eat_fsq_results or []
                )
                st.session_state.eat_llm_response = response
                st.session_state.eat_fsq_results = selected

    if st.session_state.eat_llm_response:
        fsq_count = len(st.session_state.eat_fsq_results or [])
        st.caption(f"📍 {fsq_count} recommendations · Google Places")
        st.markdown('<div class="results-label">Top picks for you</div>', unsafe_allow_html=True)
        for i, r in enumerate(st.session_state.eat_fsq_results or []):
            render_restaurant_card(r, source="fsq", idx=i, blurb=r.get("blurb", ""))


def render_cook_tab(client):
    st.markdown('<div class="section-label">What are you cooking?</div>', unsafe_allow_html=True)

    with st.form(key="cook_form", enter_to_submit=True):
        craving = st.text_input("What do you feel like making?", placeholder="e.g. something lemony with pasta")
        pantry_input = st.text_area(
            "What's in your fridge/pantry? (comma separated)",
            value=", ".join(st.session_state.profile.get("pantry", [])) if st.session_state.profile.get("pantry") else "",
            placeholder="chicken, garlic, lemon, olive oil, pasta",
            key="cook_pantry",
            height=90,
        )
        run_cook = st.form_submit_button("Generate Recipe")

    if run_cook:
        if not craving:
            st.warning("Tell me what you're craving.")
        else:
            pantry = [p.strip() for p in pantry_input.split(",") if p.strip()]
            st.session_state.profile["pantry"] = pantry
            save_profile(st.session_state.profile)

            from src.recommend import recommend_recipe
            with st.spinner("Crafting your recipe..."):
                response = recommend_recipe(craving, st.session_state.profile)
                st.session_state.cook_response = response

    if st.session_state.cook_response:
        st.markdown('<div class="results-label">Your recipe</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="llm-response">{st.session_state.cook_response}</div>', unsafe_allow_html=True)


def render_cocktail_tab(client):
    st.markdown('<div class="section-label">What are you drinking?</div>', unsafe_allow_html=True)

    with st.form(key="cocktail_form", enter_to_submit=True):
        vibe = st.text_input("What's the vibe?", placeholder="e.g. herbal and refreshing, something smoky, not too sweet")
        bar_input = st.text_area(
            "What's in your bar? (comma separated)",
            value=", ".join(st.session_state.profile.get("bar_inventory", [])) if st.session_state.profile.get("bar_inventory") else "",
            placeholder="gin, lemon juice, honey, rosemary, soda water",
            key="cocktail_bar",
            height=90,
        )
        mocktail = st.checkbox("Mocktail only")
        run_cocktail = st.form_submit_button("Mix Something")

    if run_cocktail:
        if not vibe:
            st.warning("Describe the vibe first.")
        else:
            bar = [b.strip() for b in bar_input.split(",") if b.strip()]
            st.session_state.profile["bar_inventory"] = bar
            save_profile(st.session_state.profile)

            from src.recommend import recommend_cocktail
            full_vibe = vibe + (" (mocktail, no alcohol)" if mocktail else "")
            with st.spinner("Mixing..."):
                response = recommend_cocktail(full_vibe, st.session_state.profile)
                st.session_state.cocktail_response = response

    if st.session_state.cocktail_response:
        st.markdown('<div class="results-label">Your drink</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="llm-response">{st.session_state.cocktail_response}</div>', unsafe_allow_html=True)


def main():
    init_session()
    render_sidebar()

    st.markdown('<div class="foodai-header">What are you <em>hungry</em> for?</div>', unsafe_allow_html=True)
    st.markdown('<div class="foodai-subtitle">Tell me what you want. I\'ll figure out the rest.</div>', unsafe_allow_html=True)

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