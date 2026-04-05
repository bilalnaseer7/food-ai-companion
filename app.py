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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    :root {
        --dark-bg-1: #00202b;
        --dark-bg-2: #0a2f3d;
        --light-blue-1: #00B6E6;
        --light-blue-2: rgba(0, 182, 230, 0.15);
        --light-blue-3: rgba(0, 182, 230, 0.3);
        --accent-red-1: #eb123f;
        --accent-red-2: rgba(235, 18, 63, 0.15);
        --accent-red-3: rgba(235, 18, 63, 0.3);
        --white: #FFFFFF;
        --white-dim: rgba(255,255,255,0.7);
        --light-grey-1: #f5f5f5;
        --card-bg: rgba(255,255,255,0.05);
        --card-border: rgba(255,255,255,0.1);
    }

    html, body, .stApp {
        background-color: var(--dark-bg-1) !important;
        color: var(--white) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 300 !important;
        min-height: 100vh !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: var(--dark-bg-1) !important;
        min-height: 100vh !important;
    }

    .main .block-container {
        background: transparent !important;
        padding-top: 2rem !important;
        padding-bottom: 6rem !important;
        max-width: 1200px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--dark-bg-2) !important;
        border-right: 1px solid var(--card-border) !important;
    }

    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] button,
    [data-testid="collapsedControl"] svg {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
    }

    /* Header */
    .foodai-header {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        color: var(--white) !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem;
    }

    .foodai-subtitle {
        color: var(--white-dim) !important;
        font-size: 0.9rem !important;
        font-weight: 300 !important;
        margin-bottom: 1.5rem !important;
    }

    /* Mode tabs */
    [data-testid="stTabs"] [role="tablist"] {
        background: var(--card-bg) !important;
        border-radius: 10px !important;
        padding: 4px !important;
        border: 1px solid var(--card-border) !important;
    }

    [data-testid="stTabs"] [role="tab"] {
        color: var(--white-dim) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 400 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: var(--light-blue-1) !important;
        color: var(--dark-bg-1) !important;
        font-weight: 600 !important;
    }

    /* Text inputs */
    .stTextInput > div > div, .stSelectbox > div > div, .stTextArea > div > div {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 8px !important;
        color: var(--white) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .stTextInput input, .stTextArea textarea {
        color: var(--white) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .stSelectbox [data-baseweb="select"] {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        color: var(--white) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--light-blue-1), #0095c7) !important;
        color: var(--dark-bg-1) !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        padding: 0.5rem 1.5rem !important;
        transition: opacity 0.2s ease !important;
    }

    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    .stButton > button[kind="secondary"] {
        background: var(--card-bg) !important;
        color: var(--white) !important;
        border: 1px solid var(--card-border) !important;
    }

    /* Accept/reject buttons */
    .accept-btn > button {
        background: linear-gradient(135deg, #00c851, #007a32) !important;
        color: var(--white) !important;
        font-size: 0.8rem !important;
        padding: 0.3rem 0.8rem !important;
    }

    .reject-btn > button {
        background: linear-gradient(135deg, var(--accent-red-1), #a00020) !important;
        color: var(--white) !important;
        font-size: 0.8rem !important;
        padding: 0.3rem 0.8rem !important;
    }

    /* Restaurant cards */
    .restaurant-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s ease;
    }

    .restaurant-card:hover {
        border-color: var(--light-blue-3);
    }

    .restaurant-card.accepted {
        border-color: #00c851 !important;
        background: rgba(0, 200, 81, 0.08) !important;
    }

    .restaurant-card.rejected {
        border-color: var(--accent-red-3) !important;
        background: var(--accent-red-2) !important;
        opacity: 0.6;
    }

    .restaurant-name {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1.05rem;
        color: var(--white);
        margin-bottom: 0.2rem;
    }

    .restaurant-meta {
        font-size: 0.8rem;
        color: var(--white-dim);
        margin-bottom: 0.4rem;
    }

    .restaurant-tip {
        font-size: 0.82rem;
        color: var(--light-blue-1);
        font-style: italic;
        margin-top: 0.3rem;
    }

    /* LLM response box */
    .llm-response {
        background: var(--light-blue-2);
        border: 1px solid var(--light-blue-3);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Profile card */
    .profile-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .profile-card h4 {
        font-family: '', serif;
        color: var(--light-blue-1);
        margin-bottom: 0.75rem;
        font-size: 1rem;
    }

    .profile-tag {
        display: inline-block;
        background: var(--light-blue-2);
        border: 1px solid var(--light-blue-3);
        color: var(--light-blue-1);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px;
    }

    .llm-response {
        background: var(--light-blue-2);
        border: 1px solid var(--light-blue-3);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }

    .profile-tag.dislike {
        background: var(--accent-red-2);
        border-color: var(--accent-red-3);
        color: var(--accent-red-1);
    }

    /* Section labels */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--white-dim);
        margin-bottom: 0.5rem;
    }

    /* Divider */
    hr {
        border-color: var(--card-border) !important;
        margin: 1.5rem 0 !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--light-blue-1) !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--light-blue-1) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--white-dim) !important;
        font-size: 0.75rem !important;
    }

    /* Markdown headers inside app */
    .stMarkdown h3 {
        font-family: 'DM Sans', sans-serif !important;
        color: var(--white) !important;
        font-size: 1.1rem !important;
    }

    /* Labels */
    label, .stSelectbox label, .stTextInput label {
        color: var(--white-dim) !important;
        font-size: 0.82rem !important;
        font-weight: 400 !important;
    }

    html, body {
        overscroll-behavior: none !important;
        -webkit-overflow-scrolling: auto !important;
    }

    .stApp {
        overscroll-behavior: none !important;
    }

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

    if "eat_results" not in st.session_state:
        st.session_state.eat_results = None
    if "eat_fsq_results" not in st.session_state:
        st.session_state.eat_fsq_results = None
    if "eat_llm_response" not in st.session_state:
        st.session_state.eat_llm_response = None
    if "eat_fsq_response" not in st.session_state:
        st.session_state.eat_fsq_response = None
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()
    if "cook_response" not in st.session_state:
        st.session_state.cook_response = None
    if "cocktail_response" not in st.session_state:
        st.session_state.cocktail_response = None


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="foodai-header">Food AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="foodai-subtitle">Your AI food companion</div>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown('<div class="section-label">Taste Profile</div>', unsafe_allow_html=True)

        profile = st.session_state.profile

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accepted", len(profile.get("accepted", [])))
        with col2:
            st.metric("Rejected", len(profile.get("rejected", [])))

        st.markdown("<br>", unsafe_allow_html=True)

        cuisines = profile.get("preferred_cuisines", [])
        if cuisines:
            tags = " ".join([f'<span class="profile-tag">{c}</span>' for c in cuisines])
            st.markdown(f'<div style="margin-bottom:0.5rem"><div class="section-label">Cuisines</div>{tags}</div>', unsafe_allow_html=True)

        liked = profile.get("liked_foods", [])
        if liked:
            tags = " ".join([f'<span class="profile-tag">{f}</span>' for f in liked])
            st.markdown(f'<div style="margin-bottom:0.5rem"><div class="section-label">Likes</div>{tags}</div>', unsafe_allow_html=True)

        disliked = profile.get("disliked_foods", [])
        if disliked:
            tags = " ".join([f'<span class="profile-tag dislike">{f}</span>' for f in disliked])
            st.markdown(f'<div style="margin-bottom:0.5rem"><div class="section-label">Dislikes</div>{tags}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-label">Profile Settings</div>', unsafe_allow_html=True)

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
            os.remove("data/taste_profile.json") if os.path.exists("data/taste_profile.json") else None
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

    price_str  = PRICE_LABEL.get(r["price"], "")
    rating_str = f"{r['rating']} ⭐" if r.get('rating') else ""
    open_str   = " · open now" if r["open_now"] else ""
    cats       = ", ".join(r.get("categories", [])[:2]) if r.get("categories") else r.get("category", "")
    name       = r.get("name") or r.get("title", "")
    address    = r.get("address", "")

    meta_parts = [p for p in [cats, price_str, rating_str, address] if p]
    meta_str   = " · ".join(meta_parts)

    blurb_html = f'<div class="restaurant-blurb">{blurb}</div>' if blurb else ""

    st.markdown(
        f'<div class="{card_class}"><div class="restaurant-name">{name}</div>'
        f'<div class="restaurant-meta">{meta_str}</div>{blurb_html}</div>',
        unsafe_allow_html=True
    )

    if not already_accepted and not already_rejected:
        col1, col2, _ = st.columns([1, 1, 4])
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
        st.markdown('<span style="color:#00c851;font-size:0.8rem">✓ Accepted</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#eb123f;font-size:0.8rem">✗ Passed</span>', unsafe_allow_html=True)
        
def render_eat_out_tab(client, df):
    st.markdown('<div class="section-label">Eat Out / Order In</div>', unsafe_allow_html=True)

    with st.form(key="eat_form", enter_to_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("What are you craving?", placeholder="e.g. cozy ramen, cheap tacos, date night Italian")
        with col2:
            zipcode = st.text_input("Zip Code", placeholder="Searching All NYC")
        run_search = st.form_submit_button("Find Restaurants")

    if run_search:
        if not query:
            st.warning("Enter a craving first.")
        else:
            with st.spinner("Searching..."):
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
        csv_count = len(st.session_state.eat_results or [])
        st.caption(f"📊 {csv_count} from dataset · 📍 {fsq_count} live from Places")
        st.markdown(f'<div class="llm-response">{st.session_state.eat_llm_response}</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
        for i, r in enumerate(st.session_state.eat_fsq_results or []):
            render_restaurant_card(r, source="fsq", idx=i)


def render_cook_tab(client):
    st.markdown('<div class="section-label">Cook at Home</div>', unsafe_allow_html=True)

    craving = st.text_input("What do you feel like making?", placeholder="e.g. something lemony with pasta", key="cook_craving")

    pantry_input = st.text_area(
        "What's in your fridge/pantry? (comma separated)",
        value=", ".join(st.session_state.profile.get("pantry", [])) if st.session_state.profile.get("pantry") else "",
        placeholder="chicken, garlic, lemon, olive oil, pasta",
        key="cook_pantry",
        height=80,
    )

    if st.button("Generate Recipe", key="cook_search"):
        if not craving:
            st.warning("Tell me what you're craving.")
            return

        pantry = [p.strip() for p in pantry_input.split(",") if p.strip()]
        st.session_state.profile["pantry"] = pantry
        save_profile(st.session_state.profile)

        from src.recommend import recommend_recipe
        with st.spinner("Crafting your recipe..."):
            response = recommend_recipe(craving, st.session_state.profile)
            st.session_state.cook_response = response

    if st.session_state.cook_response:
        st.markdown(f'<div class="llm-response">{st.session_state.cook_response}</div>', unsafe_allow_html=True)


def render_cocktail_tab(client):
    st.markdown('<div class="section-label">Cocktails & Drinks</div>', unsafe_allow_html=True)

    vibe = st.text_input("What's the vibe?", placeholder="e.g. herbal and refreshing, something smoky, not too sweet", key="cocktail_vibe")

    bar_input = st.text_area(
        "What's in your bar? (comma separated)",
        value=", ".join(st.session_state.profile.get("bar_inventory", [])) if st.session_state.profile.get("bar_inventory") else "",
        placeholder="gin, lemon juice, honey, rosemary, soda water",
        key="cocktail_bar",
        height=80,
    )

    mocktail = st.checkbox("Mocktail mode", key="mocktail_mode")

    if st.button("Mix Something", key="cocktail_search"):
        if not vibe:
            st.warning("Describe the vibe first.")
            return

        bar = [b.strip() for b in bar_input.split(",") if b.strip()]
        st.session_state.profile["bar_inventory"] = bar
        save_profile(st.session_state.profile)

        from src.recommend import recommend_cocktail
        full_vibe = vibe + (" (mocktail, no alcohol)" if mocktail else "")
        with st.spinner("Mixing..."):
            response = recommend_cocktail(full_vibe, st.session_state.profile)
            st.session_state.cocktail_response = response

    if st.session_state.cocktail_response:
        st.markdown(f'<div class="llm-response">{st.session_state.cocktail_response}</div>', unsafe_allow_html=True)


def main():
    init_session()
    render_sidebar()

    st.markdown('<div class="foodai-header">Food AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="foodai-subtitle">Tell me what you want. I\'ll figure out the rest.</div>', unsafe_allow_html=True)

    client = get_client()
    df = get_df()

    tab_eat, tab_cook, tab_cocktail = st.tabs(["🍜 Eat Out", "🍳 Cook", "🍹 Cocktails"])

    with tab_eat:
        render_eat_out_tab(client, df)

    with tab_cook:
        render_cook_tab(client)

    with tab_cocktail:
        render_cocktail_tab(client)


if __name__ == "__main__":
    main()

