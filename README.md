# Food AI Companion

Food AI Companion is an AI-powered recommendation system for food decisions. The full project vision, based on the proposal for Advanced Data Science, is a shared platform that helps users decide where to eat out, what to cook at home, and what to drink by using a persistent taste profile plus external grounding data.

The current repository implements the **Milestone 2 Eat Out prototype** alongside a working **Streamlit application** that extends the system into a fully interactive multi-mode experience.

## Team

- Bilal Naseer
- Hoerim Kim
- Owen Nie

Course: Advanced Data Science

## Project Vision

The full proposed system includes three modes:

- Eat Out: personalized restaurant recommendations grounded in external restaurant data
- Cook at Home: recipe generation from available ingredients
- Drink/Cocktail: drink suggestions with flavor-aware substitution logic

All three modes share a persistent taste profile that captures preferences such as cuisine, liked and disliked foods, budget sensitivity, and occasion context. The system is designed to improve personalization over time as users accept and reject recommendations.

## Milestone 3 Bilal Update

- Added a modular Cook at Home backend in `src/cook_mode.py` that uses pantry ingredients, cuisine preferences, disliked foods/restrictions, budget, time, and occasion context to generate transparent recipe ideas. Because no recipe dataset is present, this mode is framed as LLM + taste-profile generation rather than recipe RAG.
- Connected Cook at Home into the shared recommendation/app flow through `src/recommend.py` and `app.py`, including pantry persistence, recipe remixing, and per-recipe save/pass feedback behavior for multi-recipe outputs.
- Added Cook Mode demo artifacts through `scripts/run_cook_mode_demo.py`, producing `results/cook_mode_demo_preview.md` and optional generated outputs for the final demo/report.
- Added a personalization vs. filter-bubble analysis in `src/filter_bubble.py` and `scripts/run_filter_bubble_analysis.py`, with overlap, novelty, diversity, entropy, alignment, and filter-bubble metrics.
- Added a Milestone 3 evaluation harness in `src/evaluation.py` and `scripts/run_milestone3_evaluation.py`, producing deterministic system-side metrics plus a blank human-scoring template so no evaluation scores are fabricated.
- Added a Reddit/PRAW grounding artifact with Hoerim's validation support in `src/reddit_grounding.py` and `scripts/run_reddit_grounding_analysis.py`, producing report-ready community-signal outputs without changing the working Streamlit app.
- Added compact recommendation trace/explainability panels in `app.py` that show each result's source and grounding path, evidence used, taste-profile match, inventory or pantry match, live-photo/hour availability, and data limitations for final-demo transparency.
- Added an explicit-cuisine guard for Eat Out fallback search so current requests such as Pakistani/South Asian food are prioritized over stale taste-profile preferences when live Google Places results are unavailable.
- Polished the Streamlit app integration for final demo stability, including Cook/Cocktail tab persistence after remix/save/pass actions and a clear fallback warning when Google Places photos are unavailable without `GOOGLE_PLACES_API_KEY`.

## Milestone 3 Hoerim Update
- Improved `src/retrieval.py` to incorporate learned `cuisine_scores` from the taste profile into the embedding query and reranking stage, so retrieval results become more personalized as users provide accept/reject feedback.
- Fixed a filtering bug in `src/retrieval.py` where strict cuisine and food filters could return zero results. Replaced with a three-stage fallback strategy (strict → cuisine-only → no filter) to ensure recommendations are always returned.
- Expanded `CATEGORY_FOOD_HINTS` in `src/retrieval.py` to cover Chinese, Mexican, Indian, Thai, and American cuisines, reducing food-category mismatch penalties for a broader range of queries.
- Fixed a bug in `src/filter_bubble.py` where `filter_bubble_index` returned a misleading non-zero score for first-time users with no history. The function now correctly returns 0.0 when no prior session exists.
- Added `weighted_profile_alignment` to `src/filter_bubble.py`, a continuous alignment metric using learned `cuisine_scores` and `food_scores` that improves as the user gives feedback, complementing the existing binary alignment ratio.
- Extended `diversity_rerank` in `src/filter_bubble.py` to accept an optional `profile` argument, giving a small tiebreaking bonus to cuisine-preferred restaurants during diversity reranking.
- Helped validate and debug the Milestone 3 personalization/evaluation logic around retrieval fallback behavior, profile-weighted alignment, filter-bubble measurement, and the Reddit/PRAW grounding artifact added with Bilal.

## Milestone 3 Owen Update

- Added agentic Food Companion chat panel that can infer whether the user wants Eat Out, Cook at Home, Cocktail, or remix actions; has context of active tabs and current search results; switch tabs if user asks for something not on current tab; ask for missing zip code, pantry items, or bar inventory when needed; confirm searches before running them; ask clarifying questions when intent is ambiguous; and route confirmed actions into the same recommendation state used by the main UI.
- Polished the chat experience with preserved message formatting, message autoscroll behavior, search/remix follow-up messages, and support for running Eat Out, Cook, and Cocktail searches from the companion.
- Added RAG to the Food Companion chat layer, grounding conversational responses in the 10k-row NYC restaurant dataset by retrieving the 4 most relevant restaurants via embedding similarity on each turn.
- Implemented result cards across Eat Out, Cook at Home, and Cocktail modes so recommendations use a consistent card layout with clearer titles, strong visual hierarchy, action buttons, closed-by-default recipe/details expanders, and restaurant opening/closing time information.
- Cocktail result cards with left-edge image rails, CocktailDB/photo fallback handling, formatted full recipes, per-card save/pass/remix controls, and image-muted loading behavior that matches the Eat Out interaction pattern.
- Cook at Home cards with drink-mode-sized dish titles, better spacing around full recipe expanders, cleaned recipe formatting, and targeted remix behavior so a single remixed dish stays in the same card position while other results remain stable.
- Added sidebar analytics/profile visibility in `app.py`, including a clearer taste-profile summary and mode-level feedback signals so saved, passed, pantry, bar, and preference data are easier to inspect.

## Current Milestone 2 Scope

This repository implements the **Eat Out** mode prototype for evaluation, plus a full interactive Streamlit app covering all three modes.

What is implemented:

- Restaurant dataset loading and preprocessing
- Taste-profile-based restaurant recommendation
- Embedding-based retrieval over restaurant records
- A RAG pipeline that injects retrieved restaurant evidence into the LLM prompt
- A four-way comparison workflow: baseline, profile-aware, RAG (static), and RAG (live Google Places)
- Live restaurant lookup via Google Places API (Text Search New)
- LLM-powered restaurant selection and blurb generation from live results
- A persistent taste profile stored as JSON, updated from user accept/reject feedback
- A Streamlit UI covering Eat Out, Cook at Home, and Cocktail modes
- A supplemental Reddit/PRAW grounding artifact for community-discussion signal analysis
- Saved evaluation outputs for multiple sample queries

What is not yet implemented:

- LangGraph orchestration across all modes
- Recipe dataset grounding for Cook at Home
- Full live Reddit-to-RAG integration inside the Streamlit app

## Recent Improvements

To address recent TA feedback and improve demo credibility, the codebase now includes:

- stronger dataset cleaning for noisy and inconsistent restaurant rows
- category repair using title and review evidence
- better `popular_food` cleanup for obviously incorrect values
- restaurant-level consolidation of duplicate rows
- stricter reranking and filtering to reduce bad cuisine matches
- a more conservative grounded recommendation prompt for the RAG stage
- Google Places API integration as a fourth pipeline mode with live NYC restaurant data
- LLM-driven restaurant selection from live results with per-restaurant blurbs
- persistent taste profile with accept/reject feedback loop
- full Streamlit UI with three modes, sidebar profile display, and real-time updates

## Repository Structure
```text
food-ai-companion-main/
├── app.py
├── main.py
├── scripts/
│   ├── run_milestone3_evaluation.py
│   ├── run_filter_bubble_analysis.py
│   ├── run_reddit_grounding_analysis.py
│   └── run_cook_mode_demo.py
├── data/
│   ├── restaurants.csv
│   └── restaurant_embeddings.npz
├── results/
│   └── milestone2_outputs.txt
├── src/
│   ├── data_loader.py
│   ├── cook_mode.py
│   ├── evaluation.py
│   ├── reddit_grounding.py
│   ├── recommend.py
│   ├── retrieval.py
│   ├── places.py
│   ├── taste_profile.py
│   └── requirements.txt
└── task_breakdown.txt
```

## Installation

### 1. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r src/requirements.txt
```

### 3. Add your API keys

Create a local `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_key_here
GOOGLE_PLACES_API_KEY=your_google_places_key_here
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=food-ai-companion-milestone3 by u/your_username
```

The Reddit keys are only needed for the optional Reddit grounding artifact. The app and other evaluation scripts still run without them.

## How to Run

### Milestone 2 evaluation pipeline
```bash
python3 main.py
```

This will:

- load the cleaned restaurant dataset
- run the baseline LLM pipeline
- run the taste-profile pipeline
- run the taste-profile + RAG pipeline (static)
- run the taste-profile + RAG pipeline (live Google Places)
- save outputs to `results/milestone2_outputs.txt`

### Interactive Streamlit app
```bash
streamlit run app.py
```

This launches the full multi-mode application in your browser.

### Cook at Home Milestone 3 demo
```bash
python3 scripts/run_cook_mode_demo.py
```

This creates `results/cook_mode_demo_preview.md`, a deterministic preview of the Cook mode contexts and prompt contract. To generate real LLM recipe outputs for the report/demo, run:

```bash
python3 scripts/run_cook_mode_demo.py --generate
```

### Personalization/filter-bubble analysis
```bash
python3 scripts/run_filter_bubble_analysis.py
```

This creates `results/filter_bubble_metrics.csv` and `results/filter_bubble_analysis.md`. The metrics are deterministic metadata-based estimates for Milestone 3 discussion; they are not human evaluation scores.

### Milestone 3 evaluation artifacts
```bash
python3 scripts/run_milestone3_evaluation.py
```

This creates:

- `results/milestone3_evaluation_metrics.csv`
- `results/milestone3_human_eval_template.csv`
- `results/milestone3_evaluation_summary.md`

The metrics are deterministic system-side checks over local restaurant metadata. The human-evaluation template is intentionally blank so the team can score actual generated outputs from `main.py` or the Streamlit demo without fabricating evaluator scores.

### Reddit grounding artifact
```bash
python3 scripts/run_reddit_grounding_analysis.py
```

This creates:

- `results/reddit_grounding_posts.csv`
- `results/reddit_grounding_summary.md`

If Reddit credentials are not configured, the script still writes a setup/preview summary and an empty CSV with headers. It does not fabricate Reddit posts or scores. With valid Reddit credentials, it collects public Reddit discussion signals, matches restaurant names against the local dataset, and summarizes cuisine/restaurant mention patterns for the Milestone 3 report.

## Pipeline Overview

### 1. Data Loading and Cleaning

The restaurant dataset is loaded in `src/data_loader.py`. The loader normalizes inconsistent column names, cleans noisy text fields, repairs category mismatches using title and review evidence, cleans low-quality `popular_food` values, groups duplicate rows into cleaner restaurant-level records, and builds `combined_text` used for retrieval embeddings.

### 2. Taste Profile

The persistent taste profile is managed in `src/taste_profile.py`. It stores cuisine affinities, flavor scores, liked and disliked foods, budget sensitivity, pantry, bar inventory, and occasion context. The profile is saved as JSON and updated incrementally each time a user accepts or rejects a recommendation. It is injected into every LLM prompt at runtime.

### 3. Retrieval Layer

The retrieval pipeline in `src/retrieval.py` builds embeddings with `text-embedding-3-small`, caches them to avoid recomputation, embeds a query enriched with the taste profile, ranks restaurants using cosine similarity, and reranks using cuisine match, liked/disliked foods, budget cues, and ordering preference.

### 4. Live Restaurant Lookup

`src/places.py` integrates the Google Places API (New) Text Search endpoint. It uses field masking to stay within the Enterprise SKU tier, fetches ratings, price level, opening hours, and up to 3 reviews per result, and restricts review calls to the top 3 results to minimize API usage.

### 5. Recommendation Layer

`src/recommend.py` provides five output modes:

- `baseline_recommend`: generic LLM recommendation without retrieved evidence
- `profile_recommend`: LLM recommendation conditioned on a taste profile
- `rag_recommend`: LLM recommendation grounded in retrieved restaurant records
- `map_recommend`: LLM recommendation grounded in live Google Places results
- `combined_recommend`: LLM selects the best 5 from both static and live sources, generating a blurb per restaurant

## Current Evaluation Setup

The Milestone 2 pipeline evaluates multiple sample queries across four settings and includes a profile comparison experiment showing how different user preferences change outputs for the same query. Saved outputs include generated recommendations, retrieved restaurants, retrieval scores, and qualitative comparison across settings.

## Task Breakdown for Milestone 2

### Bilal Naseer

- set up the overall project structure and execution flow
- defined the taste profile schema used by the recommendation pipeline
- integrated the OpenAI LLM pipeline
- implemented the three comparison settings in `main.py`
- connected inputs, profile context, and retrieved evidence into end-to-end recommendation generation
- strengthened the system by improving the data-cleaning and reranking workflow

### Hoerim Kim

- finalized and restructured the NYC restaurant dataset
- improved restaurant text representation for retrieval
- built and refined the embedding-based retrieval pipeline
- added embedding cache validation
- improved grounding by passing richer restaurant evidence to the LLM

### Owen Nie

- integrated Google Places API as a fourth pipeline mode with live NYC restaurant data
- implemented `combined_recommend` which uses the LLM to select and rank the best results across static and live sources, with per-restaurant descriptions
- built persistent taste profile (`taste_profile.py`) with JSON persistence and incremental updates from user feedback
- implemented `map_recommend`, `recommend_recipe`, and `recommend_cocktail` in `recommend.py`
- built Streamlit application (`app.py`) containing the Eat Out Mode prototype and outline for Cook at Home and Cocktail modes
## Limitations

- evaluation is primarily qualitative and demo-oriented
- the restaurant dataset is static rather than fully live for the RAG pipeline
- some proposed external data integrations are planned but not yet complete

## Future Work

- improve evaluation with clearer relevance, diversity, and consistency metrics
- connect the Reddit grounding artifact into the live RAG/ranking layer after more validation
- extend the taste profile into a richer memory component with longer-term learning
- connect all modes into one shared LangGraph orchestration workflow

## Notes

- The active milestone is intentionally narrower than the full project proposal.
- If you change the dataset or retrieval logic substantially, delete `data/restaurant_embeddings.npz` and rerun `main.py` to regenerate embeddings.
- The `.env` file should never be committed to the repository.
