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
- Saved evaluation outputs for multiple sample queries

What is not yet implemented:

- Reddit/PRAW grounding pipeline
- LangGraph orchestration across all modes
- Recipe dataset grounding for Cook at Home

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
├── data/
│   ├── restaurants.csv
│   └── restaurant_embeddings.npz
├── results/
│   └── milestone2_outputs.txt
├── src/
│   ├── data_loader.py
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
```

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
- add grounding from Reddit food discussions via PRAW
- extend the taste profile into a richer memory component with longer-term learning
- connect all modes into one shared LangGraph orchestration workflow

## Notes

- The active milestone is intentionally narrower than the full project proposal.
- If you change the dataset or retrieval logic substantially, delete `data/restaurant_embeddings.npz` and rerun `main.py` to regenerate embeddings.
- The `.env` file should never be committed to the repository.
