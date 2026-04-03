# Food AI Companion

Food AI Companion is an AI-powered recommendation system for food decisions. The full project vision, based on the proposal for Advanced Data Science, is a shared platform that helps users decide where to eat out, what to cook at home, and what to drink by using a persistent taste profile plus external grounding data.

The current repository implements the **Milestone 2 Eat Out prototype**. This milestone focuses on restaurant recommendation in New York City and compares three settings:

- Baseline LLM
- LLM + taste profile
- LLM + taste profile + retrieval-augmented generation (RAG)

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

All three modes are intended to share a taste profile that captures preferences such as cuisine, liked and disliked foods, budget sensitivity, and context. As described in the proposal, the longer-term goal is a recommendation system that improves personalization while staying grounded in real-world data.

## Current Milestone 2 Scope

This repository currently implements the **Eat Out** mode prototype only.

What is implemented:

- Restaurant dataset loading and preprocessing
- Taste-profile-based restaurant recommendation
- Embedding-based retrieval over restaurant records
- A RAG pipeline that injects retrieved restaurant evidence into the LLM prompt
- A comparison workflow across baseline, profile-aware, and RAG-based recommendations
- Saved evaluation outputs for multiple sample queries

What is not yet implemented in this repository:

- Cook at Home mode
- Drink/Cocktail mode
- Persistent user memory that updates from accepted/rejected recommendations
- LangGraph orchestration across all modes
- Reddit/PRAW grounding pipeline
- Recipe dataset grounding

## Recent Improvements

To address recent TA feedback and improve demo credibility, the current codebase now includes:

- stronger dataset cleaning for noisy and inconsistent restaurant rows
- category repair using title and review evidence
- better `popular_food` cleanup for obviously incorrect values
- restaurant-level consolidation of duplicate rows
- stricter reranking and filtering to reduce bad cuisine matches
- a more conservative grounded recommendation prompt for the RAG stage

These changes were added to reduce visibly incorrect outputs such as Korean BBQ being labeled as pizza or Italian restaurants surfacing with unrelated foods like fried rice.

## Repository Structure

```text
food-ai-companion-main/
├── data/
│   ├── restaurants.csv
│   └── restaurant_embeddings.npz
├── results/
│   └── milestone2_outputs.txt
├── src/
│   ├── data_loader.py
│   ├── recommend.py
│   ├── retrieval.py
│   ├── foursquare_places.py
│   └── requirements.txt
├── main.py
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

### 3. Add your OpenAI API key

Create a local `.env` file in the project root:

```bash
OPENAI_API_KEY=your_api_key_here
```

## How to Run

Run the full Milestone 2 comparison pipeline with:

```bash
python3 main.py
```

This will:

- load the cleaned restaurant dataset
- run the baseline LLM pipeline
- run the taste-profile pipeline
- run the taste-profile + RAG pipeline
- save outputs to [results/milestone2_outputs.txt](https://github.com/bilalnaseer7/food-ai-companion/blob/bn's-new-branch/results/milestone2_outputs.txt)

## Pipeline Overview

### 1. Data Loading and Cleaning

The restaurant dataset is loaded in [src/data_loader.py](/Users/bilalnaseer/Documents/Spring%20'26/Intro%20to%20LLM/Project_G17/food-ai-companion-main/src/data_loader.py). The loader:

- normalizes inconsistent column names
- cleans noisy text fields
- repairs obvious category mismatches using title and review evidence
- cleans low-quality `popular_food` values
- groups duplicate rows into cleaner restaurant-level records
- builds `combined_text` used for retrieval embeddings

### 2. Taste Profile

The current taste profile includes:

- preferred cuisines
- liked foods
- disliked foods
- budget
- online ordering preference
- occasion
- city

This profile is injected into prompt construction and retrieval scoring.

### 3. Retrieval Layer

The retrieval pipeline in [src/retrieval.py](/Users/bilalnaseer/Documents/Spring%20'26/Intro%20to%20LLM/Project_G17/food-ai-companion-main/src/retrieval.py):

- builds embeddings with `text-embedding-3-small`
- caches embeddings to avoid recomputation
- embeds a query enriched with the taste profile
- ranks restaurants using cosine similarity
- reranks using cuisine match, liked/disliked foods, budget cues, and ordering preference
- filters obviously weak or contradictory matches

### 4. Recommendation Layer

The recommendation pipeline in [src/recommend.py](/Users/bilalnaseer/Documents/Spring%20'26/Intro%20to%20LLM/Project_G17/food-ai-companion-main/src/recommend.py) provides three output settings:

- `baseline_recommend`: generic LLM recommendation without retrieved evidence
- `profile_recommend`: LLM recommendation conditioned on a taste profile
- `rag_recommend`: LLM recommendation grounded in retrieved restaurant records

## Current Evaluation Setup

The Milestone 2 pipeline currently evaluates multiple sample queries and compares:

- baseline LLM
- LLM + taste profile
- LLM + taste profile + RAG

It also includes a profile comparison experiment to show how different user preferences change outputs for the same query.

The saved outputs include:

- generated recommendations
- retrieved restaurants
- retrieval scores
- qualitative comparison across settings

See [results/milestone2_outputs.txt](/Users/bilalnaseer/Documents/Spring%20'26/Intro%20to%20LLM/Project_G17/food-ai-companion-main/results/milestone2_outputs.txt).

## Task Breakdown for Milestone 2

This breakdown is aligned with the proposal and the current milestone plan.

### Bilal Naseer

- set up the overall project structure and execution flow
- defined the taste profile schema used by the recommendation pipeline
- integrated the OpenAI LLM pipeline
- implemented the three comparison settings in `main.py`
- connected inputs, profile context, and retrieved evidence into end-to-end recommendation generation
- recently strengthened the Milestone 2 system further by improving the data-cleaning and reranking workflow used for demo-facing outputs

### Hoerim Kim

- finalized and restructured the NYC restaurant dataset
- improved restaurant text representation for retrieval
- built and refined the embedding-based retrieval pipeline
- added embedding cache validation
- improved grounding by passing richer restaurant evidence to the LLM

### Owen Nie

- focused on ranking and evaluation logic
- planned mismatch filtering and stronger profile-aware ranking
- prepared query-based comparison goals for baseline vs profile vs RAG evaluation

## Limitations

This repository is still a prototype. Current limitations include:

- only the Eat Out mode is implemented
- evaluation is primarily qualitative and demo-oriented
- the restaurant dataset is static rather than fully live
- some proposed external data integrations are planned but not yet complete

## Future Work

Based on the original proposal, the next major steps are:

- improve evaluation with clearer relevance, diversity, and consistency analysis
- add stronger grounding from external food discussion and recipe sources
- implement Cook at Home mode
- implement Drink/Cocktail mode
- extend the taste profile into a persistent memory component
- connect the full multi-mode system into one shared workflow

## Notes

- The active milestone is intentionally narrower than the full project proposal.
- The README describes both the broader project vision and the current implemented milestone so the documentation stays accurate.
- If you change the dataset or retrieval logic substantially, regenerate outputs by rerunning `main.py`.
