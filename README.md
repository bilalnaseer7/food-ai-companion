# food-ai-companion

[Bilal] - Changelog

Set up the GitHub repository and overall project structure (data/, src/, main.py, etc.) for the Milestone 2 prototype.

Added and updated the restaurant review dataset used for the Eat Out recommendation mode (aligned with project proposal focus on restaurant-based retrieval).

Built the core data loading pipeline to correctly read and process the review dataset (data_loader.py), including fixing column mapping issues based on TA feedback.

Implemented the baseline LLM recommendation pipeline using OpenAI API to generate restaurant suggestions from a user query.

Added a structured taste profile layer (preferences, budget, disliked foods, etc.) and integrated it into the recommendation pipeline to personalize outputs.

Implemented a retrieval (RAG) pipeline: 

retrieves relevant restaurants from dataset

passes retrieved context to the LLM

generates grounded recommendations based on real review data

Combined all three stages into a single workflow for comparison:

baseline LLM

LLM + taste profile

LLM + taste profile + RAG

Built end-to-end execution in main.py:

loads dataset

runs all three pipelines

saves outputs to results/milestone2_outputs.txt for evaluation

Integrated .env setup for secure API key handling and ensured it is excluded via .gitignore.

🧱 Data Layer Improvements (Layer 1)

added column alias handling so small dataset header variations do not break the loader

added text cleaning helpers for safer preprocessing

normalized online_order values into Yes / No / Unknown

parsed category and popular_food into structured list-style fields

removed old row-level noisy aggregation

introduced restaurant-level aggregation

limited to top 3 distinct review snippets per restaurant

aggregated restaurant metadata into a cleaner restaurant profile

rebuilt combined_text for stronger semantic representation

sorted restaurants by review count before truncation


🔍 Retrieval + Embeddings (Layer 2)

integrated embedding-based semantic retrieval (vector search)

preserved embedding cache + vector architecture

replaced row-level logic with restaurant-level retrieval

improved query construction using user profile + city context

added safer phrase matching utilities

implemented hybrid reranking:

cuisine matching boost

liked food boost

disliked food penalties

budget-aware scoring

review-count confidence boost

added noise filtering guard to reduce mismatched results (e.g., non-Italian foods ranking for Italian queries)


🤖 LLM + RAG Pipeline (Layer 3)

fixed field mismatches (review_text → review_snippets)

improved prompt design across all three modes:

baseline LLM

profile-aware LLM

RAG-based system

reduced temperature for more stable outputs

clarified baseline/profile limitations for transparency

strengthened RAG grounding requirements:

explicit request match

taste profile alignment

required evidence phrase

added uncertainty/caution reasoning

improved overall output structure and professionalism




[Hoerim Kim] - Changelog

Refactored the dataset representation to improve retrieval quality by converting structured fields into natural-language text (combined_text).

Improved the embedding-based retrieval pipeline:
- ensured cosine similarity ranking works correctly
- added caching with dataset validation to avoid stale embeddings
- refined retrieval queries to better incorporate user preferences

Enhanced the RAG pipeline:
- passed richer retrieved context (reviews + attributes) to the LLM
- enforced grounding so recommendations come only from retrieved results

Strengthened personalization:
- integrated taste profile into both retrieval and reranking steps

Extended the evaluation pipeline:
- added multiple queries instead of a single test case
- compared baseline LLM, LLM + profile, and LLM + profile + RAG
- added a profile comparison experiment to show how preferences affect outputs

Improved output formatting for clearer comparison and analysis.

Aligned the implementation with the proposal by making the Eat Out mode a complete end-to-end RAG system.

