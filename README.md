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

