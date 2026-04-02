# food-ai-companion

[Bilal] - Changelog


Set up the GitHub repository and project structure (data/, src/, main.py, etc.) for the Milestone 2 prototype.
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
