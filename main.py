import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None
from openai import OpenAI

from src.data_loader import load_reviews
from src.recommend import baseline_recommend, profile_recommend, rag_recommend


def write_section_header(f, title: str) -> None:
    line = "=" * 80
    f.write(f"{line}\n{title}\n{line}\n\n")


def write_retrieved_restaurants(f, retrieved: list) -> None:
    for rank, row in enumerate(retrieved, start=1):
        f.write(
            f"{rank}. {row['title']} | "
            f"Category: {row['category']} | "
            f"Popular food: {row['popular_food']} | "
            f"Online order: {row['online_order']} | "
            f"Reviews: {row['num_reviews']} | "
            f"Score: {row['retrieval_score']}\n"
        )
        f.write(f"   Review snippets: {row['review_snippets'][:350]}\n")
    f.write("\n")


def main():
    if load_dotenv is not None:
        load_dotenv()

    # If a local proxy is configured in your environment, it can break OpenAI requests
    # (e.g., returning 403 via the proxy). Clear proxy env vars for direct API access.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found. Add it to your local .env file.")

    client = OpenAI()

    # Keep this moderate for faster testing and lower API cost.
    # You can increase later if needed.
    df = load_reviews(path="data/restaurants.csv", max_rows=1500)

    query_profiles = [
        {
            "query": "I want cheap Italian food in NYC with good pasta and online ordering.",
            "profile": {
                "preferred_cuisines": ["Italian", "Pizza"],
                "liked_foods": ["pasta", "pizza"],
                "disliked_foods": ["seafood"],
                "budget": "cheap",
                "online_order": "Yes",
                "occasion": "casual dinner",
                "city": "New York City",
            },
        },
        {
            "query": "I am looking for a casual Korean restaurant in NYC with generous portions.",
            "profile": {
                "preferred_cuisines": ["Korean"],
                "liked_foods": ["bbq", "noodles", "rice"],
                "disliked_foods": ["seafood"],
                "budget": "moderate",
                "online_order": "No",
                "occasion": "casual dinner",
                "city": "New York City",
            },
        },
        {
            "query": "Find me a date-night Japanese restaurant in NYC that is not too expensive.",
            "profile": {
                "preferred_cuisines": ["Japanese"],
                "liked_foods": ["sushi", "ramen"],
                "disliked_foods": ["pizza"],
                "budget": "moderate",
                "online_order": "No",
                "occasion": "date night",
                "city": "New York City",
            },
        },
    ]

    profile_a = {
        "preferred_cuisines": ["Italian", "Pizza"],
        "liked_foods": ["pasta", "pizza"],
        "disliked_foods": ["seafood"],
        "budget": "cheap",
        "online_order": "Yes",
        "occasion": "casual dinner",
        "city": "New York City",
    }

    profile_b = {
        "preferred_cuisines": ["Japanese"],
        "liked_foods": ["sushi", "ramen"],
        "disliked_foods": ["pizza"],
        "budget": "moderate",
        "online_order": "No",
        "occasion": "date night",
        "city": "New York City",
    }

    os.makedirs("results", exist_ok=True)
    #output_path = Path("results/milestone2_outputs.txt")
    output_path = Path("results/milestone3_outputs.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        write_section_header(f, "MILESTONE 2 OUTPUTS")

        f.write("Dataset summary:\n")
        f.write(f"- Number of restaurant-level records loaded: {len(df)}\n")
        f.write("- Pipeline settings: baseline LLM vs taste-profile LLM vs taste-profile + RAG\n")
        f.write("- Retrieval: embedding-based semantic search + hybrid reranking\n\n")

        # Main comparison across multiple queries
        for i, item in enumerate(query_profiles, start=1):
            query = item["query"]
            profile = item["profile"]

            write_section_header(f, f"QUERY {i}: {query}")

            baseline_output = baseline_recommend(client, query)
            profile_output = profile_recommend(client, query, profile)
            rag_output, retrieved = rag_recommend(client, query, profile, df, top_k=5)

            f.write("=== BASELINE LLM ===\n")
            f.write(baseline_output + "\n\n")

            f.write("=== LLM + TASTE PROFILE ===\n")
            f.write(profile_output + "\n\n")

            f.write("=== LLM + TASTE PROFILE + RAG ===\n")
            f.write(rag_output + "\n\n")

            f.write("=== RETRIEVED RESTAURANTS ===\n")
            write_retrieved_restaurants(f, retrieved)

        # Taste profile comparison with same query
        comparison_query = "I want something good for dinner in NYC."
        write_section_header(f, f"TASTE PROFILE COMPARISON: {comparison_query}")

        rag_a, retrieved_a = rag_recommend(client, comparison_query, profile_a, df, top_k=5)
        rag_b, retrieved_b = rag_recommend(client, comparison_query, profile_b, df, top_k=5)

        f.write("=== PROFILE A ===\n")
        f.write(str(profile_a) + "\n\n")
        f.write(rag_a + "\n\n")
        f.write("Retrieved restaurants for Profile A:\n")
        write_retrieved_restaurants(f, retrieved_a)

        f.write("=== PROFILE B ===\n")
        f.write(str(profile_b) + "\n\n")
        f.write(rag_b + "\n\n")
        f.write("Retrieved restaurants for Profile B:\n")
        write_retrieved_restaurants(f, retrieved_b)

        write_section_header(f, "SHORT PROGRESS NOTES")
        f.write(
            "- We implemented an end-to-end Eat Out mode.\n"
            "- The system supports three settings: baseline LLM, taste-profile LLM, and taste-profile + RAG.\n"
            "- Retrieval is embedding-based using OpenAI embeddings and cosine similarity.\n"
            "- A lightweight reranking layer adjusts retrieval using cuisine, liked/disliked foods, budget, and ordering preferences.\n"
            "- This milestone focuses on one proposal mode (Eat Out) before expanding to Cook at Home and Drink modes.\n"
        )

    print(f"Saved outputs to {output_path}")


if __name__ == "__main__":
    main()
