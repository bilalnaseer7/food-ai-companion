import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.data_loader import load_reviews
from src.recommend import baseline_recommend, profile_recommend, rag_recommend


def write_section_header(f, title: str) -> None:
    line = "=" * 80
    f.write(f"{line}\n{title}\n{line}\n\n")


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found. Add it to your local .env file.")

    client = OpenAI()

    # Adjust max_rows depending on speed/cost
    df = load_reviews(path="data/restaurants.csv", max_rows=3000)

    queries = [
        "I want cheap Italian food in NYC with good pasta and online ordering.",
        "I am looking for a casual Korean restaurant in NYC with generous portions.",
        "Find me a date-night Japanese restaurant in NYC that is not too expensive.",
    ]

    default_profile = {
        "preferred_cuisines": ["Italian", "Pizza"],
        "liked_foods": ["pasta", "pizza"],
        "disliked_foods": ["seafood"],
        "budget": "cheap",
        "online_order": "Yes",
        "occasion": "casual dinner",
    }

    profile_a = {
        "preferred_cuisines": ["Italian", "Pizza"],
        "liked_foods": ["pasta", "pizza"],
        "disliked_foods": ["seafood"],
        "budget": "cheap",
        "online_order": "Yes",
        "occasion": "casual dinner",
    }

    profile_b = {
        "preferred_cuisines": ["Japanese"],
        "liked_foods": ["sushi", "ramen"],
        "disliked_foods": ["pizza"],
        "budget": "moderate",
        "online_order": "No",
        "occasion": "date night",
    }

    os.makedirs("results", exist_ok=True)
    output_path = Path("results/milestone2_outputs.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        write_section_header(f, "MILESTONE 2 OUTPUTS")

        f.write("Dataset summary:\n")
        f.write(f"- Number of review rows loaded: {len(df)}\n")
        f.write("- Pipeline settings: baseline LLM vs taste-profile LLM vs taste-profile + RAG\n\n")

        # Main comparison across 3 queries
        for i, query in enumerate(queries, start=1):
            write_section_header(f, f"QUERY {i}: {query}")

            baseline_output = baseline_recommend(client, query)
            profile_output = profile_recommend(client, query, default_profile)
            rag_output, retrieved = rag_recommend(client, query, default_profile, df, top_k=5)

            f.write("=== BASELINE LLM ===\n")
            f.write(baseline_output + "\n\n")

            f.write("=== LLM + TASTE PROFILE ===\n")
            f.write(profile_output + "\n\n")

            f.write("=== LLM + TASTE PROFILE + RAG ===\n")
            f.write(rag_output + "\n\n")

            f.write("=== RETRIEVED RESTAURANTS ===\n")
            for rank, row in enumerate(retrieved, start=1):
                f.write(
                    f"{rank}. {row['title']} | "
                    f"Category: {row['category']} | "
                    f"Popular food: {row['popular_food']} | "
                    f"Online order: {row['online_order']} | "
                    f"Reviews: {row['num_reviews']} | "
                    f"Score: {row['retrieval_score']}\n"
                )
                f.write(f"   Review excerpt: {row['review_text'][:250]}\n")
            f.write("\n")

        # Taste profile comparison with same query
        comparison_query = "I want something good for dinner in NYC."
        write_section_header(f, f"TASTE PROFILE COMPARISON: {comparison_query}")

        rag_a, retrieved_a = rag_recommend(client, comparison_query, profile_a, df, top_k=5)
        rag_b, retrieved_b = rag_recommend(client, comparison_query, profile_b, df, top_k=5)

        f.write("=== PROFILE A ===\n")
        f.write(str(profile_a) + "\n\n")
        f.write(rag_a + "\n\n")

        f.write("Retrieved restaurants for Profile A:\n")
        for rank, row in enumerate(retrieved_a, start=1):
            f.write(
                f"{rank}. {row['title']} | "
                f"Category: {row['category']} | "
                f"Popular food: {row['popular_food']} | "
                f"Score: {row['retrieval_score']}\n"
            )
        f.write("\n")

        f.write("=== PROFILE B ===\n")
        f.write(str(profile_b) + "\n\n")
        f.write(rag_b + "\n\n")

        f.write("Retrieved restaurants for Profile B:\n")
        for rank, row in enumerate(retrieved_b, start=1):
            f.write(
                f"{rank}. {row['title']} | "
                f"Category: {row['category']} | "
                f"Popular food: {row['popular_food']} | "
                f"Score: {row['retrieval_score']}\n"
            )
        f.write("\n")

        # Short interpretation note for professor check-in
        write_section_header(f, "SHORT PROGRESS NOTES")
        f.write(
            "- We implemented an end-to-end Eat Out mode.\n"
            "- The system supports three settings: baseline LLM, taste-profile LLM, and taste-profile + RAG.\n"
            "- Retrieval is embedding-based using OpenAI embeddings and cosine similarity.\n"
            "- A lightweight personalization reranking layer adjusts retrieval using cuisine, liked/disliked foods, budget, and ordering preferences.\n"
            "- The current milestone focuses on one proposal mode (Eat Out) before expanding to Cook at Home and Drink modes.\n"
        )

    print(f"Saved outputs to {output_path}")


if __name__ == "__main__":
    main()
