import os
from dotenv import load_dotenv
from openai import OpenAI

from src.data_loader import load_reviews
from src.recommend import baseline_recommend, profile_recommend, rag_recommend


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found. Add it to your local .env file.")

    client = OpenAI()

    # can raise or lower max_rows depending on speed/cost.
    df = load_reviews(path="data/restaurants.csv", max_rows=3000)

    query = "I want cheap Italian food in NYC with good pasta and online ordering."

    user_profile = {
        "preferred_cuisines": ["Italian", "Pizza"],
        "liked_foods": ["pasta", "pizza"],
        "disliked_foods": ["seafood"],
        "budget": "cheap",
        "online_order": "Yes",
        "occasion": "casual dinner",
    }

    baseline_output = baseline_recommend(client, query)
    profile_output = profile_recommend(client, query, user_profile)
    rag_output, retrieved = rag_recommend(client, query, user_profile, df, top_k=5)

    os.makedirs("results", exist_ok=True)

    output_path = "results/milestone2_outputs.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== BASELINE LLM ===\n")
        f.write(baseline_output + "\n\n")

        f.write("=== LLM + TASTE PROFILE ===\n")
        f.write(profile_output + "\n\n")

        f.write("=== LLM + TASTE PROFILE + RAG ===\n")
        f.write(rag_output + "\n\n")

        f.write("=== RETRIEVED RESTAURANTS ===\n")
        for i, row in enumerate(retrieved, start=1):
            f.write(
                f"{i}. {row['title']} | {row['category']} | Popular food: {row['popular_food']} "
                f"| Online order: {row['online_order']} | Retrieval score: {row['retrieval_score']}\n"
            )

    print("Saved outputs to results/milestone2_outputs.txt")


if __name__ == "__main__":
    main()

