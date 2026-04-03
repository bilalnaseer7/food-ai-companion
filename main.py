import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.data_loader import load_reviews
from src.recommend import baseline_recommend, profile_recommend, rag_recommend, foursquare_recommend
from src.taste_profile import load_profile, save_profile, update_profile, profile_summary


def write_section_header(f, title: str) -> None:
    line = "=" * 80
    f.write(f"{line}\n{title}\n{line}\n\n")


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found. Add it to your local .env file.")

    client = OpenAI()

    df = load_reviews(path="data/restaurants.csv", max_rows=3000)

    # Load persistent profile from disk, fall back to defaults if first run
    default_profile = load_profile()

    # Seed with known preferences if profile is empty (first run)
    if not default_profile["preferred_cuisines"] and not default_profile["cuisine_scores"]:
        default_profile.update({
            "preferred_cuisines": ["Italian", "Pizza"],
            "liked_foods": ["pasta", "pizza"],
            "disliked_foods": ["seafood"],
            "budget": "cheap",
            "online_order": "Yes",
            "occasion": "casual dinner",
        })
        save_profile(default_profile)

    profile_a = {
        "preferred_cuisines": ["Italian", "Pizza"],
        "liked_foods": ["pasta", "pizza"],
        "disliked_foods": ["seafood"],
        "budget": "cheap",
        "online_order": "Yes",
        "occasion": "casual dinner",
        "cuisine_scores": {},
        "food_scores": {},
        "accepted": [],
        "rejected": [],
    }

    profile_b = {
        "preferred_cuisines": ["Japanese"],
        "liked_foods": ["sushi", "ramen"],
        "disliked_foods": ["pizza"],
        "budget": "moderate",
        "online_order": "No",
        "occasion": "date night",
        "cuisine_scores": {},
        "food_scores": {},
        "accepted": [],
        "rejected": [],
    }

    queries = [
        "I want cheap Italian food in NYC with good pasta and online ordering.",
        "I am looking for a casual Korean restaurant in NYC with generous portions.",
        "Find me a date-night Japanese restaurant in NYC that is not too expensive.",
    ]

    os.makedirs("results", exist_ok=True)
    output_path = Path("results/milestone2_outputs.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        write_section_header(f, "MILESTONE 2 OUTPUTS")

        f.write("Dataset summary:\n")
        f.write(f"- Number of review rows loaded: {len(df)}\n")
        f.write("- Pipeline settings: baseline LLM vs taste-profile LLM vs taste-profile + RAG vs Foursquare RAG\n\n")

        write_section_header(f, "ACTIVE TASTE PROFILE")
        f.write(profile_summary(default_profile) + "\n\n")

        for i, query in enumerate(queries, start=1):
            write_section_header(f, f"QUERY {i}: {query}")

            baseline_output = baseline_recommend(client, query)
            profile_output = profile_recommend(client, query, default_profile)
            rag_output, retrieved = rag_recommend(client, query, default_profile, df, top_k=5)
            fsq_output, fsq_restaurants = foursquare_recommend(client, query, default_profile)

            f.write("=== BASELINE LLM ===\n")
            f.write(baseline_output + "\n\n")

            f.write("=== LLM + TASTE PROFILE ===\n")
            f.write(profile_output + "\n\n")

            f.write("=== LLM + TASTE PROFILE + RAG (static CSV) ===\n")
            f.write(rag_output + "\n\n")

            f.write("=== RETRIEVED RESTAURANTS (static CSV) ===\n")
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

            f.write("=== LLM + TASTE PROFILE + RAG (live Foursquare) ===\n")
            f.write(fsq_output + "\n\n")
            f.write("=== RETRIEVED RESTAURANTS (Foursquare) ===\n")
            for rank, r in enumerate(fsq_restaurants, start=1):
                price_str = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}.get(r.get("price"), "?")
                f.write(
                    f"{rank}. {r['name']} | "
                    f"{', '.join(r['categories'][:2])} | "
                    f"{price_str} | "
                    f"Rating: {r.get('rating', 'N/A')}/10 | "
                    f"{r['address']}\n"
                )
            f.write("\n")

            if retrieved:
                default_profile = update_profile(
                    default_profile,
                    restaurant_name=retrieved[0]["title"],
                    accepted=True,
                    cuisines=[retrieved[0]["category"]],
                    foods=[retrieved[0]["popular_food"]],
                )
            if len(retrieved) > 1:
                default_profile = update_profile(
                    default_profile,
                    restaurant_name=retrieved[1]["title"],
                    accepted=False,
                    cuisines=[retrieved[1]["category"]],
                    foods=[retrieved[1]["popular_food"]],
                )
            
            # Update profile from top Foursquare result
            if fsq_restaurants:
                default_profile = update_profile(
                    default_profile,
                    restaurant_name=fsq_restaurants[0]["name"],
                    accepted=True,
                    cuisines=fsq_restaurants[0]["categories"][:1],
                    foods=None,
                )
            if len(fsq_restaurants) > 1:
                default_profile = update_profile(
                    default_profile,
                    restaurant_name=fsq_restaurants[1]["name"],
                    accepted=False,
                    cuisines=fsq_restaurants[1]["categories"][:1],
                    foods=None,
                )

        save_profile(default_profile)
        f.write("\n")

        write_section_header(f, "UPDATED TASTE PROFILE (after feedback simulation)")
        f.write(profile_summary(default_profile) + "\n\n")

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

        write_section_header(f, "SHORT PROGRESS NOTES")
        f.write(
            "- We implemented an end-to-end Eat Out mode.\n"
            "- The system supports four settings: baseline LLM, taste-profile LLM, taste-profile + RAG (static), and taste-profile + RAG (live Foursquare).\n"
            "- Retrieval is embedding-based using OpenAI embeddings and cosine similarity.\n"
            "- A lightweight personalization reranking layer adjusts retrieval using cuisine, liked/disliked foods, budget, and ordering preferences.\n"
            "- Taste profile persists to disk (data/taste_profile.json) and updates incrementally from simulated feedback.\n"
            "- Foursquare integration provides live NYC restaurant data as an alternative retrieval source.\n"
            "- The current milestone focuses on one proposal mode (Eat Out) before expanding to Cook at Home and Drink modes.\n"
        )

    print(f"Saved outputs to {output_path}")


if __name__ == "__main__":
    main()
    
