import hashlib
from pathlib import Path

import numpy as np
from openai import OpenAI


EMBED_MODEL = "text-embedding-3-small"
GENERIC_POPULAR_FOODS = {"", "bar", "beer", "diner", "garden", "pub", "pub food", "the pub", "vegetarian"}
CATEGORY_FOOD_HINTS = {
    "italian": {"pizza", "pasta", "spaghetti", "lasagna", "carbonara", "cannoli", "tiramisu"},
    "pizza": {"pizza", "pasta", "spaghetti", "lasagna", "cannoli"},
    "korean": {"korean bbq", "bbq", "bulgogi", "bibimbap", "kimchi", "noodles", "rice"},
    "japanese": {"sushi", "sashimi", "ramen", "omakase", "tempura"},
    "sushi": {"sushi", "sashimi", "omakase"},
}


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _build_retrieval_query(query: str, user_profile: dict) -> str:
    preferred_cuisines = ", ".join(user_profile.get("preferred_cuisines", []))
    liked_foods = ", ".join(user_profile.get("liked_foods", []))
    disliked_foods = ", ".join(user_profile.get("disliked_foods", []))
    budget = user_profile.get("budget", "")
    online_order = user_profile.get("online_order", "")

    return (
        f"User query: {query}. "
        f"Preferred cuisines: {preferred_cuisines}. "
        f"Liked foods: {liked_foods}. "
        f"Disliked foods: {disliked_foods}. "
        f"Budget: {budget}. "
        f"Online order preference: {online_order}."
    )


def _contains_any(target_text: str, phrases: list[str]) -> bool:
    target_text = target_text.lower()
    for phrase in phrases:
        phrase = phrase.strip().lower()
        if phrase and phrase in target_text:
            return True
    return False


def _food_matches_category(food: str, category: str) -> bool:
    food = food.lower()
    category = category.lower()
    matched_hint_groups = 0

    for label, hints in CATEGORY_FOOD_HINTS.items():
        if label in category:
            matched_hint_groups += 1
            if food in hints:
                return True

    return matched_hint_groups == 0


def _has_strong_preference_mismatch(row_text: str, preferred_cuisines: list[str]) -> bool:
    if not preferred_cuisines:
        return False
    return not _contains_any(row_text, preferred_cuisines)

def _dataset_signature(df) -> str:
    joined = "||".join(df["combined_text"].astype(str).tolist())
    return hashlib.md5(joined.encode("utf-8")).hexdigest()

def build_or_load_embeddings(
    df,
    client: OpenAI,
    cache_path: str = "data/restaurant_embeddings.npz",
    batch_size: int = 100,
) -> np.ndarray:
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    current_signature = _dataset_signature(df)

    if cache_file.exists():
        cached = np.load(cache_file, allow_pickle=True)
        cached_embeddings = cached["embeddings"]
        cached_n_rows = int(cached["n_rows"])
        cached_signature = str(cached["signature"])

        if cached_n_rows == len(df) and cached_signature == current_signature:
            return cached_embeddings

    texts = df["combined_text"].tolist()
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    embeddings = np.array(all_embeddings, dtype="float32")
    embeddings = _normalize(embeddings)

    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        n_rows=len(df),
        signature=current_signature,
    )

    return embeddings


def retrieve_restaurants(
    query: str,
    user_profile: dict,
    df,
    client: OpenAI,
    top_k: int = 5,
    cache_path: str = "data/restaurant_embeddings.npz",
):
    embeddings = build_or_load_embeddings(df, client, cache_path=cache_path)

    retrieval_query = _build_retrieval_query(query, user_profile)
    query_embedding = client.embeddings.create(
        model=EMBED_MODEL,
        input=[retrieval_query],
    ).data[0].embedding

    query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
    query_embedding = _normalize(query_embedding)[0]

    cosine_scores = embeddings @ query_embedding

    preferred_cuisines = [x.lower() for x in user_profile.get("preferred_cuisines", [])]
    liked_foods = [x.lower() for x in user_profile.get("liked_foods", [])]
    disliked_foods = [x.lower() for x in user_profile.get("disliked_foods", [])]
    online_pref = str(user_profile.get("online_order", "")).strip().lower()
    budget_pref = str(user_profile.get("budget", "")).strip().lower()

    scored = []
    for idx, base_score in enumerate(cosine_scores):
        row = df.iloc[idx]
        score = float(base_score)

        category = str(row["category"]).lower()
        review_text = str(row["review_text"]).lower()
        popular_food = str(row["popular_food"]).lower()
        online_order = str(row["online_order"]).lower()
        title = str(row["title"]).lower()
        row_text = " ".join([title, category, review_text, popular_food])
        quality_score = float(row.get("quality_score", 1.0))

        if preferred_cuisines and _contains_any(category, preferred_cuisines):
            score += 0.15
        elif _has_strong_preference_mismatch(row_text, preferred_cuisines):
            score -= 0.25

        if liked_foods and _contains_any(f"{popular_food} {review_text}", liked_foods):
            score += 0.10

        if disliked_foods and _contains_any(f"{popular_food} {review_text}", disliked_foods):
            score -= 0.12

        if online_pref in {"yes", "no"} and online_order == online_pref:
            score += 0.05

        if budget_pref == "cheap":
            cheap_signals = ["cheap", "affordable", "budget", "value", "inexpensive"]
            if any(signal in review_text for signal in cheap_signals):
                score += 0.08
        elif budget_pref == "moderate":
            moderate_signals = ["moderate", "reasonable", "fair price", "casual"]
            if any(signal in review_text for signal in moderate_signals):
                score += 0.04
        elif budget_pref == "premium":
            premium_signals = ["fine dining", "upscale", "premium", "expensive", "high-end"]
            if any(signal in review_text for signal in premium_signals):
                score += 0.08

        if popular_food in GENERIC_POPULAR_FOODS:
            score -= 0.05
        elif not _food_matches_category(popular_food, category):
            score -= 0.25

        score += (quality_score - 0.5) * 0.10
        score += min(row["num_reviews"] / 10000.0, 0.05)

        scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    seen_titles = set()

    for score, idx in scored:
        row = df.iloc[idx].to_dict()
        title = row["title"]
        category = str(row["category"]).lower()
        row_text = " ".join(
            [
                str(row["title"]).lower(),
                category,
                str(row["review_text"]).lower(),
                str(row["popular_food"]).lower(),
            ]
        )

        if title in seen_titles:
            continue

        if _has_strong_preference_mismatch(row_text, preferred_cuisines):
            continue

        if liked_foods and not _contains_any(row_text, liked_foods) and not _contains_any(category, preferred_cuisines):
            # Keep stricter filtering only when the user has given concrete foods and cuisines.
            continue

        row["retrieval_score"] = round(score, 4)
        results.append(row)
        seen_titles.add(title)

        if len(results) >= top_k:
            break

    return results
