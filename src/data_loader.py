import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI


EMBED_MODEL = "text-embedding-3-small"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _dataset_signature(df) -> str:
    joined = "||".join(df["combined_text"].astype(str).tolist())
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def _build_retrieval_query(query: str, user_profile: dict) -> str:
    preferred_cuisines = ", ".join(user_profile.get("preferred_cuisines", []))
    liked_foods = ", ".join(user_profile.get("liked_foods", []))
    disliked_foods = ", ".join(user_profile.get("disliked_foods", []))
    budget = user_profile.get("budget", "")
    online_order = user_profile.get("online_order", "")
    occasion = user_profile.get("occasion", "")
    city = user_profile.get("city", "New York City")

    return (
        f"User query: {query}. "
        f"City: {city}. "
        f"Preferred cuisines: {preferred_cuisines}. "
        f"Liked foods: {liked_foods}. "
        f"Disliked foods: {disliked_foods}. "
        f"Budget: {budget}. "
        f"Online order preference: {online_order}. "
        f"Occasion: {occasion}."
    )


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


def _contains_any(target_text: str, phrases: list[str]) -> bool:
    target_text = target_text.lower()
    for phrase in phrases:
        phrase = phrase.strip().lower()
        if phrase and phrase in target_text:
            return True
    return False


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

    cheap_signals = ["cheap", "affordable", "budget", "value", "inexpensive", "reasonable"]
    moderate_signals = ["moderate", "reasonable", "casual", "solid value"]
    premium_signals = ["fine dining", "upscale", "premium", "expensive", "high-end", "luxury"]

    scored = []

    for idx, base_score in enumerate(cosine_scores):
        row = df.iloc[idx]
        score = float(base_score)

        category = str(row["category"]).lower()
        popular_food = str(row["popular_food"]).lower()
        review_snippets = str(row["review_snippets"]).lower()
        online_order = str(row["online_order"]).lower()
        combined_text = str(row["combined_text"]).lower()

        # ---- Layer 3: hybrid reranking ----

        # cuisine/category alignment
        if preferred_cuisines and _contains_any(category, preferred_cuisines):
            score += 0.18

        # liked foods in review or popular food
        if liked_foods and (
            _contains_any(popular_food, liked_foods) or _contains_any(review_snippets, liked_foods)
        ):
            score += 0.12

        # disliked foods penalty
        if disliked_foods and (
            _contains_any(popular_food, disliked_foods) or _contains_any(review_snippets, disliked_foods)
        ):
            score -= 0.18

        # online ordering preference
        if online_pref in {"yes", "no"} and online_order == online_pref:
            score += 0.06

        # budget alignment from review language
        if budget_pref == "cheap" and _contains_any(combined_text, cheap_signals):
            score += 0.10
        elif budget_pref == "moderate" and _contains_any(combined_text, moderate_signals):
            score += 0.05
        elif budget_pref == "premium" and _contains_any(combined_text, premium_signals):
            score += 0.10

        # review-count confidence boost
        score += min(row["num_reviews"] / 15000.0, 0.05)

        # noisy signal guard: if category says Italian/Pizza but popular_food is clearly off-domain,
        # reduce its influence slightly instead of letting it dominate
        if _contains_any(category, ["italian", "pizza"]):
            off_domain_foods = ["fried rice", "sushi", "biryani", "ramen", "taco"]
            if _contains_any(popular_food, off_domain_foods):
                score -= 0.08

        scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    seen_titles = set()

    for score, idx in scored:
        row = df.iloc[idx].to_dict()
        title = row["title"]

        if title in seen_titles:
            continue

        row["retrieval_score"] = round(score, 4)
        results.append(row)
        seen_titles.add(title)

        if len(results) >= top_k:
            break

    return results


def load_reviews(path: str = "data/restaurants.csv", max_rows: int | None = None):
    df = pd.read_csv(path, nrows=max_rows)

    # Normalize likely column name variants (original dataset has typos).
    rename_map = {
        "Title": "title",
        "Number of review": "num_reviews",
        "Catagory": "category",
        "Category": "category",
        "Reveiw Comment": "review_snippets",
        "Review Comment": "review_snippets",
        "Popular food": "popular_food",
        "Online Order": "online_order",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["title", "num_reviews", "category", "review_snippets", "popular_food", "online_order"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"restaurants.csv is missing required columns: {missing}. Found: {list(df.columns)}")

    # Clean review counts like "2,998" -> 2998.
    # Some rows contain values like "No" or "1 review"; coerce those safely.
    num_reviews_clean = (
        df["num_reviews"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+)", expand=False)
    )
    df["num_reviews"] = pd.to_numeric(num_reviews_clean, errors="coerce").fillna(0).astype(int)

    for col in ["title", "category", "review_snippets", "popular_food", "online_order"]:
        df[col] = df[col].astype(str)

    # Compatibility: other modules expect `review_text`.
    df["review_text"] = df["review_snippets"]

    df["combined_text"] = (
        "Restaurant: " + df["title"]
        + " | Category: " + df["category"]
        + " | Popular food: " + df["popular_food"]
        + " | Online order: " + df["online_order"]
        + " | Reviews: " + df["review_snippets"]
    )

    return df

