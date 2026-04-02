import os
import numpy as np
from openai import OpenAI


EMBED_MODEL = "text-embedding-3-small"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _build_retrieval_query(query: str, user_profile: dict) -> str:
    preferred_cuisines = ", ".join(user_profile.get("preferred_cuisines", []))
    liked_foods = ", ".join(user_profile.get("liked_foods", []))
    disliked_foods = ", ".join(user_profile.get("disliked_foods", []))
    budget = user_profile.get("budget", "")
    online_order = user_profile.get("online_order", "")
    occasion = user_profile.get("occasion", "")

    return (
        f"User query: {query}. "
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
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        cached_embeddings = cached["embeddings"]
        cached_n_rows = int(cached["n_rows"])
        if cached_n_rows == len(df):
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

    scored = []
    for idx, base_score in enumerate(cosine_scores):
        row = df.iloc[idx]
        score = float(base_score)

        category = row["category"].lower()
        review_text = row["review_text"].lower()
        popular_food = row["popular_food"].lower()
        online_order = row["online_order"].lower()

        if preferred_cuisines and any(c in category for c in preferred_cuisines):
            score += 0.15

        if liked_foods and any(food in popular_food or food in review_text for food in liked_foods):
            score += 0.10

        if disliked_foods and any(food in popular_food or food in review_text for food in disliked_foods):
            score -= 0.12

        if online_pref in {"yes", "no"} and online_order == online_pref:
            score += 0.05

        score += min(row["num_reviews"] / 10000.0, 0.05)

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
