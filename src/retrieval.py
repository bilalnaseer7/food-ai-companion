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
    "chinese": {"dim sum", "dumplings", "noodles", "fried rice", "peking duck", "hot pot"},
    "mexican": {"tacos", "burritos", "enchiladas", "guacamole", "quesadilla"},
    "indian": {"curry", "naan", "biryani", "tikka", "samosa", "dal"},
    "thai": {"pad thai", "curry", "spring rolls", "satay", "tom yum"},
    "american": {"burger", "fries", "steak", "bbq", "wings", "sandwich"},
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


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


# ── retrieval query builder ───────────────────────────────────────────────────

def _build_retrieval_query(query: str, user_profile: dict) -> str:
    """
    Build an enriched retrieval query from the user query + taste profile.

    Changes vs. original:
    - Top cuisines are weighted by their learned cuisine_scores (not just the
      binary preferred_cuisines list), so a cuisine the user really loves gets
      repeated for stronger embedding signal.
    - High-scoring foods from food_scores are included even if they haven't
      crossed the liked_foods threshold yet.
    - Occasion is surfaced more explicitly to help semantic matching (e.g.
      "date night" pulls romantic/upscale restaurant descriptions).
    """
    cuisine_scores: dict = user_profile.get("cuisine_scores", {})
    food_scores: dict = user_profile.get("food_scores", {})

    # Build a weighted cuisine string: repeat high-scoring cuisines so the
    # embedding space naturally pulls toward them.
    weighted_cuisines: list[str] = []
    for cuisine, score in sorted(cuisine_scores.items(), key=lambda x: -x[1]):
        if score > 0.0:
            repeats = 3 if score >= 0.6 else (2 if score >= 0.3 else 1)
            weighted_cuisines.extend([cuisine] * repeats)

    # Fall back to the static preferred_cuisines list when scores are empty.
    if not weighted_cuisines:
        weighted_cuisines = list(user_profile.get("preferred_cuisines", []))

    # Include foods with a positive score even below the liked_foods threshold.
    learned_liked = [f for f, s in food_scores.items() if s > 0.05]
    explicit_liked = user_profile.get("liked_foods", [])
    all_liked = list(dict.fromkeys(explicit_liked + learned_liked))  # preserve order, dedupe

    learned_disliked = [f for f, s in food_scores.items() if s < -0.05]
    explicit_disliked = user_profile.get("disliked_foods", [])
    all_disliked = list(dict.fromkeys(explicit_disliked + learned_disliked))

    occasion = user_profile.get("occasion", "")
    budget = user_profile.get("budget", "")
    online_order = user_profile.get("online_order", "")

    parts = [f"User query: {query}."]
    if weighted_cuisines:
        parts.append(f"Cuisine preferences (weighted by strength): {', '.join(weighted_cuisines)}.")
    if all_liked:
        parts.append(f"Liked foods: {', '.join(all_liked)}.")
    if all_disliked:
        parts.append(f"Disliked foods: {', '.join(all_disliked)}.")
    if budget:
        parts.append(f"Budget: {budget}.")
    if online_order:
        parts.append(f"Online order preference: {online_order}.")
    if occasion:
        parts.append(f"Occasion context: {occasion}.")

    return " ".join(parts)


# ── embedding cache ───────────────────────────────────────────────────────────

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
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)

    embeddings = np.array(all_embeddings, dtype="float32")
    embeddings = _normalize(embeddings)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        n_rows=len(df),
        signature=current_signature,
    )
    return embeddings


# ── score adjustment from learned profile ────────────────────────────────────

def _profile_score_adjustment(row_text: str, category: str, popular_food: str, user_profile: dict) -> float:
    """
    Apply fine-grained score nudges based on the *learned* cuisine_scores and
    food_scores from the taste profile (incremental feedback loop).

    This runs on top of the static heuristic adjustments already in
    retrieve_restaurants, so keep magnitudes modest to avoid dominating.
    """
    adjustment = 0.0
    cuisine_scores: dict = user_profile.get("cuisine_scores", {})
    food_scores: dict = user_profile.get("food_scores", {})

    # Cuisine score contribution — scaled to ±0.15 max.
    for cuisine, score in cuisine_scores.items():
        if cuisine.lower() in category:
            adjustment += score * 0.15  # score in [-1, 1] → contribution in [-0.15, 0.15]
            break  # one cuisine match is enough

    # Food score contribution — scaled to ±0.10 max.
    search_text = f"{popular_food} {row_text}"
    for food, score in food_scores.items():
        if food.lower() in search_text:
            adjustment += score * 0.10
            # Don't break — multiple food signals can accumulate, but cap total.

    return max(-0.20, min(0.20, adjustment))  # cap net adjustment


# ── main retrieval function ───────────────────────────────────────────────────

def retrieve_restaurants(
    query: str,
    user_profile: dict,
    df,
    client: OpenAI,
    top_k: int = 5,
    cache_path: str = "data/restaurant_embeddings.npz",
) -> list[dict]:
    embeddings = build_or_load_embeddings(df, client, cache_path=cache_path)

    retrieval_query = _build_retrieval_query(query, user_profile)
    query_embedding = (
        client.embeddings.create(model=EMBED_MODEL, input=[retrieval_query])
        .data[0]
        .embedding
    )
    query_embedding = _normalize(np.array(query_embedding, dtype="float32").reshape(1, -1))[0]
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

        # ── static heuristic adjustments (unchanged from original) ──
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
            if any(s in review_text for s in ["cheap", "affordable", "budget", "value", "inexpensive"]):
                score += 0.08
        elif budget_pref == "moderate":
            if any(s in review_text for s in ["moderate", "reasonable", "fair price", "casual"]):
                score += 0.04
        elif budget_pref == "premium":
            if any(s in review_text for s in ["fine dining", "upscale", "premium", "expensive", "high-end"]):
                score += 0.08

        if popular_food in GENERIC_POPULAR_FOODS:
            score -= 0.05
        elif not _food_matches_category(popular_food, category):
            score -= 0.25

        score += (quality_score - 0.5) * 0.10
        score += min(row["num_reviews"] / 10000.0, 0.05)

        # ── NEW: learned profile score adjustment ──
        score += _profile_score_adjustment(row_text, category, popular_food, user_profile)

        scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)

    # ── filtering pass with fallback ─────────────────────────────────────────
    # Original code had a hard filter that could return 0 results when both
    # liked_foods and preferred_cuisines were set but no single restaurant
    # matched both. We now do two passes:
    #   Pass 1 (strict)  – original logic, cuisine + food match required.
    #   Pass 2 (relaxed) – cuisine match alone is enough, used only if pass 1
    #                      yields fewer than top_k results.

    def _collect(scored_list, strict: bool) -> list[dict]:
        results: list[dict] = []
        seen_titles: set[str] = set()

        for score, idx in scored_list:
            row = df.iloc[idx].to_dict()
            title = row["title"]
            category = str(row["category"]).lower()
            row_text = " ".join([
                str(row["title"]).lower(),
                category,
                str(row["review_text"]).lower(),
                str(row["popular_food"]).lower(),
            ])

            if title in seen_titles:
                continue
            if _has_strong_preference_mismatch(row_text, preferred_cuisines):
                continue
            if strict and liked_foods and not _contains_any(row_text, liked_foods) and not _contains_any(category, preferred_cuisines):
                continue

            row["retrieval_score"] = round(score, 4)
            results.append(row)
            seen_titles.add(title)

            if len(results) >= top_k:
                break

        return results

    results = _collect(scored, strict=True)

    # Fallback: relax the liked_foods filter if we didn't get enough results.
    if len(results) < top_k:
        results = _collect(scored, strict=False)

    # Last-resort fallback: drop cuisine filter too, return top scored rows.
    if not results:
        seen: set[str] = set()
        for score, idx in scored:
            row = df.iloc[idx].to_dict()
            title = row["title"]
            if title not in seen:
                row["retrieval_score"] = round(score, 4)
                results.append(row)
                seen.add(title)
            if len(results) >= top_k:
                break

    return results
