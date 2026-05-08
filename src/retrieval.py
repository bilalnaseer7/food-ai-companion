import hashlib
import re
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
    "pakistani": {"biryani", "karahi", "nihari", "haleem", "kebab", "kabob", "tikka", "naan"},
    "thai": {"pad thai", "curry", "spring rolls", "satay", "tom yum"},
    "american": {"burger", "fries", "steak", "bbq", "wings", "sandwich"},
    "south asian": {"biryani", "karahi", "nihari", "haleem", "kebab", "kabob", "tikka", "naan", "curry", "samosa", "dal", "dosa"},
    "halal": {"halal", "gyro", "kebab", "kabob", "chicken", "lamb", "rice", "falafel", "biryani"},
}

REQUEST_CUISINE_ALIASES = {
    # Explicit query intent should beat the learned taste profile. Pakistani
    # is sparse in the static dataset, so related South Asian / halal terms are
    # allowed as static fallback, but unrelated cuisines such as Korean are not.
    "pakistani": {
        "pakistani", "pakistan", "desi", "indian", "halal", "biryani",
        "karahi", "nihari", "haleem", "kebab", "kabob", "tikka", "naan",
        "middle eastern",
    },
    "pakistan": {
        "pakistani", "pakistan", "desi", "indian", "halal", "biryani",
        "karahi", "nihari", "haleem", "kebab", "kabob", "tikka", "naan",
        "middle eastern",
    },
    "indian": {"indian", "curry", "naan", "biryani", "tikka", "samosa", "dal", "dosa"},
    "korean": {"korean", "korean bbq", "bulgogi", "bibimbap", "kimchi"},
    "japanese": {"japanese", "sushi", "ramen", "omakase", "tempura"},
    "sushi": {"japanese", "sushi", "sashimi", "omakase"},
    "chinese": {"chinese", "dim sum", "dumplings", "fried rice", "hot pot"},
    "mexican": {"mexican", "tacos", "burritos", "quesadilla"},
    "italian": {"italian", "pizza", "pasta", "lasagna", "trattoria"},
    "thai": {"thai", "pad thai", "tom yum", "satay"},
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


def _contains_phrase(target_text: str, phrase: str) -> bool:
    phrase = str(phrase or "").strip().lower()
    if not phrase:
        return False
    return re.search(rf"\b{re.escape(phrase)}\b", str(target_text or "").lower()) is not None


def _contains_explicit_alias(target_text: str, phrases: list[str]) -> bool:
    return any(_contains_phrase(target_text, phrase) for phrase in phrases)


def _explicit_query_triggers(query: str) -> list[str]:
    query_text = str(query or "").lower()
    return [
        trigger
        for trigger in REQUEST_CUISINE_ALIASES
        if _contains_phrase(query_text, trigger)
    ]


def _explicit_query_aliases(query: str) -> list[str]:
    aliases: list[str] = []
    for trigger in _explicit_query_triggers(query):
        aliases.extend(sorted(REQUEST_CUISINE_ALIASES[trigger]))
    return list(dict.fromkeys(aliases))


def _explicit_match_strength(
    query: str,
    title: str,
    category: str,
    review_text: str,
    popular_food: str,
) -> int:
    """
    Return 0/1/2/3 for explicit cuisine intent alignment.

    3 = strong evidence in restaurant name or exact requested category.
    2 = supporting evidence in review snippets or popular-food text.
    1 = acceptable static fallback category for sparse cuisines such as Pakistani.
    0 = no support; should not be shown for explicit cuisine queries.
    """
    triggers = _explicit_query_triggers(query)
    aliases = _explicit_query_aliases(query)
    if not triggers:
        return 0

    evidence_text = " ".join([title, review_text, popular_food])
    if _contains_explicit_alias(title, aliases):
        return 3
    if _contains_explicit_alias(category, triggers):
        return 3

    if any(trigger in {"pakistani", "pakistan"} for trigger in triggers):
        if _contains_explicit_alias(category, ["pakistani", "halal"]):
            return 3
        if _contains_explicit_alias(title, ["indian", "halal", "desi", "biryani", "kebab", "kabob"]):
            return 3
        if _contains_explicit_alias(" ".join([review_text, popular_food]), aliases):
            return 2
        if _contains_explicit_alias(category, ["indian", "middle eastern"]):
            return 1

    if _contains_explicit_alias(category, aliases):
        return 3
    if _contains_explicit_alias(evidence_text, aliases):
        return 2

    return 0


def _explicit_category_override(
    query: str,
    title: str,
    category: str,
    review_text: str,
    popular_food: str,
) -> str:
    triggers = _explicit_query_triggers(query)
    if not triggers:
        return ""

    evidence_text = " ".join([title, category, review_text, popular_food])

    if any(trigger in {"pakistani", "pakistan"} for trigger in triggers):
        if _contains_explicit_alias(evidence_text, ["pakistani", "pakistan"]):
            return "Pakistani, South Asian"
        if _contains_explicit_alias(evidence_text, ["halal"]):
            return "Halal, South Asian"
        if _contains_explicit_alias(
            evidence_text,
            ["indian", "biryani", "karahi", "tikka", "naan", "samosa", "dal", "dosa", "curry"],
        ):
            return "Indian, South Asian"
        if _contains_explicit_alias(category, ["middle eastern"]):
            return "Middle Eastern, South Asian"

    if any(trigger == "indian" for trigger in triggers):
        if _contains_explicit_alias(evidence_text, ["indian", "biryani", "tikka", "naan", "samosa", "dal", "dosa", "curry"]):
            return "Indian"

    return ""


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


def find_static_cuisine_matches(query: str, df, top_k: int = 5) -> list[dict]:
    """
    Lightweight exact/fallback matcher for explicit cuisine requests.

    The Streamlit demo keeps the embedding index intentionally small for speed,
    so sparse cuisines can exist later in the raw CSV without appearing in the
    RAG cache. This matcher lets an explicit query like "Pakistani" use those
    local rows without rebuilding embeddings or inventing live results.
    """
    explicit_aliases = _explicit_query_aliases(query)
    if not explicit_aliases or df is None or len(df) == 0:
        return []

    scored_rows: list[tuple[float, int, dict]] = []
    triggers = _explicit_query_triggers(query)
    exact_terms = list(triggers)
    if any(trigger in {"pakistani", "pakistan"} for trigger in triggers):
        exact_terms.extend(["pakistani", "pakistan"])
    exact_terms = list(dict.fromkeys(exact_terms))

    for idx, row_obj in enumerate(df.to_dict("records")):
        title = str(row_obj.get("title", "") or "").lower()
        category = str(row_obj.get("category", "") or "").lower()
        review_text = str(row_obj.get("review_text", row_obj.get("review_snippets", "")) or "").lower()
        popular_food = str(row_obj.get("popular_food", "") or "").lower()
        strength = _explicit_match_strength(query, title, category, review_text, popular_food)
        if strength == 0:
            continue

        row = dict(row_obj)
        category_override = _explicit_category_override(query, title, category, review_text, popular_food)
        if category_override:
            row["category"] = category_override
            category = category_override.lower()

        if row.get("popular_food") and not _food_matches_category(str(row["popular_food"]), category):
            row["popular_food"] = ""

        original_evidence = " ".join([title, str(row_obj.get("category", "") or "").lower()])
        exact_requested_signal = any(_contains_phrase(original_evidence, term) for term in exact_terms)
        source_kind = "static_exact_cuisine" if exact_requested_signal else "static_sparse_cuisine_fallback"
        row["match_source"] = source_kind
        row["match_note"] = (
            "Exact local cuisine/category signal from the full restaurant CSV."
            if source_kind == "static_exact_cuisine"
            else "Closest local cuisine fallback because exact matches are sparse in the static CSV."
        )

        try:
            review_count = int(row.get("num_reviews", 0) or 0)
        except (TypeError, ValueError):
            review_count = 0
        try:
            quality_score = float(row.get("quality_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            quality_score = 0.0

        score = (strength * 1000.0) + (500.0 if exact_requested_signal else 0.0)
        score += min(review_count, 5000) / 10.0
        score += quality_score * 10.0
        row["retrieval_score"] = round(score / 1000.0, 4)
        scored_rows.append((score, idx, row))

    scored_rows.sort(key=lambda item: (-item[0], item[1]))

    results: list[dict] = []
    seen_titles: set[str] = set()
    for _, _, row in scored_rows:
        title_key = str(row.get("title", "")).strip().lower()
        if not title_key or title_key in seen_titles:
            continue
        results.append(row)
        seen_titles.add(title_key)
        if len(results) >= top_k:
            break

    return results


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

    explicit_aliases = _explicit_query_aliases(query)

    # Build a weighted cuisine string: repeat high-scoring cuisines so the
    # embedding space naturally pulls toward them.
    weighted_cuisines: list[str] = []
    if not explicit_aliases:
        for cuisine, score in sorted(cuisine_scores.items(), key=lambda x: -x[1]):
            if score > 0.0:
                repeats = 3 if score >= 0.6 else (2 if score >= 0.3 else 1)
                weighted_cuisines.extend([cuisine] * repeats)

    # Fall back to the static preferred_cuisines list when scores are empty.
    if not weighted_cuisines and not explicit_aliases:
        weighted_cuisines = list(user_profile.get("preferred_cuisines", []))

    # Include foods with a positive score even below the liked_foods threshold.
    learned_liked = [f for f, s in food_scores.items() if s > 0.05]
    explicit_liked = user_profile.get("liked_foods", [])
    all_liked = list(dict.fromkeys(explicit_liked + learned_liked))  # preserve order, dedupe

    learned_disliked = [f for f, s in food_scores.items() if s < -0.05]
    explicit_disliked = user_profile.get("disliked_foods", [])
    removed_foods = user_profile.get("removed_foods", [])
    all_disliked = list(dict.fromkeys(explicit_disliked + learned_disliked + removed_foods))

    occasion = user_profile.get("occasion", "")
    budget = user_profile.get("budget", "")
    online_order = user_profile.get("online_order", "")

    parts = [f"User query: {query}."]
    if explicit_aliases:
        parts.append(
            "Current request cuisine intent must be prioritized over saved taste profile: "
            f"{', '.join(explicit_aliases)}."
        )
    if weighted_cuisines:
        parts.append(f"Cuisine preferences (weighted by strength): {', '.join(weighted_cuisines)}.")
    if all_liked:
        parts.append(f"Liked foods: {', '.join(all_liked)}.")
    if all_disliked:
        parts.append(f"Disliked foods: {', '.join(all_disliked)}.")
    removed_foods = user_profile.get("removed_foods", [])
    if removed_foods:
        parts.append(f"Removed foods: {', '.join(removed_foods)}.")
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
    explicit_aliases = _explicit_query_aliases(query)
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
        explicit_strength = _explicit_match_strength(query, title, category, review_text, popular_food)

        if explicit_aliases:
            if explicit_strength == 3:
                score += 0.75
            elif explicit_strength == 2:
                score += 0.25
            elif explicit_strength == 1:
                score += 0.08
            else:
                score -= 0.85

        # ── static heuristic adjustments (unchanged from original) ──
        if not explicit_aliases and preferred_cuisines and _contains_any(category, preferred_cuisines):
            score += 0.15
        elif not explicit_aliases and _has_strong_preference_mismatch(row_text, preferred_cuisines):
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
        if not explicit_aliases:
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
            explicit_strength = _explicit_match_strength(
                query,
                str(row["title"]).lower(),
                category,
                str(row["review_text"]).lower(),
                str(row["popular_food"]).lower(),
            )

            if explicit_aliases and explicit_strength == 0:
                continue
            if not explicit_aliases and _has_strong_preference_mismatch(row_text, preferred_cuisines):
                continue
            if (
                strict
                and not explicit_aliases
                and liked_foods
                and not _contains_any(row_text, liked_foods)
                and not _contains_any(category, preferred_cuisines)
            ):
                continue

            if explicit_aliases:
                category_override = _explicit_category_override(
                    query,
                    str(row["title"]).lower(),
                    category,
                    str(row["review_text"]).lower(),
                    str(row["popular_food"]).lower(),
                )
                if category_override:
                    row["category"] = category_override
                    category = category_override.lower()
                if row.get("popular_food") and not _food_matches_category(str(row["popular_food"]), category):
                    row["popular_food"] = ""

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
    if not results and explicit_aliases:
        return []

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
