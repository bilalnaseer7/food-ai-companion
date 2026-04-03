import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI


EMBED_MODEL = "text-embedding-3-small"

MISSING_TEXT_VALUES = {"", "no", "none", "nan", "n/a", "null"}
GENERIC_FOOD_VALUES = {
    "bar",
    "beer",
    "beers",
    "diner",
    "garden",
    "pub",
    "pub food",
    "the pub",
    "vegetarian",
    "an italian restaurant",
}
FOOD_PATTERNS = [
    "korean bbq",
    "bbq",
    "omakase",
    "sushi",
    "sashimi",
    "ramen",
    "pizza",
    "pasta",
    "spaghetti",
    "lasagna",
    "carbonara",
    "cannoli",
    "tiramisu",
    "bibimbap",
    "bulgogi",
    "kimchi",
    "dumplings",
    "steak",
    "burger",
    "tacos",
    "paella",
]
CATEGORY_FOOD_HINTS = {
    "italian": {"pizza", "pasta", "spaghetti", "lasagna", "carbonara", "cannoli", "tiramisu"},
    "pizza": {"pizza", "pasta", "spaghetti", "lasagna", "cannoli"},
    "korean": {"korean bbq", "bbq", "bulgogi", "bibimbap", "kimchi", "noodles", "rice"},
    "japanese": {"sushi", "sashimi", "ramen", "omakase", "tempura"},
    "sushi": {"sushi", "sashimi", "omakase"},
    "steakhouse": {"steak", "ribeye"},
}
CATEGORY_RULES = [
    ("korean", ["Asian", "Korean"]),
    ("bbq", ["Asian", "Korean"]),
    ("mexican", ["Mexican", "Latin"]),
    ("burrito", ["Mexican", "Latin"]),
    ("taco", ["Mexican", "Latin"]),
    ("sushi", ["Japanese", "Sushi"]),
    ("omakase", ["Japanese", "Sushi"]),
    ("ramen", ["Japanese", "Asian"]),
    ("japanese", ["Japanese"]),
    ("pizza", ["Italian", "Pizza"]),
    ("pizzeria", ["Italian", "Pizza"]),
    ("italian", ["Italian"]),
    ("pasta", ["Italian"]),
    ("trattoria", ["Italian"]),
]
CATEGORY_PRIORITY = {
    "Italian": 1,
    "Pizza": 2,
    "Japanese": 3,
    "Sushi": 4,
    "Asian": 5,
    "Korean": 6,
    "American": 7,
    "Steakhouse": 8,
    "Bar": 9,
    "Cafe": 10,
    "Mexican": 11,
    "Latin": 12,
}


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _clean_text(value) -> str:
    text = str(value).strip().replace("“", "").replace("”", "").replace('"', "")
    if text.lower() in MISSING_TEXT_VALUES:
        return ""
    return " ".join(text.split())


def _normalize_online_order(value) -> str:
    text = _clean_text(value).lower()
    if text == "yes":
        return "Yes"
    if text == "no":
        return "No"
    return "No"


def _split_category_labels(category: str) -> list[str]:
    return [part.strip() for part in category.split(",") if part.strip()]


def _order_category_labels(labels: list[str]) -> list[str]:
    unique = []
    for label in labels:
        if label not in unique:
            unique.append(label)
    return sorted(unique, key=lambda item: (CATEGORY_PRIORITY.get(item, 999), item))


def _infer_category_from_text(title: str, category: str, review: str) -> list[str]:
    text = f"{title} {review}".lower()
    inferred = []
    for keyword, labels in CATEGORY_RULES:
        if keyword in text:
            inferred.extend(labels)
    return _order_category_labels(inferred)


def _clean_category(title: str, category: str, review: str) -> str:
    raw_labels = _order_category_labels(_split_category_labels(_clean_text(category)))
    inferred_labels = _infer_category_from_text(title, category, review)

    if not raw_labels:
        return ", ".join(inferred_labels)

    if inferred_labels:
        raw_lower = {label.lower() for label in raw_labels}
        inferred_lower = {label.lower() for label in inferred_labels}
        if raw_lower.isdisjoint(inferred_lower) and len(inferred_labels) >= 2:
            return ", ".join(inferred_labels)
        return ", ".join(_order_category_labels(raw_labels + inferred_labels))

    return ", ".join(raw_labels)


def _extract_food_candidates(text: str) -> list[str]:
    lowered = text.lower()
    return [food for food in FOOD_PATTERNS if food in lowered]


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


def _clean_popular_food(title: str, category: str, popular_food: str, review: str) -> str:
    food = _clean_text(popular_food)
    review_text = _clean_text(review)
    text_for_inference = f"{title} {review_text}"
    inferred_candidates = _extract_food_candidates(text_for_inference)

    if food:
        lowered_food = food.lower()
        is_generic = lowered_food in GENERIC_FOOD_VALUES
        is_off_domain = not _food_matches_category(lowered_food, category)
        if not is_generic and not is_off_domain:
            return food

    for candidate in inferred_candidates:
        if _food_matches_category(candidate, category):
            return candidate

    if inferred_candidates:
        return inferred_candidates[0]

    if "pizza" in category.lower():
        return "pizza"
    if "korean" in category.lower():
        return "korean bbq" if "bbq" in text_for_inference.lower() else "bbq"
    if "sushi" in category.lower():
        return "sushi"
    if "japanese" in category.lower():
        return "ramen" if "ramen" in text_for_inference.lower() else "sushi"
    if "italian" in category.lower():
        return "pasta" if "pasta" in text_for_inference.lower() else "pizza"

    return food


def _quality_score(category: str, popular_food: str, review: str) -> float:
    score = 1.0

    if not review:
        score -= 0.15
    if not popular_food:
        score -= 0.20
    elif popular_food.lower() in GENERIC_FOOD_VALUES:
        score -= 0.20
    elif not _food_matches_category(popular_food, category):
        score -= 0.30

    return max(0.0, min(1.0, score))


def _consolidate_restaurants(df: pd.DataFrame) -> pd.DataFrame:
    aggregated_rows = []

    for title, group in df.groupby("title", sort=False):
        group = group.copy()
        group = group.sort_values(by=["quality_score", "num_reviews"], ascending=[False, False])

        best = group.iloc[0]
        categories = group["category"].dropna().astype(str).tolist()
        category_counts = pd.Series(categories).value_counts()
        category = best["category"]
        if not category_counts.empty:
            category = category_counts.index[0]

        food_candidates = (
            group[["popular_food", "quality_score", "num_reviews"]]
            .dropna()
            .sort_values(by=["quality_score", "num_reviews"], ascending=[False, False])
        )
        popular_food = ""
        if not food_candidates.empty:
            for candidate in food_candidates["popular_food"]:
                candidate = _clean_text(candidate)
                if candidate:
                    popular_food = candidate
                    break

        review_parts = []
        seen_reviews = set()
        for text in group["review_snippets"].astype(str):
            cleaned = _clean_text(text)
            if cleaned and cleaned.lower() not in seen_reviews:
                review_parts.append(cleaned)
                seen_reviews.add(cleaned.lower())
            if len(review_parts) >= 3:
                break

        review_snippets = " | ".join(review_parts)

        online_counts = group["online_order"].value_counts()
        online_order = online_counts.index[0] if not online_counts.empty else "No"

        aggregated_rows.append(
            {
                "title": title,
                "num_reviews": int(group["num_reviews"].max()),
                "category": category,
                "review_snippets": review_snippets,
                "popular_food": popular_food,
                "online_order": online_order,
                "quality_score": float(group["quality_score"].max()),
            }
        )

    result = pd.DataFrame(aggregated_rows)
    result["review_text"] = result["review_snippets"]
    result["combined_text"] = (
        "Restaurant: " + result["title"]
        + " | Category: " + result["category"]
        + " | Popular food: " + result["popular_food"]
        + " | Online order: " + result["online_order"]
        + " | Reviews: " + result["review_snippets"]
    )
    return result


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
        df[col] = df[col].astype(str).map(_clean_text)

    df["online_order"] = df["online_order"].map(_normalize_online_order)
    df["category"] = df.apply(
        lambda row: _clean_category(row["title"], row["category"], row["review_snippets"]),
        axis=1,
    )
    df["popular_food"] = df.apply(
        lambda row: _clean_popular_food(
            row["title"],
            row["category"],
            row["popular_food"],
            row["review_snippets"],
        ),
        axis=1,
    )
    df["quality_score"] = df.apply(
        lambda row: _quality_score(row["category"], row["popular_food"], row["review_snippets"]),
        axis=1,
    )

    return _consolidate_restaurants(df)
