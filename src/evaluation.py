import re
from statistics import mean
from typing import Iterable

from src.filter_bubble import (
    category_diversity_ratio,
    category_entropy,
    category_labels,
    novelty_ratio,
    primary_category,
    restaurant_name,
)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "but",
    "for",
    "from",
    "good",
    "in",
    "is",
    "me",
    "not",
    "of",
    "or",
    "that",
    "the",
    "too",
    "want",
    "with",
}


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def row_text(row: dict) -> str:
    return " ".join(
        [
            restaurant_name(row),
            ", ".join(category_labels(row)),
            str(row.get("popular_food", "")),
            str(row.get("review_snippets", "")),
            str(row.get("review_text", "")),
            str(row.get("combined_text", "")),
        ]
    ).lower()


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    lowered = str(text or "").lower()
    return any(str(phrase).strip().lower() in lowered for phrase in phrases if str(phrase).strip())


def query_relevance_score(row: dict, query: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    text = row_text(row)
    row_tokens = tokenize(text)
    overlap = len(query_tokens & row_tokens) / len(query_tokens)

    category_text = ", ".join(category_labels(row)).lower()
    food_text = str(row.get("popular_food", "")).lower()
    category_bonus = 0.15 if contains_any(category_text, query_tokens) else 0.0
    food_bonus = 0.10 if contains_any(food_text, query_tokens) else 0.0
    review_bonus = 0.05 if contains_any(str(row.get("review_snippets", "")), query_tokens) else 0.0

    return round(min(1.0, overlap + category_bonus + food_bonus + review_bonus), 4)


def profile_alignment_score(row: dict, profile: dict) -> float:
    text = row_text(row)
    category_text = ", ".join(category_labels(row)).lower()

    preferred = profile.get("preferred_cuisines", [])
    liked = profile.get("liked_foods", [])
    disliked = profile.get("disliked_foods", [])

    weights = []
    scores = []

    if preferred:
        weights.append(0.6)
        scores.append(1.0 if contains_any(category_text, preferred) else 0.0)

    if liked:
        weights.append(0.4)
        scores.append(1.0 if contains_any(text, liked) else 0.0)

    if not weights:
        score = 0.5
    else:
        score = sum(weight * value for weight, value in zip(weights, scores)) / sum(weights)

    if disliked and contains_any(text, disliked):
        score -= 0.35

    return round(max(0.0, min(1.0, score)), 4)


def grounding_quality_score(row: dict) -> float:
    category_ok = 1.0 if category_labels(row) else 0.0
    food_ok = 1.0 if str(row.get("popular_food", "")).strip() else 0.0
    review_ok = 1.0 if len(str(row.get("review_snippets", "")).strip()) >= 80 else 0.0
    volume_ok = 1.0 if float(row.get("num_reviews", 0) or 0) > 0 else 0.0
    quality = float(row.get("quality_score", 1.0) or 1.0)

    score = (
        0.25 * category_ok
        + 0.20 * food_ok
        + 0.30 * review_ok
        + 0.10 * volume_ok
        + 0.15 * max(0.0, min(1.0, quality))
    )
    return round(score, 4)


def disliked_conflict_count(results: list[dict], profile: dict) -> int:
    disliked = profile.get("disliked_foods", [])
    if not disliked:
        return 0
    return sum(1 for row in results if contains_any(row_text(row), disliked))


def average(values: Iterable[float]) -> float:
    values = list(values)
    return round(mean(values), 4) if values else 0.0


def evaluate_recommendation_set(
    results: list[dict],
    query: str,
    profile: dict,
    history: list[dict] | None = None,
) -> dict:
    history = history or []
    return {
        "n_recommendations": len(results),
        "relevance": average(query_relevance_score(row, query) for row in results),
        "profile_alignment": average(profile_alignment_score(row, profile) for row in results),
        "category_diversity": category_diversity_ratio(results),
        "category_entropy": category_entropy(results),
        "novelty": novelty_ratio(results, history),
        "grounding_quality": average(grounding_quality_score(row) for row in results),
        "disliked_conflicts": disliked_conflict_count(results, profile),
    }


def summarize_by_mode(rows: list[dict], metric_names: list[str]) -> dict[str, dict[str, float]]:
    summary = {}
    modes = sorted({row["mode"] for row in rows})
    for mode in modes:
        subset = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            metric: average(float(row[metric]) for row in subset)
            for metric in metric_names
        }
    return summary


def recommendation_names_for_report(results: list[dict]) -> str:
    return "; ".join(restaurant_name(row) for row in results)


def primary_categories_for_report(results: list[dict]) -> str:
    return "; ".join(primary_category(row) for row in results)
