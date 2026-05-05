import math
import re
from collections import Counter
from typing import Iterable


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def restaurant_name(row: dict) -> str:
    return str(row.get("title") or row.get("name") or "").strip()


def category_labels(row: dict) -> list[str]:
    raw = row.get("category") or row.get("categories") or ""
    if isinstance(raw, list):
        labels = raw
    else:
        labels = str(raw).split(",")

    cleaned = []
    seen = set()
    for label in labels:
        value = " ".join(str(label).strip().split())
        key = value.lower()
        if value and key not in seen:
            cleaned.append(value)
            seen.add(key)
    return cleaned


def primary_category(row: dict) -> str:
    labels = category_labels(row)
    return labels[0] if labels else "Unknown"


def recommendation_names(results: Iterable[dict]) -> list[str]:
    return [restaurant_name(row) for row in results if restaurant_name(row)]


def name_overlap_ratio(results: list[dict], history: list[dict]) -> float:
    result_names = {normalize_name(name) for name in recommendation_names(results)}
    history_names = {normalize_name(name) for name in recommendation_names(history)}
    if not result_names:
        return 0.0
    return round(len(result_names & history_names) / len(result_names), 4)


def category_overlap_ratio(results: list[dict], history: list[dict]) -> float:
    result_categories = {primary_category(row).lower() for row in results}
    history_categories = {primary_category(row).lower() for row in history}
    result_categories.discard("unknown")
    history_categories.discard("unknown")
    if not result_categories:
        return 0.0
    return round(len(result_categories & history_categories) / len(result_categories), 4)


def novelty_ratio(results: list[dict], history: list[dict]) -> float:
    return round(1.0 - name_overlap_ratio(results, history), 4)


def category_diversity_ratio(results: list[dict]) -> float:
    if not results:
        return 0.0
    unique_categories = {primary_category(row).lower() for row in results}
    unique_categories.discard("unknown")
    return round(len(unique_categories) / len(results), 4)


def category_entropy(results: list[dict]) -> float:
    categories = [primary_category(row).lower() for row in results if primary_category(row)]
    if not categories:
        return 0.0

    counts = Counter(categories)
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return round(max(0.0, entropy / max_entropy), 4)


def filter_bubble_index(results: list[dict], history: list[dict]) -> float:
    """Estimate repetition risk from exact restaurant overlap and cuisine/category overlap."""
    name_overlap = name_overlap_ratio(results, history)
    category_overlap = category_overlap_ratio(results, history)
    concentration = 1.0 - category_diversity_ratio(results)
    score = (0.5 * name_overlap) + (0.3 * category_overlap) + (0.2 * concentration)
    return round(max(0.0, min(1.0, score)), 4)


def _contains_any(text: str, phrases: list[str]) -> bool:
    lowered = str(text or "").lower()
    return any(str(phrase).strip().lower() in lowered for phrase in phrases if str(phrase).strip())


def profile_alignment_ratio(results: list[dict], profile: dict) -> float:
    if not results:
        return 0.0

    preferred = [str(x).lower() for x in profile.get("preferred_cuisines", [])]
    liked = [str(x).lower() for x in profile.get("liked_foods", [])]
    disliked = [str(x).lower() for x in profile.get("disliked_foods", [])]

    aligned = 0
    for row in results:
        category_text = ", ".join(category_labels(row)).lower()
        full_text = " ".join(
            [
                restaurant_name(row),
                category_text,
                str(row.get("popular_food", "")),
                str(row.get("review_snippets", "")),
                str(row.get("review_text", "")),
            ]
        ).lower()

        positive_match = not preferred and not liked
        if preferred and _contains_any(category_text, preferred):
            positive_match = True
        elif not preferred and liked and _contains_any(full_text, liked):
            positive_match = True

        negative_match = disliked and _contains_any(full_text, disliked)
        if positive_match and not negative_match:
            aligned += 1

    return round(aligned / len(results), 4)


def summarize_metrics(results: list[dict], history: list[dict], profile: dict) -> dict:
    return {
        "n_recommendations": len(results),
        "name_overlap": name_overlap_ratio(results, history),
        "category_overlap": category_overlap_ratio(results, history),
        "novelty": novelty_ratio(results, history),
        "category_diversity": category_diversity_ratio(results),
        "category_entropy": category_entropy(results),
        "filter_bubble_index": filter_bubble_index(results, history),
        "profile_alignment": profile_alignment_ratio(results, profile),
    }


def diversity_rerank(
    results: list[dict],
    history: list[dict] | None = None,
    top_k: int = 5,
    diversity_weight: float = 0.35,
    novelty_weight: float = 0.25,
) -> list[dict]:
    selected = []
    remaining = list(results)
    used_categories = set()
    history = history or []
    history_names = {normalize_name(restaurant_name(row)) for row in history}
    history_categories = {primary_category(row).lower() for row in history}

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = float("-inf")

        for idx, row in enumerate(remaining):
            base_score = float(row.get("analysis_score", row.get("retrieval_score", 0.0)))
            category = primary_category(row).lower()
            name = normalize_name(restaurant_name(row))
            diversity_bonus = diversity_weight if category not in used_categories else 0.0
            history_category_bonus = (diversity_weight * 0.5) if category not in history_categories else 0.0
            novelty_bonus = novelty_weight if name not in history_names else 0.0
            score = base_score + diversity_bonus + history_category_bonus + novelty_bonus
            if score > best_score:
                best_idx = idx
                best_score = score

        chosen = remaining.pop(best_idx).copy()
        chosen["diversity_adjusted_score"] = round(best_score, 4)
        selected.append(chosen)
        used_categories.add(primary_category(chosen).lower())

    return selected
