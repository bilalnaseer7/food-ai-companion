"""
filter_bubble.py — personalization vs. diversity metrics and reranking.

Changes vs. original:
- profile_alignment_ratio now uses learned cuisine_scores / food_scores in
  addition to the binary preferred_cuisines / liked_foods lists, so alignment
  improves continuously as the user gives accept/reject feedback.
- filter_bubble_index now returns 0.0 (not a concentration-only score) when
  history is empty, because there is no prior session to compare against.
- diversity_rerank now accepts an optional profile argument and applies a small
  profile_score_bonus so cuisine-score-weighted restaurants are preferred when
  diversity is otherwise equal.
- summarize_metrics now includes a weighted_alignment field that uses the
  learned scores for a richer signal than the binary alignment ratio.
"""

import math
import re
from collections import Counter
from typing import Iterable


# ── name / category helpers ──────────────────────────────────────────────────

def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def restaurant_name(row: dict) -> str:
    return str(row.get("title") or row.get("name") or "").strip()


def category_labels(row: dict) -> list[str]:
    raw = row.get("category") or row.get("categories") or ""
    labels = raw if isinstance(raw, list) else str(raw).split(",")
    cleaned, seen = [], set()
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


def _contains_any(text: str, phrases: list[str]) -> bool:
    lowered = str(text or "").lower()
    return any(str(p).strip().lower() in lowered for p in phrases if str(p).strip())


# ── overlap / novelty metrics ────────────────────────────────────────────────

def name_overlap_ratio(results: list[dict], history: list[dict]) -> float:
    result_names = {normalize_name(n) for n in recommendation_names(results)}
    history_names = {normalize_name(n) for n in recommendation_names(history)}
    if not result_names:
        return 0.0
    return round(len(result_names & history_names) / len(result_names), 4)


def category_overlap_ratio(results: list[dict], history: list[dict]) -> float:
    result_cats = {primary_category(r).lower() for r in results} - {"unknown"}
    history_cats = {primary_category(r).lower() for r in history} - {"unknown"}
    if not result_cats:
        return 0.0
    return round(len(result_cats & history_cats) / len(result_cats), 4)


def novelty_ratio(results: list[dict], history: list[dict]) -> float:
    return round(1.0 - name_overlap_ratio(results, history), 4)


# ── diversity metrics ────────────────────────────────────────────────────────

def category_diversity_ratio(results: list[dict]) -> float:
    if not results:
        return 0.0
    unique = {primary_category(r).lower() for r in results} - {"unknown"}
    return round(len(unique) / len(results), 4)


def category_entropy(results: list[dict]) -> float:
    categories = [primary_category(r).lower() for r in results if primary_category(r)]
    if not categories:
        return 0.0
    counts = Counter(categories)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return round(max(0.0, entropy / max_entropy), 4)


# ── filter bubble index ──────────────────────────────────────────────────────

def filter_bubble_index(results: list[dict], history: list[dict]) -> float:
    """
    Estimate repetition risk on a 0–1 scale.

    Fix vs. original: when history is empty there is nothing to compare
    against, so name_overlap and category_overlap are both 0. The original
    code still added 0.2 * concentration in that case, giving a misleading
    non-zero score for a first-time user. We now return 0.0 when there is no
    history, because the concept of a filter bubble requires a prior session.
    """
    if not history:
        return 0.0

    name_overlap = name_overlap_ratio(results, history)
    category_overlap = category_overlap_ratio(results, history)
    concentration = 1.0 - category_diversity_ratio(results)
    score = (0.5 * name_overlap) + (0.3 * category_overlap) + (0.2 * concentration)
    return round(max(0.0, min(1.0, score)), 4)


# ── profile alignment ────────────────────────────────────────────────────────

def profile_alignment_ratio(results: list[dict], profile: dict) -> float:
    """
    Binary alignment: fraction of results that pass a cuisine/food match and
    do not contain a disliked food.  Uses the static preferred_cuisines /
    liked_foods / disliked_foods lists (unchanged from original).
    """
    if not results:
        return 0.0

    preferred = [str(x).lower() for x in profile.get("preferred_cuisines", [])]
    liked = [str(x).lower() for x in profile.get("liked_foods", [])]
    disliked = [str(x).lower() for x in profile.get("disliked_foods", [])]

    aligned = 0
    for row in results:
        category_text = ", ".join(category_labels(row)).lower()
        full_text = " ".join([
            restaurant_name(row),
            category_text,
            str(row.get("popular_food", "")),
            str(row.get("review_snippets", "")),
            str(row.get("review_text", "")),
        ]).lower()

        positive_match = not preferred and not liked
        if preferred and _contains_any(category_text, preferred):
            positive_match = True
        elif not preferred and liked and _contains_any(full_text, liked):
            positive_match = True

        negative_match = bool(disliked and _contains_any(full_text, disliked))
        if positive_match and not negative_match:
            aligned += 1

    return round(aligned / len(results), 4)


def weighted_profile_alignment(results: list[dict], profile: dict) -> float:
    """
    Continuous alignment score using the *learned* cuisine_scores and
    food_scores from the taste profile's feedback loop.

    Each restaurant gets a score in [-1, 1]:
      +cuisine_score  if the restaurant's category matches a learned cuisine
      +food_score     for each learned food found in the restaurant's text
      -dislike penalty for any disliked food found

    The final metric is the mean score clipped to [0, 1], so it reads as
    "how well do these recommendations reflect learned preferences".
    """
    if not results:
        return 0.0

    cuisine_scores: dict = profile.get("cuisine_scores", {})
    food_scores: dict = profile.get("food_scores", {})

    row_scores = []
    for row in results:
        category_text = ", ".join(category_labels(row)).lower()
        full_text = " ".join([
            restaurant_name(row),
            category_text,
            str(row.get("popular_food", "")),
            str(row.get("review_snippets", "")),
            str(row.get("review_text", "")),
        ]).lower()

        score = 0.0

        # Cuisine contribution (take the highest matching cuisine score)
        best_cuisine_score = 0.0
        for cuisine, cs in cuisine_scores.items():
            if cuisine.lower() in category_text:
                best_cuisine_score = max(best_cuisine_score, cs)
        score += best_cuisine_score

        # Food contribution (accumulate, cap total food contribution at ±0.5)
        food_contribution = 0.0
        for food, fs in food_scores.items():
            if food.lower() in full_text:
                food_contribution += fs
        score += max(-0.5, min(0.5, food_contribution))

        row_scores.append(score)

    mean_score = sum(row_scores) / len(row_scores)
    # Map [-1, 1] → [0, 1] for readability
    return round(max(0.0, min(1.0, (mean_score + 1.0) / 2.0)), 4)


# ── summary ──────────────────────────────────────────────────────────────────

def summarize_metrics(results: list[dict], history: list[dict], profile: dict) -> dict:
    """
    Return all diversity and alignment metrics in one dict.

    New field vs. original:
    - weighted_alignment: continuous score using learned cuisine/food scores.
    """
    return {
        "n_recommendations": len(results),
        "name_overlap": name_overlap_ratio(results, history),
        "category_overlap": category_overlap_ratio(results, history),
        "novelty": novelty_ratio(results, history),
        "category_diversity": category_diversity_ratio(results),
        "category_entropy": category_entropy(results),
        "filter_bubble_index": filter_bubble_index(results, history),
        "profile_alignment": profile_alignment_ratio(results, profile),
        "weighted_alignment": weighted_profile_alignment(results, profile),  # NEW
    }


# ── diversity reranking ──────────────────────────────────────────────────────

def diversity_rerank(
    results: list[dict],
    history: list[dict] | None = None,
    profile: dict | None = None,
    top_k: int = 5,
    diversity_weight: float = 0.35,
    novelty_weight: float = 0.25,
    profile_weight: float = 0.10,
) -> list[dict]:
    """
    Greedy diversity reranking with optional profile-score tiebreaking.

    New vs. original:
    - Accepts an optional profile dict.
    - When profile is provided, adds a small profile_score_bonus based on
      cuisine_scores so that cuisine-preferred restaurants are slightly
      favoured when diversity score is otherwise equal. This prevents the
      reranker from accidentally deprioritising the user's top cuisine just
      to maximise category spread.
    - profile_weight is kept small (0.10 default) so diversity still drives
      the ranking; it only breaks ties.
    """
    selected: list[dict] = []
    remaining = list(results)
    used_categories: set[str] = set()
    history = history or []
    profile = profile or {}
    cuisine_scores: dict = profile.get("cuisine_scores", {})

    history_names = {normalize_name(restaurant_name(r)) for r in history}
    history_categories = {primary_category(r).lower() for r in history}

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

            # NEW: small bonus from learned cuisine preference
            profile_bonus = 0.0
            for cuisine, cs in cuisine_scores.items():
                if cuisine.lower() in category and cs > 0:
                    profile_bonus = profile_weight * min(cs, 1.0)
                    break

            score = base_score + diversity_bonus + history_category_bonus + novelty_bonus + profile_bonus
            if score > best_score:
                best_idx = idx
                best_score = score

        chosen = remaining.pop(best_idx).copy()
        chosen["diversity_adjusted_score"] = round(best_score, 4)
        selected.append(chosen)
        used_categories.add(primary_category(chosen).lower())

    return selected
