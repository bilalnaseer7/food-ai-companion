import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_reviews
from src.filter_bubble import (
    category_labels,
    diversity_rerank,
    primary_category,
    restaurant_name,
    summarize_metrics,
)


PROFILES = {
    "italian_pasta": {
        "preferred_cuisines": ["Italian", "Pizza"],
        "liked_foods": ["pasta", "pizza", "tomato sauce"],
        "disliked_foods": ["seafood"],
        "budget": "moderate",
        "online_order": "Yes",
        "occasion": "casual dinner",
    },
    "japanese_date_night": {
        "preferred_cuisines": ["Japanese", "Sushi"],
        "liked_foods": ["sushi", "ramen", "omakase"],
        "disliked_foods": ["pizza"],
        "budget": "moderate",
        "online_order": "No",
        "occasion": "date night",
    },
    "exploratory_low_bubble": {
        "preferred_cuisines": ["Japanese", "Korean", "Mexican", "Mediterranean"],
        "liked_foods": ["noodles", "rice", "tacos", "fresh herbs"],
        "disliked_foods": ["heavy cream"],
        "budget": "moderate",
        "online_order": "No",
        "occasion": "try something new",
    },
}


QUERIES = [
    "I want something good for dinner in NYC.",
    "Find a casual but memorable place for tonight.",
    "Recommend something flavorful but not too expensive.",
]


def _contains_any(text: str, phrases: list[str]) -> bool:
    lowered = str(text or "").lower()
    return any(str(phrase).lower() in lowered for phrase in phrases if str(phrase).strip())


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


def score_restaurant(row: dict, profile: dict, query: str = "") -> float:
    text = row_text(row)
    category_text = ", ".join(category_labels(row)).lower()
    score = 0.0

    preferred = profile.get("preferred_cuisines", [])
    liked = profile.get("liked_foods", [])

    has_preferred_category = _contains_any(category_text, preferred)
    has_liked_signal = _contains_any(text, liked)

    if has_preferred_category:
        score += 0.45
    elif preferred:
        score -= 0.18

    if has_liked_signal and has_preferred_category:
        score += 0.30
    elif has_liked_signal:
        score += 0.08

    if _contains_any(text, profile.get("disliked_foods", [])):
        score -= 0.50

    budget = str(profile.get("budget", "")).lower()
    if budget == "budget" and _contains_any(text, ["cheap", "value", "affordable", "reasonable"]):
        score += 0.12
    elif budget == "moderate" and _contains_any(text, ["casual", "reasonable", "solid value"]):
        score += 0.08

    online_pref = str(profile.get("online_order", "")).strip().lower()
    if online_pref in {"yes", "no"} and str(row.get("online_order", "")).strip().lower() == online_pref:
        score += 0.08

    if query and _contains_any(text, query.split()):
        score += 0.05

    score += min(float(row.get("num_reviews", 0)) / 15000.0, 0.10)
    score += (float(row.get("quality_score", 1.0)) - 0.5) * 0.10
    return round(score, 4)


def rank_restaurants(df, profile: dict, query: str, limit: int = 15) -> list[dict]:
    scored = []
    seen = set()

    for row in df.to_dict("records"):
        name = restaurant_name(row)
        if not name or name in seen:
            continue
        seen.add(name)
        item = dict(row)
        item["analysis_score"] = score_restaurant(item, profile, query)
        scored.append(item)

    scored.sort(key=lambda row: row["analysis_score"], reverse=True)
    return scored[:limit]


def recommendation_table(results: list[dict]) -> str:
    lines = ["| Rank | Restaurant | Category | Popular food | Analysis score |", "|---:|---|---|---|---:|"]
    for idx, row in enumerate(results, start=1):
        lines.append(
            f"| {idx} | {restaurant_name(row)} | {primary_category(row)} | "
            f"{row.get('popular_food', '') or 'n/a'} | {row.get('analysis_score', row.get('diversity_adjusted_score', 0))} |"
        )
    return "\n".join(lines)


def metrics_rows(df, top_k: int) -> tuple[list[dict], list[str]]:
    rows = []
    detail_sections = []

    for profile_name, profile in PROFILES.items():
        history_pool = rank_restaurants(df, profile, "previous accepted favorites", limit=top_k)

        for query in QUERIES:
            personalized_pool = rank_restaurants(df, profile, query, limit=max(75, top_k * 15))
            personalized = personalized_pool[:top_k]
            diversified = diversity_rerank(
                personalized_pool,
                history=history_pool,
                top_k=top_k,
                diversity_weight=0.45,
                novelty_weight=0.30,
            )

            for mode, results in [
                ("profile_proxy", personalized),
                ("profile_proxy_diversity_constraint", diversified),
            ]:
                metrics = summarize_metrics(results, history_pool, profile)
                row = {
                    "profile": profile_name,
                    "query": query,
                    "mode": mode,
                    **metrics,
                    "recommended_restaurants": "; ".join(restaurant_name(item) for item in results),
                    "primary_categories": "; ".join(primary_category(item) for item in results),
                }
                rows.append(row)

            detail_sections.append(
                "\n".join(
                    [
                        f"## {profile_name} - {query}",
                        "",
                        "### Simulated accepted-history set",
                        recommendation_table(history_pool),
                        "",
                        "### Profile-personalized proxy recommendations",
                        recommendation_table(personalized),
                        "",
                        "### Diversity-constrained recommendations",
                        recommendation_table(diversified),
                        "",
                    ]
                )
            )

    return rows, detail_sections


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict], detail_sections: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    avg_by_mode = {}
    for mode in sorted({row["mode"] for row in rows}):
        subset = [row for row in rows if row["mode"] == mode]
        avg_by_mode[mode] = {
            "filter_bubble_index": sum(float(row["filter_bubble_index"]) for row in subset) / len(subset),
            "novelty": sum(float(row["novelty"]) for row in subset) / len(subset),
            "category_diversity": sum(float(row["category_diversity"]) for row in subset) / len(subset),
            "profile_alignment": sum(float(row["profile_alignment"]) for row in subset) / len(subset),
        }

    lines = [
        "# Personalization vs Filter Bubble Analysis",
        "",
        "This analysis supports Bilal's Milestone 3 contribution. It uses the cleaned local restaurant metadata to compare a profile-personalized recommendation proxy against the same candidate pool with a diversity constraint.",
        "",
        "Important limitation: this is a deterministic metadata analysis, not a human evaluation and not an LLM quality score. The accepted-history set is simulated from each profile's strongest local matches so the filter-bubble metric can be computed reproducibly without inventing user survey scores.",
        "",
        "## Metric Definitions",
        "",
        "- `name_overlap`: fraction of current recommendations that exactly repeat simulated accepted restaurants.",
        "- `category_overlap`: fraction of current primary cuisine/category labels also present in accepted history.",
        "- `novelty`: `1 - name_overlap`; higher means fewer repeated restaurant names.",
        "- `category_diversity`: unique primary categories divided by recommendation count.",
        "- `category_entropy`: normalized spread of primary categories.",
        "- `filter_bubble_index`: weighted repetition-risk estimate using name overlap, category overlap, and category concentration.",
        "- `profile_alignment`: fraction of recommendations matching preferred cuisines/liked foods while avoiding disliked foods.",
        "",
        "## Average Metrics by Mode",
        "",
        "| Mode | Filter bubble index | Novelty | Category diversity | Profile alignment |",
        "|---|---:|---:|---:|---:|",
    ]

    for mode, metrics in avg_by_mode.items():
        lines.append(
            f"| {mode} | {metrics['filter_bubble_index']:.4f} | {metrics['novelty']:.4f} | "
            f"{metrics['category_diversity']:.4f} | {metrics['profile_alignment']:.4f} |"
        )

    lines.extend(["", "## Recommendation Sets", ""])
    lines.extend(detail_sections)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run filter-bubble analysis for Milestone 3.")
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--csv-output", type=Path, default=ROOT / "results" / "filter_bubble_metrics.csv")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "results" / "filter_bubble_analysis.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_reviews(path=str(ROOT / "data" / "restaurants.csv"), max_rows=args.max_rows)
    rows, detail_sections = metrics_rows(df, top_k=args.top_k)
    write_csv(rows, args.csv_output)
    write_summary(rows, detail_sections, args.summary_output)
    print(f"Wrote {args.csv_output}")
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
