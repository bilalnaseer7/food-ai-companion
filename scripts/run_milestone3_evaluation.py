import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_reviews
from src.evaluation import (
    contains_any,
    evaluate_recommendation_set,
    grounding_quality_score,
    primary_categories_for_report,
    profile_alignment_score,
    query_relevance_score,
    recommendation_names_for_report,
    row_text,
    summarize_by_mode,
)
from src.filter_bubble import category_labels, primary_category, restaurant_name


EVALUATION_CASES = [
    {
        "case_id": "italian_pasta_budget",
        "query": "I want cheap Italian food in NYC with good pasta and online ordering.",
        "profile": {
            "preferred_cuisines": ["Italian", "Pizza"],
            "liked_foods": ["pasta", "pizza"],
            "disliked_foods": ["seafood"],
            "budget": "cheap",
            "online_order": "Yes",
            "occasion": "casual dinner",
            "city": "New York City",
        },
    },
    {
        "case_id": "korean_casual",
        "query": "I am looking for a casual Korean restaurant in NYC with generous portions.",
        "profile": {
            "preferred_cuisines": ["Korean"],
            "liked_foods": ["bbq", "noodles", "rice"],
            "disliked_foods": ["seafood"],
            "budget": "moderate",
            "online_order": "No",
            "occasion": "casual dinner",
            "city": "New York City",
        },
    },
    {
        "case_id": "japanese_date_night",
        "query": "Find me a date-night Japanese restaurant in NYC that is not too expensive.",
        "profile": {
            "preferred_cuisines": ["Japanese", "Sushi"],
            "liked_foods": ["sushi", "ramen"],
            "disliked_foods": ["pizza"],
            "budget": "moderate",
            "online_order": "No",
            "occasion": "date night",
            "city": "New York City",
        },
    },
    {
        "case_id": "open_dinner_profile_comparison",
        "query": "I want something good for dinner in NYC.",
        "profile": {
            "preferred_cuisines": ["Mexican", "Japanese", "Korean"],
            "liked_foods": ["tacos", "sushi", "rice", "spicy food"],
            "disliked_foods": ["heavy cream"],
            "budget": "moderate",
            "online_order": "No",
            "occasion": "casual dinner",
            "city": "New York City",
        },
    },
]


MODE_LABELS = {
    "baseline_query_only_proxy": "Baseline LLM target behavior",
    "taste_profile_proxy": "LLM + taste profile target behavior",
    "taste_profile_rag_proxy": "LLM + taste profile + RAG target behavior",
}


METRIC_NAMES = [
    "relevance",
    "profile_alignment",
    "category_diversity",
    "category_entropy",
    "novelty",
    "grounding_quality",
]


def _budget_fit(row: dict, profile: dict) -> float:
    budget = str(profile.get("budget", "")).lower()
    text = row_text(row)
    if budget in {"cheap", "budget"}:
        return 1.0 if contains_any(text, ["cheap", "affordable", "budget", "value", "inexpensive"]) else 0.0
    if budget == "moderate":
        return 1.0 if contains_any(text, ["casual", "moderate", "reasonable", "solid value"]) else 0.3
    if budget in {"premium", "upscale"}:
        return 1.0 if contains_any(text, ["upscale", "fine dining", "premium", "expensive"]) else 0.0
    return 0.0


def _online_fit(row: dict, profile: dict) -> float:
    desired = str(profile.get("online_order", "")).strip().lower()
    actual = str(row.get("online_order", "")).strip().lower()
    if desired not in {"yes", "no"}:
        return 0.0
    return 1.0 if desired == actual else 0.0


def _quality_prior(row: dict) -> float:
    review_volume = min(float(row.get("num_reviews", 0) or 0) / 20000.0, 1.0)
    quality = max(0.0, min(1.0, float(row.get("quality_score", 1.0) or 1.0)))
    return (0.65 * quality) + (0.35 * review_volume)


def baseline_score(row: dict, query: str) -> float:
    return round(
        0.75 * query_relevance_score(row, query)
        + 0.25 * _quality_prior(row),
        4,
    )


def taste_profile_score(row: dict, query: str, profile: dict) -> float:
    return round(
        0.35 * query_relevance_score(row, query)
        + 0.45 * profile_alignment_score(row, profile)
        + 0.10 * _budget_fit(row, profile)
        + 0.05 * _online_fit(row, profile)
        + 0.05 * _quality_prior(row),
        4,
    )


def rag_candidate_score(row: dict, query: str, profile: dict) -> float:
    return round(
        0.30 * query_relevance_score(row, query)
        + 0.35 * profile_alignment_score(row, profile)
        + 0.25 * grounding_quality_score(row)
        + 0.05 * _budget_fit(row, profile)
        + 0.05 * _online_fit(row, profile),
        4,
    )


def rank_rows(df, score_fn, limit: int) -> list[dict]:
    scored = []
    seen_names = set()
    for row in df.to_dict("records"):
        name = restaurant_name(row)
        key = name.lower().strip()
        if not key or key in seen_names:
            continue
        seen_names.add(key)
        item = dict(row)
        item["analysis_score"] = score_fn(item)
        scored.append(item)

    scored.sort(key=lambda item: item["analysis_score"], reverse=True)
    return scored[:limit]


def accepted_history(df, profile: dict, top_k: int) -> list[dict]:
    return rank_rows(
        df,
        lambda row: taste_profile_score(row, "previous accepted favorites", profile),
        limit=top_k,
    )


def mode_recommendations(df, case: dict, top_k: int) -> dict[str, list[dict]]:
    query = case["query"]
    profile = case["profile"]

    return {
        "baseline_query_only_proxy": rank_rows(
            df,
            lambda row: baseline_score(row, query),
            limit=top_k,
        ),
        "taste_profile_proxy": rank_rows(
            df,
            lambda row: taste_profile_score(row, query, profile),
            limit=top_k,
        ),
        "taste_profile_rag_proxy": rank_rows(
            df,
            lambda row: rag_candidate_score(row, query, profile),
            limit=top_k,
        ),
    }


def profile_summary(profile: dict) -> str:
    parts = [
        f"preferred={', '.join(profile.get('preferred_cuisines', [])) or 'none'}",
        f"liked={', '.join(profile.get('liked_foods', [])) or 'none'}",
        f"disliked={', '.join(profile.get('disliked_foods', [])) or 'none'}",
        f"budget={profile.get('budget', 'n/a')}",
        f"occasion={profile.get('occasion', 'n/a')}",
    ]
    return " | ".join(parts)


def build_rows(df, top_k: int) -> tuple[list[dict], list[dict], list[str]]:
    metric_rows = []
    human_rows = []
    detail_sections = []

    for case in EVALUATION_CASES:
        query = case["query"]
        profile = case["profile"]
        history = accepted_history(df, profile, top_k=top_k)
        mode_outputs = mode_recommendations(df, case, top_k=top_k)

        detail_lines = [
            f"## {case['case_id']}",
            "",
            f"Query: {query}",
            "",
            f"Profile: {profile_summary(profile)}",
            "",
        ]

        for mode, results in mode_outputs.items():
            metrics = evaluate_recommendation_set(results, query, profile, history=history)
            metric_row = {
                "case_id": case["case_id"],
                "query": query,
                "mode": mode,
                "mode_description": MODE_LABELS[mode],
                **metrics,
                "recommended_restaurants": recommendation_names_for_report(results),
                "primary_categories": primary_categories_for_report(results),
            }
            metric_rows.append(metric_row)

            human_rows.append(
                {
                    "case_id": case["case_id"],
                    "query": query,
                    "mode": mode,
                    "mode_description": MODE_LABELS[mode],
                    "recommended_restaurants": recommendation_names_for_report(results),
                    "human_relevance_1_to_5": "",
                    "human_profile_alignment_1_to_5": "",
                    "human_grounding_1_to_5": "",
                    "human_diversity_1_to_5": "",
                    "evaluator_notes": "",
                }
            )

            detail_lines.extend(
                [
                    f"### {mode}",
                    "",
                    "| Rank | Restaurant | Primary category | Popular food | Analysis score |",
                    "|---:|---|---|---|---:|",
                ]
            )
            for rank, row in enumerate(results, start=1):
                detail_lines.append(
                    f"| {rank} | {restaurant_name(row)} | {primary_category(row)} | "
                    f"{row.get('popular_food', '') or 'n/a'} | {row.get('analysis_score', 0)} |"
                )
            detail_lines.append("")

        detail_sections.append("\n".join(detail_lines))

    return metric_rows, human_rows, detail_sections


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(metric_rows: list[dict], detail_sections: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_by_mode(metric_rows, METRIC_NAMES)

    lines = [
        "# Milestone 3 Evaluation Harness",
        "",
        "This report supports Bilal's final contribution by creating a reproducible evaluation layer for the Food AI Companion recommendation pipeline.",
        "",
        "Important limitation: the numeric metrics below are deterministic metadata-side checks over local restaurant records. They are not presented as human scores and they do not automatically grade generated language quality. The separate human-evaluation CSV should be filled using the actual outputs from `main.py` or the Streamlit demo.",
        "",
        "## Compared Modes",
        "",
        "- `baseline_query_only_proxy`: query-only recommendation behavior, used as the baseline reference.",
        "- `taste_profile_proxy`: query plus persistent taste-profile behavior.",
        "- `taste_profile_rag_proxy`: query plus taste profile plus explicit restaurant evidence quality, representing the RAG target behavior.",
        "",
        "## Metric Definitions",
        "",
        "- `relevance`: keyword and food/category match between the query and candidate restaurant metadata.",
        "- `profile_alignment`: match to preferred cuisines and liked foods, with penalties for disliked foods.",
        "- `category_diversity`: unique primary categories divided by recommendation count.",
        "- `category_entropy`: normalized category spread.",
        "- `novelty`: fraction of recommendations not repeated from a simulated accepted-history set.",
        "- `grounding_quality`: completeness of the restaurant evidence available for RAG explanation.",
        "",
        "## Average Metrics by Mode",
        "",
        "| Mode | Relevance | Profile alignment | Diversity | Entropy | Novelty | Grounding quality |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for mode, metrics in summary.items():
        lines.append(
            f"| {mode} | {metrics['relevance']:.4f} | {metrics['profile_alignment']:.4f} | "
            f"{metrics['category_diversity']:.4f} | {metrics['category_entropy']:.4f} | "
            f"{metrics['novelty']:.4f} | {metrics['grounding_quality']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## How To Use This In The Final Report",
            "",
            "Use the average-metrics table as system-side evidence that the project evaluates relevance, personalization, diversity, novelty, and grounding. Use `results/milestone3_human_eval_template.csv` for manual review of the actual generated outputs so the final report does not rely on invented evaluator scores.",
            "",
            "## Candidate Recommendation Sets",
            "",
        ]
    )
    lines.extend(detail_sections)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Milestone 3 evaluation artifacts.")
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=ROOT / "results" / "milestone3_evaluation_metrics.csv",
    )
    parser.add_argument(
        "--human-template-output",
        type=Path,
        default=ROOT / "results" / "milestone3_human_eval_template.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "milestone3_evaluation_summary.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_reviews(path=str(ROOT / "data" / "restaurants.csv"), max_rows=args.max_rows)
    metric_rows, human_rows, detail_sections = build_rows(df, top_k=args.top_k)

    write_csv(metric_rows, args.metrics_output)
    write_csv(human_rows, args.human_template_output)
    write_summary(metric_rows, detail_sections, args.summary_output)

    print(f"Wrote {args.metrics_output}")
    print(f"Wrote {args.human_template_output}")
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
