"""
Reddit grounding helpers for Food AI Companion.

This Milestone 3 artifact supports Bilal's contribution by adding a small,
auditable external community-signal layer without changing the working
Streamlit app. It uses Reddit/PRAW only when local credentials are provided.
"""

from __future__ import annotations

import csv
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_QUERY_CASES: tuple[tuple[str, str], ...] = (
    ("italian_pasta_budget", "cheap Italian pasta restaurant NYC"),
    ("korean_casual", "casual Korean restaurant NYC generous portions"),
    ("japanese_date_night", "Japanese restaurant NYC date night not too expensive"),
    ("open_dinner_recommendation", "best dinner restaurant recommendation NYC"),
)

DEFAULT_SUBREDDITS: tuple[str, ...] = ("FoodNYC", "AskNYC", "nyc")

CUISINE_KEYWORDS: tuple[str, ...] = (
    "american",
    "bar",
    "bbq",
    "burger",
    "chinese",
    "cocktail",
    "dessert",
    "halal",
    "indian",
    "italian",
    "japanese",
    "korean",
    "mediterranean",
    "mexican",
    "pizza",
    "ramen",
    "seafood",
    "sushi",
    "thai",
    "vegetarian",
    "vegan",
)

GENERIC_RESTAURANT_TOKENS = {
    "bar",
    "cafe",
    "deli",
    "grill",
    "kitchen",
    "pizza",
    "restaurant",
    "sushi",
}


class RedditGroundingError(RuntimeError):
    """Raised when live Reddit collection cannot run."""


@dataclass(frozen=True)
class RedditSearchConfig:
    query_cases: tuple[tuple[str, str], ...] = DEFAULT_QUERY_CASES
    subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS
    limit_per_query: int = 8
    sort: str = "relevance"
    time_filter: str = "year"


@dataclass(frozen=True)
class RedditPostRecord:
    case_id: str
    query: str
    subreddit: str
    reddit_id: str
    title: str
    selftext: str
    comment_snippets: str
    score: int
    num_comments: int
    upvote_ratio: float | None
    created_utc: float | None
    permalink: str
    url: str

    @property
    def source_text(self) -> str:
        return " ".join([self.title, self.selftext, self.comment_snippets])

    def to_row(self) -> dict[str, str | int | float]:
        created = ""
        if self.created_utc:
            created = datetime.fromtimestamp(self.created_utc, tz=timezone.utc).isoformat()

        return {
            "case_id": self.case_id,
            "query": self.query,
            "subreddit": self.subreddit,
            "reddit_id": self.reddit_id,
            "title": self.title,
            "selftext_snippet": compact_text(self.selftext, limit=400),
            "comment_snippets": compact_text(self.comment_snippets, limit=500),
            "score": self.score,
            "num_comments": self.num_comments,
            "upvote_ratio": "" if self.upvote_ratio is None else round(float(self.upvote_ratio), 3),
            "created_utc": created,
            "permalink": self.permalink,
            "url": self.url,
        }


@dataclass(frozen=True)
class GroundingAnalysis:
    rows: list[dict[str, str | int | float]]
    restaurant_counts: Counter
    cuisine_counts: Counter
    subreddit_counts: Counter
    case_counts: Counter


def compact_text(value: str, limit: int = 280) -> str:
    cleaned = " ".join(str(value or "").replace("\n", " ").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def missing_reddit_environment(env: dict[str, str] | None = None) -> list[str]:
    env = env or os.environ
    required = ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"]
    return [key for key in required if not str(env.get(key, "")).strip()]


def create_reddit_client(env: dict[str, str] | None = None):
    """
    Build a read-only PRAW client from environment variables.

    Required variables:
    - REDDIT_CLIENT_ID
    - REDDIT_CLIENT_SECRET
    - REDDIT_USER_AGENT
    """
    env = env or os.environ
    missing = missing_reddit_environment(env)
    if missing:
        raise RedditGroundingError(
            "Missing Reddit environment variable(s): " + ", ".join(missing)
        )

    try:
        import praw
    except ImportError as exc:
        raise RedditGroundingError(
            "PRAW is not installed. Run `pip install -r src/requirements.txt` first."
        ) from exc

    reddit = praw.Reddit(
        client_id=env["REDDIT_CLIENT_ID"],
        client_secret=env["REDDIT_CLIENT_SECRET"],
        user_agent=env["REDDIT_USER_AGENT"],
    )
    reddit.read_only = True
    return reddit


def load_known_restaurant_names(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return []

    names: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = row.get("title") or row.get("name") or row.get("restaurant_name")
            name = " ".join(str(raw_name or "").split())
            normalized = normalize_text(name)
            tokens = normalized.split()
            if not normalized:
                continue
            if normalized in GENERIC_RESTAURANT_TOKENS:
                continue
            if len(normalized) < 6 and len(tokens) == 1:
                continue
            names.add(name)

    return sorted(names, key=lambda item: len(item), reverse=True)


def match_known_restaurants(text: str, known_names: Iterable[str], max_matches: int = 8) -> list[str]:
    normalized = f" {normalize_text(text)} "
    matches = []
    seen = set()
    for name in known_names:
        key = normalize_text(name)
        if not key or key in seen:
            continue
        if f" {key} " in normalized:
            matches.append(name)
            seen.add(key)
        if len(matches) >= max_matches:
            break
    return matches


def extract_cuisine_terms(text: str) -> list[str]:
    normalized = f" {normalize_text(text)} "
    matches = [
        term
        for term in CUISINE_KEYWORDS
        if f" {normalize_text(term)} " in normalized
    ]
    return sorted(set(matches))


def _comment_snippets(submission, limit: int) -> str:
    if limit <= 0:
        return ""
    try:
        submission.comment_sort = "top"
        submission.comments.replace_more(limit=0)
        comments = [
            compact_text(getattr(comment, "body", ""), limit=220)
            for comment in submission.comments[:limit]
            if getattr(comment, "body", "")
        ]
        return " || ".join(comments)
    except Exception:
        return ""


def collect_reddit_posts(
    reddit,
    config: RedditSearchConfig,
    include_comments: bool = False,
    comment_limit: int = 2,
) -> tuple[list[RedditPostRecord], list[str]]:
    records: list[RedditPostRecord] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for case_id, query in config.query_cases:
        for subreddit_name in config.subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                submissions = subreddit.search(
                    query,
                    sort=config.sort,
                    time_filter=config.time_filter,
                    syntax="lucene",
                    limit=config.limit_per_query,
                )
                for submission in submissions:
                    reddit_id = str(getattr(submission, "id", ""))
                    if not reddit_id or reddit_id in seen_ids:
                        continue
                    seen_ids.add(reddit_id)
                    records.append(
                        RedditPostRecord(
                            case_id=case_id,
                            query=query,
                            subreddit=str(getattr(submission, "subreddit", subreddit_name)),
                            reddit_id=reddit_id,
                            title=str(getattr(submission, "title", "")),
                            selftext=str(getattr(submission, "selftext", "")),
                            comment_snippets=(
                                _comment_snippets(submission, comment_limit)
                                if include_comments
                                else ""
                            ),
                            score=int(getattr(submission, "score", 0) or 0),
                            num_comments=int(getattr(submission, "num_comments", 0) or 0),
                            upvote_ratio=getattr(submission, "upvote_ratio", None),
                            created_utc=getattr(submission, "created_utc", None),
                            permalink="https://www.reddit.com"
                            + str(getattr(submission, "permalink", "")),
                            url=str(getattr(submission, "url", "")),
                        )
                    )
            except Exception as exc:
                errors.append(f"{case_id}/{subreddit_name}: {type(exc).__name__}: {exc}")

    return records, errors


def analyze_reddit_grounding(
    records: list[RedditPostRecord],
    known_restaurant_names: Iterable[str],
) -> GroundingAnalysis:
    rows = []
    restaurant_counts: Counter = Counter()
    cuisine_counts: Counter = Counter()
    subreddit_counts: Counter = Counter()
    case_counts: Counter = Counter()

    for record in records:
        matched_restaurants = match_known_restaurants(
            record.source_text,
            known_restaurant_names,
        )
        cuisine_terms = extract_cuisine_terms(record.source_text)

        restaurant_counts.update(matched_restaurants)
        cuisine_counts.update(cuisine_terms)
        subreddit_counts.update([record.subreddit])
        case_counts.update([record.case_id])

        rows.append(
            {
                **record.to_row(),
                "matched_restaurants": "; ".join(matched_restaurants),
                "cuisine_terms": "; ".join(cuisine_terms),
            }
        )

    return GroundingAnalysis(
        rows=rows,
        restaurant_counts=restaurant_counts,
        cuisine_counts=cuisine_counts,
        subreddit_counts=subreddit_counts,
        case_counts=case_counts,
    )


def markdown_table(headers: list[str], rows: list[list[str]], empty_message: str) -> list[str]:
    if not rows:
        return [empty_message]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        escaped = [str(value).replace("|", "/") for value in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return lines


def build_summary_markdown(
    analysis: GroundingAnalysis,
    config: RedditSearchConfig,
    csv_output: Path,
    collection_errors: list[str],
    setup_notes: list[str] | None = None,
) -> str:
    setup_notes = setup_notes or []
    lines = [
        "# Reddit Grounding Artifact",
        "",
        "This Milestone 3 artifact supports Bilal's external-grounding contribution by collecting public Reddit discussion signals for Food AI Companion restaurant queries.",
        "",
        "It is a supplementary analysis artifact, not a live ranking layer in the Streamlit app. Reddit mentions are treated as noisy community signals and are not presented as ground truth.",
        "",
        "## Collection Status",
        "",
    ]

    if setup_notes:
        lines.extend(["Live Reddit collection was not run.", ""])
        lines.extend(f"- {note}" for note in setup_notes)
    else:
        lines.extend(
            [
                f"- Posts collected: {len(analysis.rows)}",
                f"- Query cases: {len(config.query_cases)}",
                f"- Subreddits searched: {', '.join(config.subreddits)}",
                f"- Rows written to: `{csv_output}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Search public Reddit posts for fixed restaurant-intent queries aligned with the Milestone 2/3 evaluation cases.",
            "- Store only post metadata, snippets, URLs, matched local restaurant names, and cuisine keywords.",
            "- Match restaurant names against the local `data/restaurants.csv` inventory to avoid inventing unsupported restaurant entities.",
            "- Use the artifact for final-report discussion of external community grounding, not as fabricated evaluation scores.",
            "",
            "## Query Coverage",
            "",
        ]
    )

    query_rows = [
        [case_id, str(analysis.case_counts.get(case_id, 0)), query]
        for case_id, query in config.query_cases
    ]
    lines.extend(markdown_table(["Case", "Posts", "Query"], query_rows, "No query rows collected."))

    lines.extend(["", "## Top Cuisine Signals", ""])
    cuisine_rows = [[term, str(count)] for term, count in analysis.cuisine_counts.most_common(12)]
    lines.extend(markdown_table(["Cuisine/food term", "Mentions"], cuisine_rows, "No cuisine terms collected yet."))

    lines.extend(["", "## Top Local Restaurant Mentions", ""])
    restaurant_rows = [[name, str(count)] for name, count in analysis.restaurant_counts.most_common(12)]
    lines.extend(markdown_table(["Restaurant matched in local data", "Mentions"], restaurant_rows, "No local restaurant matches collected yet."))

    lines.extend(["", "## Subreddit Mix", ""])
    subreddit_rows = [[name, str(count)] for name, count in analysis.subreddit_counts.most_common()]
    lines.extend(markdown_table(["Subreddit", "Posts"], subreddit_rows, "No subreddit rows collected yet."))

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Reddit discussions are informal, biased toward active posters, and may include outdated recommendations.",
            "- Name matching is conservative and only counts restaurants already present in the local restaurant dataset.",
            "- This artifact does not replace Google Places, RAG retrieval, or human evaluation.",
            "- The CSV may be empty if Reddit credentials are not configured; that is intentional and avoids fake results.",
        ]
    )

    if collection_errors:
        lines.extend(["", "## Collection Warnings", ""])
        lines.extend(f"- {error}" for error in collection_errors[:20])

    lines.extend(
        [
            "",
            "## Milestone 3 Report Note",
            "",
            "Bilal added a Reddit/PRAW grounding artifact that collects public community discussion signals, maps them back to local restaurant metadata, and produces report-ready CSV/Markdown outputs without changing the existing Eat Out or Cook at Home app flow.",
            "",
        ]
    )

    return "\n".join(lines)
