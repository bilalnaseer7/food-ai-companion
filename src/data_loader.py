import pandas as pd


COLUMN_ALIASES = {
    "Title": "title",
    "Number of review": "num_reviews",
    "Catagory": "category",
    "Category": "category",
    "Reveiw Comment": "review_text",
    "Review Comment": "review_text",
    "Reviw Comment": "review_text",
    "Popular food": "popular_food",
    "Online Order": "online_order",
}


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace('"', "").strip()


def _normalize_online_order(value: str) -> str:
    value = value.strip().title()
    if value not in {"Yes", "No"}:
        return "Unknown"
    return value


def _split_list_field(value: str) -> list[str]:
    if not value:
        return []
    parts = [x.strip() for x in value.split(",")]
    return [x for x in parts if x and x.lower() != "no"]


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def load_reviews(path: str = "data/restaurants.csv", max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Clean raw column names first
    df.columns = [str(col).strip() for col in df.columns]

    rename_dict = {}
    for col in df.columns:
        if col in COLUMN_ALIASES:
            rename_dict[col] = COLUMN_ALIASES[col]

    df = df.rename(columns=rename_dict)

    required_cols = ["title", "num_reviews", "category", "review_text", "popular_food", "online_order"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected cleaned columns in dataset: {missing_cols}")

    df = df[required_cols].copy()

    # Clean fields
    text_cols = ["title", "category", "review_text", "popular_food", "online_order"]
    for col in text_cols:
        df[col] = df[col].apply(_clean_text)

    df["num_reviews"] = (
        df["num_reviews"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
    )
    df["num_reviews"] = pd.to_numeric(df["num_reviews"], errors="coerce").fillna(0).astype(int)

    df["online_order"] = df["online_order"].apply(_normalize_online_order)

    # Remove empty review rows
    df = df[df["review_text"] != ""].copy()

    # Deduplicate at row level first
    df = df.drop_duplicates(subset=["title", "review_text"]).reset_index(drop=True)

    # Light parsing for category and food fields
    df["category_list"] = df["category"].apply(_split_list_field)
    df["popular_food_list"] = df["popular_food"].apply(_split_list_field)

    # ---- Layer 1: restaurant-level aggregation ----
    grouped_rows = []

    for title, group in df.groupby("title", sort=False):
        category_items = []
        food_items = []
        review_items = []
        online_values = []

        for _, row in group.iterrows():
            category_items.extend(row["category_list"])
            food_items.extend(row["popular_food_list"])
            online_values.append(row["online_order"])

            review_text = row["review_text"]
            if review_text:
                review_items.append(review_text)

        categories = _unique_preserve_order(category_items)
        popular_foods = _unique_preserve_order(food_items)

        # Keep only a few review snippets so retrieval context stays focused
        review_items = _unique_preserve_order(review_items)[:3]

        # Use the max review count seen for that restaurant
        max_reviews = int(group["num_reviews"].max())

        # Majority vote on online ordering
        yes_count = sum(1 for x in online_values if x == "Yes")
        no_count = sum(1 for x in online_values if x == "No")
        if yes_count > no_count:
            online_order = "Yes"
        elif no_count > yes_count:
            online_order = "No"
        else:
            online_order = online_values[0] if online_values else "Unknown"

        category_str = ", ".join(categories) if categories else "Unknown"
        popular_food_str = ", ".join(popular_foods[:5]) if popular_foods else "Unknown"

        snippet_parts = []
        for idx, review in enumerate(review_items, start=1):
            snippet_parts.append(f"Review {idx}: {review}")
        review_snippets = " ".join(snippet_parts)

        combined_text = (
            f"{title} is a restaurant in New York City. "
            f"Category: {category_str}. "
            f"Popular foods: {popular_food_str}. "
            f"Online ordering: {online_order}. "
            f"Total reviews: {max_reviews}. "
            f"{review_snippets}"
        )

        grouped_rows.append(
            {
                "title": title,
                "num_reviews": max_reviews,
                "category": category_str,
                "popular_food": popular_food_str,
                "online_order": online_order,
                "review_snippets": review_snippets,
                "city": "New York City",
                "combined_text": combined_text,
            }
        )

    restaurant_df = pd.DataFrame(grouped_rows)

    # Prioritize stronger restaurants before optional row truncation
    restaurant_df = restaurant_df.sort_values(
        by=["num_reviews", "title"],
        ascending=[False, True]
    ).reset_index(drop=True)

    if max_rows is not None:
        restaurant_df = restaurant_df.head(max_rows).copy()

    return restaurant_df

