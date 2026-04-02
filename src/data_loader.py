import pandas as pd


COLUMN_MAP = {
    "Title": "title",
    "Number of review": "num_reviews",
    "Catagory": "category",
    "Reveiw Comment": "review_text",
    "Popular food": "popular_food",
    "Online Order": "online_order",
}


def load_reviews(path: str = "data/restaurants.csv", max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing_cols = [col for col in COLUMN_MAP if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in dataset: {missing_cols}")

    df = df.rename(columns=COLUMN_MAP)
    df = df[list(COLUMN_MAP.values())].copy()

    text_cols = ["title", "category", "review_text", "popular_food", "online_order"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["num_reviews"] = (
        df["num_reviews"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
    )
    df["num_reviews"] = pd.to_numeric(df["num_reviews"], errors="coerce").fillna(0).astype(int)

    df["category"] = df["category"].str.replace('"', "", regex=False)
    df["review_text"] = df["review_text"].str.replace('"', "", regex=False)
    df["popular_food"] = df["popular_food"].str.replace('"', "", regex=False)
    df["online_order"] = df["online_order"].str.replace('"', "", regex=False).str.title()

    df = df[df["review_text"] != ""].copy()
    df = df.drop_duplicates(subset=["title", "review_text"]).reset_index(drop=True)

    df["city"] = "New York City"

    df["combined_text"] = (
    df["title"] + " is a restaurant in New York City serving " + df["category"] + ". "
    + "It is known for " + df["popular_food"] + ". "
    + "Customers mention: " + df["review_text"] + ". "
    + "Online ordering is " + df["online_order"] + ". "
    + "It has " + df["num_reviews"].astype(str) + " reviews."
)

    if max_rows is not None:
        df = df.head(max_rows).copy()

    return df

