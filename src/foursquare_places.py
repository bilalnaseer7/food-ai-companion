import os
import requests
from typing import Optional

FSQ_API_KEY = os.getenv("FOURSQUARE_API_KEY")
FSQ_SEARCH_URL = "https://api.foursquare.com/v3/places/search"
FSQ_TIPS_URL   = "https://api.foursquare.com/v3/places/{fsq_id}/tips"

# Foursquare category ID for all F and B
DINING_CATEGORY_ID = "13000"

BOROUGH_NEAR = {
    "manhattan":     "Manhattan, New York, NY",
    "brooklyn":      "Brooklyn, New York, NY",
    "queens":        "Queens, New York, NY",
    "bronx":         "The Bronx, New York, NY",
    "staten island": "Staten Island, New York, NY",
}

PRICE_LABEL = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}

PRICE_SENSITIVITY_MAP = {
    "budget":   1,
    "moderate": 2,
    "premium":  3,
}

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {FSQ_API_KEY}",
        "Accept": "application/json",
    }


def search_restaurants(
    query: str,
    borough: str = "manhattan",
    price: Optional[int] = None,
    open_now: bool = False,
    limit: int = 8,
) -> list[dict]:
    """
    Search for restaurants in a specific NYC borough matching the query.

    Args:
        query:    Natural language search string (e.g. "cozy ramen", "cheap tacos")
        borough:  One of the 5 boroughs (defaults to manhattan)
        price:    Foursquare price tier 1-4, or None for no filter
        open_now: Only return currently open venues
        limit:    Max results (Foursquare cap is 50)
    """
    near = BOROUGH_NEAR.get(borough.lower(), "Manhattan, New York, NY")

    params = {
        "query":      query,
        "near":       near,
        "categories": DINING_CATEGORY_ID,
        "limit":      limit,
        "fields":     "fsq_id,name,location,categories,price,rating,hours,stats",
    }

    if open_now:
        params["open_now"] = "true"

    if price is not None:
        params["price"] = str(price)

    response = requests.get(FSQ_SEARCH_URL, headers=_headers(), params=params, timeout=30)
    response.raise_for_status()

    results = response.json().get("results", [])
    return [_parse_place(r) for r in results]


def get_tips(fsq_id: str, limit: int = 2) -> list[str]:
    """
    Fetch user-submitted tips for a venue.
    Tips are short opinionated blurbs — good signal for the LLM.
    Sorted by agree_count so the most-validated tips come first.
    """
    url = FSQ_TIPS_URL.format(fsq_id=fsq_id)
    params = {"limit": limit, "fields": "text,agree_count"}

    try:
        response = requests.get(url, headers=_headers(), params=params, timeout=30)
        if response.status_code != 200:
            return []
        tips = response.json()
        tips_sorted = sorted(tips, key=lambda t: t.get("agree_count", 0), reverse=True)
        return [t["text"] for t in tips_sorted if t.get("text")]
    except Exception:
        return []


def _parse_place(raw: dict) -> dict:
    """Extract relevant fields from a Foursquare v3 result."""
    location   = raw.get("location", {})
    address    = location.get("formatted_address") or location.get("address", "")
    categories = [c.get("name", "") for c in raw.get("categories", [])]
    open_now   = raw.get("hours", {}).get("open_now")
    stats      = raw.get("stats", {})

    return {
        "fsq_id":     raw.get("fsq_id", ""),
        "name":       raw.get("name", "Unknown"),
        "address":    address,
        "categories": categories,
        "price":      raw.get("price"),       # int 1-4 or None
        "rating":     raw.get("rating"),      # float 0-10 (FSQ scale)
        "open_now":   open_now,
        "total_tips": stats.get("total_tips", 0),
    }


def format_for_prompt(restaurants: list[dict], fetch_tips: bool = True) -> str:
    """
    Serialize restaurant results into a prompt-ready block for the LLM.
    Fetches top user tips per venue if fetch_tips is True.
    """
    if not restaurants:
        return "No restaurants found."

    lines = ["## Nearby restaurants (Foursquare)"]

    for r in restaurants:
        price_str  = PRICE_LABEL.get(r["price"], "?")
        rating_str = f"{r['rating']}/10" if r["rating"] else "unrated"
        open_str   = " · open now" if r["open_now"] else ""
        cats       = ", ".join(r["categories"][:2]) if r["categories"] else ""

        lines.append(
            f"\n- **{r['name']}** | {price_str} | {rating_str}{open_str}"
            f"\n  {cats} · {r['address']}"
        )

        if fetch_tips and r["fsq_id"]:
            for tip in get_tips(r["fsq_id"]):
                lines.append(f'  > "{tip}"')

    return "\n".join(lines)


def price_sensitivity_to_tier(sensitivity: str) -> Optional[int]:
    """Map taste profile price sensitivity to a Foursquare price tier int."""
    return PRICE_SENSITIVITY_MAP.get(sensitivity)
