import os
import requests
from typing import Optional

GOOGLE_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
SEARCH_URL  = "https://places.googleapis.com/v1/places:searchText"
DETAILS_URL = "https://places.googleapis.com/v1/places/{place_id}"

PRICE_LABEL = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}

PRICE_SENSITIVITY_MAP = {
    "budget":   1,
    "moderate": 2,
    "premium":  3,
}

SEARCH_FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.primaryTypeDisplayName",
    "places.types",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.regularOpeningHours.openNow",
    "places.businessStatus",
])

REVIEW_FIELD_MASK = "reviews,rating"


def _api_key() -> str:
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_PLACES_API_KEY not set. "
            "Enable billing at https://console.cloud.google.com and get a key."
        )
    return GOOGLE_API_KEY


def search_restaurants(
    query: str,
    borough: str = "New York, NY",
    price: Optional[int] = None,
    open_now: bool = False,
    limit: int = 8,
) -> list[dict]:
    body = {
        "textQuery":    f"{query} restaurant in {borough}",
        "pageSize":     min(limit, 20),
        "includedType": "restaurant",
        "languageCode": "en",
    }

    if borough == "New York, NY":
        body["locationBias"] = {
            "circle": {
                "center": {"latitude": 40.7128, "longitude": -74.0060},
                "radius": 20000.0
            }
        }
    else:
        body["textQuery"] = f"{query} restaurant near {borough} New York"

    if open_now:
        body["openNow"] = True

    if price is not None:
        price_map = {
            1: "PRICE_LEVEL_INEXPENSIVE",
            2: "PRICE_LEVEL_MODERATE",
            3: "PRICE_LEVEL_EXPENSIVE",
            4: "PRICE_LEVEL_VERY_EXPENSIVE",
        }
        if price in price_map:
            body["priceLevels"] = [price_map[price]]

    headers = {
        "Content-Type":     "application/json",
        "X-Goog-Api-Key":   _api_key(),
        "X-Goog-FieldMask": SEARCH_FIELD_MASK,
    }

    try:
        r = requests.post(SEARCH_URL, json=body, headers=headers, timeout=15)
        r.raise_for_status()
        return [_parse_place(p) for p in r.json().get("places", [])]
    except Exception as e:
        print(f"Google Places search error: {e}")
        return []


def get_reviews(place_id: str, limit: int = 3) -> list[str]:
    if not place_id:
        return []

    headers = {
        "Content-Type":     "application/json",
        "X-Goog-Api-Key":   _api_key(),
        "X-Goog-FieldMask": REVIEW_FIELD_MASK,
    }

    try:
        url = DETAILS_URL.format(place_id=place_id)
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return []
        reviews = r.json().get("reviews", [])
        return [
            rev.get("text", {}).get("text", "")
            for rev in reviews[:limit]
            if rev.get("text", {}).get("text")
        ]
    except Exception:
        return []


def _parse_place(raw: dict) -> dict:
    price_map = {
        "PRICE_LEVEL_FREE":           0,
        "PRICE_LEVEL_INEXPENSIVE":    1,
        "PRICE_LEVEL_MODERATE":       2,
        "PRICE_LEVEL_EXPENSIVE":      3,
        "PRICE_LEVEL_VERY_EXPENSIVE": 4,
    }

    open_now = None
    hours = raw.get("regularOpeningHours", {})
    if hours:
        open_now = hours.get("openNow")

    primary = raw.get("primaryTypeDisplayName", {}).get("text", "")
    types = [t.replace("_", " ").title() for t in raw.get("types", [])
             if t not in ("establishment", "food", "point_of_interest")]

    categories = [primary] if primary else types[:2]

    return {
        "fsq_id":     raw.get("id", ""),
        "name":       raw.get("displayName", {}).get("text", "Unknown"),
        "address":    raw.get("formattedAddress", ""),
        "categories": categories,
        "price":      price_map.get(raw.get("priceLevel", ""), None),
        "rating":     raw.get("rating"),
        "open_now":   open_now,
        "total_tips": raw.get("userRatingCount", 0),
    }


def get_tips(fsq_id: str, limit: int = 2) -> list[str]:
    return get_reviews(fsq_id, limit=limit)


def format_for_prompt(restaurants: list[dict], fetch_tips: bool = True) -> str:
    if not restaurants:
        return "No restaurants found."

    lines = ["## Nearby restaurants (Google Places)"]

    for i, r in enumerate(restaurants):
        price_str  = PRICE_LABEL.get(r["price"], "")
        rating_str = f"{r['rating']}/5 ({r['total_tips']} reviews)" if r["rating"] else "unrated"
        open_str   = " · open now" if r["open_now"] else ""
        cats       = ", ".join(r["categories"][:2]) if r["categories"] else ""

        lines.append(
            f"\n- **{r['name']}** | {price_str} | {rating_str}{open_str}"
            f"\n  {cats} · {r['address']}"
        )

        if fetch_tips and r["fsq_id"] and i < 3:
            for review in get_tips(r["fsq_id"]):
                snippet = review[:200] + "..." if len(review) > 200 else review
                lines.append(f'  > "{snippet}"')

    return "\n".join(lines)


def price_sensitivity_to_tier(sensitivity: str) -> Optional[int]:
    return PRICE_SENSITIVITY_MAP.get(sensitivity)



