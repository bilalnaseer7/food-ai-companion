import os
import requests
from typing import Optional

GOOGLE_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
SEARCH_URL   = "https://places.googleapis.com/v1/places:searchText"
DETAILS_URL  = "https://places.googleapis.com/v1/places/{place_id}"
PHOTO_URL    = "https://places.googleapis.com/v1/{photo_name}/media"
GEOCODE_URL  = "https://maps.googleapis.com/maps/api/geocode/json"

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
    "places.regularOpeningHours",
    "places.businessStatus",
    "places.photos",
    "places.liveMusic",
    "places.outdoorSeating",
    "places.servesCocktails",
    "places.servesWine",
    "places.servesBrunch",
    "places.servesVegetarianFood",
    "places.goodForGroups",
    "places.menuForChildren",
    "places.reservable",
    "places.location",
])

REVIEW_FIELD_MASK = "reviews,rating"

def _api_key() -> str:
    key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not key:
        raise EnvironmentError(
            "GOOGLE_PLACES_API_KEY not set. "
            "Enable billing at https://console.cloud.google.com and get a key."
        )
    return key


def geocode_location(address: str) -> Optional[tuple[float, float]]:
    try:
        r = requests.get(
            GEOCODE_URL,
            params={"address": address, "key": _api_key()},
            timeout=5,
        )
        results = r.json().get("results", [])
        if not results:
            return None
        loc = results[0]["geometry"]["location"]
        return (loc["lat"], loc["lng"])
    except Exception:
        return None


def search_restaurants(
    query: str,
    borough: str = "New York, NY",
    price: Optional[int] = None,
    open_now: bool = False,
    limit: int = 8,
) -> list[dict]:
    body = {
        "textQuery":    f"{query} restaurant near {borough}",
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


def get_photo_uri(photo_name: str, max_width: int = 640) -> str:
    if not photo_name:
        return ""

    try:
        r = requests.get(
            PHOTO_URL.format(photo_name=photo_name),
            params={
                "key": _api_key(),
                "maxWidthPx": max_width,
                "skipHttpRedirect": "true",
            },
            timeout=8,
        )
        if r.status_code != 200:
            return ""
        return r.json().get("photoUri", "")
    except Exception:
        return ""


def _fmt_time(h: int, m: int) -> str:
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {ampm}" if m else f"{h12} {ampm}"


def _closes_at_str(periods: list) -> str:
    from datetime import datetime
    if not periods:
        return ""
    now = datetime.now()
    google_today = (now.weekday() + 1) % 7
    google_yesterday = (google_today - 1) % 7
    current_mins = now.hour * 60 + now.minute

    for period in periods:
        o = period.get("open", {})
        c = period.get("close", {})
        if not o or not c:
            continue
        open_day, open_mins = o.get("day"), o.get("hour", 0) * 60 + o.get("minute", 0)
        close_day, close_h, close_m = c.get("day"), c.get("hour", 0), c.get("minute", 0)
        close_mins = close_h * 60 + close_m

        # Period started today and closes today (or tomorrow overnight)
        if open_day == google_today and open_mins <= current_mins:
            if close_day == google_today and current_mins < close_mins:
                return f"Open until {_fmt_time(close_h, close_m)}"
            if close_day != google_today:  # closes past midnight
                return f"Open until {_fmt_time(close_h, close_m)}"
        # Period started yesterday and closes today (overnight)
        if open_day == google_yesterday and close_day == google_today and current_mins < close_mins:
            return f"Open until {_fmt_time(close_h, close_m)}"
    return ""


def _next_open_str(periods: list) -> str:
    from datetime import datetime
    if not periods:
        return ""
    now = datetime.now()

    google_today = (now.weekday() + 1) % 7
    current_mins = now.hour * 60 + now.minute
    DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    for offset in range(8):
        check_day = (google_today + offset) % 7
        day_slots = sorted(
            [p["open"] for p in periods if p.get("open", {}).get("day") == check_day],
            key=lambda s: s.get("hour", 0) * 60 + s.get("minute", 0),
        )
        for slot in day_slots:
            slot_mins = slot.get("hour", 0) * 60 + slot.get("minute", 0)
            if offset == 0 and slot_mins <= current_mins:
                continue
            h, m = slot.get("hour", 0), slot.get("minute", 0)
            time_str = _fmt_time(h, m)
            if offset == 0:
                return f"Opens today {time_str}"
            if offset == 1:
                return f"Opens tomorrow {time_str}"
            return f"Opens {time_str} {DAY_NAMES[check_day]}"
    return ""


def _parse_place(raw: dict) -> dict:
    price_map = {
        "PRICE_LEVEL_FREE":           0,
        "PRICE_LEVEL_INEXPENSIVE":    1,
        "PRICE_LEVEL_MODERATE":       2,
        "PRICE_LEVEL_EXPENSIVE":      3,
        "PRICE_LEVEL_VERY_EXPENSIVE": 4,
    }

    open_now = None
    next_open = ""
    closes_at = ""
    hours = raw.get("regularOpeningHours", {})
    if hours:
        open_now = hours.get("openNow")
        periods = hours.get("periods", [])
        if open_now is False:
            next_open = _next_open_str(periods)
        elif open_now is True:
            closes_at = _closes_at_str(periods)

    primary = raw.get("primaryTypeDisplayName", {}).get("text", "")
    types = [t.replace("_", " ").title() for t in raw.get("types", [])
             if t not in ("establishment", "food", "point_of_interest")]

    categories = [primary] if primary else types[:2]
    photo = (raw.get("photos") or [{}])[0]
    photo_name = photo.get("name", "")
    photo_url = get_photo_uri(photo_name)
    photo_attribution = ", ".join(
        attr.get("displayName", "")
        for attr in photo.get("authorAttributions", [])
        if attr.get("displayName")
    )

    ATTR_MAP = [
        ("liveMusic",             "Lively"),
        ("outdoorSeating",        "Outdoor"),
        ("servesCocktails",       "Cocktails"),
        ("servesWine",            "Wine"),
        ("servesBrunch",          "Brunch"),
        ("servesVegetarianFood",  "Veggie-Friendly"),
        ("goodForGroups",         "Great for Groups"),
        ("menuForChildren",       "Family-Friendly"),
        ("reservable",            "Reservations"),
    ]
    attributes = [label for field, label in ATTR_MAP if raw.get(field)]

    loc = raw.get("location", {})

    return {
        "fsq_id":            raw.get("id", ""),
        "name":              raw.get("displayName", {}).get("text", "Unknown"),
        "address":           raw.get("formattedAddress", ""),
        "categories":        categories,
        "attributes":        attributes,
        "price":             price_map.get(raw.get("priceLevel", ""), None),
        "rating":            raw.get("rating"),
        "open_now":          open_now,
        "next_open":         next_open,
        "closes_at":         closes_at,
        "total_tips":        raw.get("userRatingCount", 0),
        "photo_url":         photo_url,
        "photo_attribution": photo_attribution,
        "lat":               loc.get("latitude"),
        "lng":               loc.get("longitude"),
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
