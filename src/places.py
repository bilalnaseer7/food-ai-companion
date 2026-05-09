import os
import requests
from typing import Optional

GOOGLE_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
SEARCH_URL   = "https://places.googleapis.com/v1/places:searchText"
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
    "places.currentOpeningHours",
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


def _period_point_datetime(point: dict, now, *, prefer_past: bool = False):
    from datetime import datetime, timedelta

    if not point:
        return None

    hour = point.get("hour", 0)
    minute = point.get("minute", 0)
    date_info = point.get("date") or {}
    if date_info.get("year") and date_info.get("month") and date_info.get("day"):
        return datetime(
            date_info["year"],
            date_info["month"],
            date_info["day"],
            hour,
            minute,
            tzinfo=now.tzinfo,
        )

    google_day = point.get("day")
    if google_day is None:
        return None

    google_today = (now.weekday() + 1) % 7
    days_until = (google_day - google_today) % 7
    dt = (now.replace(hour=0, minute=0, second=0, microsecond=0)
          + timedelta(days=days_until, hours=hour, minutes=minute))
    if prefer_past and dt > now:
        dt -= timedelta(days=7)
    return dt


def _closes_at_str(periods: list) -> str:
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    if not periods:
        return ""
    now = datetime.now(ZoneInfo("America/New_York"))

    for period in periods:
        o = period.get("open", {})
        c = period.get("close", {})
        if not o:
            continue
        if not c:
            continue

        open_dt = _period_point_datetime(o, now, prefer_past=True)
        close_dt = _period_point_datetime(c, now)
        if not open_dt or not close_dt:
            continue
        if close_dt <= open_dt:
            close_dt += timedelta(days=1)
        if open_dt <= now < close_dt:
            return f"Open until {_fmt_time(close_dt.hour, close_dt.minute)}"
    return ""


def _next_open_str(periods: list) -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    if not periods:
        return ""
    now = datetime.now(ZoneInfo("America/New_York"))
    # Google: 0=Sunday … 6=Saturday; Python weekday: 0=Monday … 6=Sunday
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
            return f"Opens {DAY_NAMES[check_day]} {time_str}"
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
    current_hours = raw.get("currentOpeningHours") or {}
    regular_hours = raw.get("regularOpeningHours") or {}
    hours = current_hours or regular_hours
    if hours:
        open_now = current_hours.get("openNow")
        if open_now is None:
            open_now = regular_hours.get("openNow")
        periods = current_hours.get("periods") or regular_hours.get("periods") or []
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



def price_sensitivity_to_tier(sensitivity: str) -> Optional[int]:
    return PRICE_SENSITIVITY_MAP.get(sensitivity)
