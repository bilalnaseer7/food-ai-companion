import json
import os

PROFILE_PATH = "data/taste_profile.json"

DEFAULT_PROFILE = {
    "preferred_cuisines": [],
    "liked_foods": [],
    "disliked_foods": [],
    "budget": "moderate",
    "online_order": "No",
    "cuisine_scores": {},
    "food_scores": {},
    "accepted": [],
    "rejected": [],
}


def load_profile(path: str = PROFILE_PATH) -> dict:
    # Load from disk if exists, otherwise return defaults
    if os.path.exists(path):
        with open(path, "r") as f:
            saved = json.load(f)
        return {**DEFAULT_PROFILE, **saved}
    return DEFAULT_PROFILE.copy()


def save_profile(profile: dict, path: str = PROFILE_PATH) -> None:
    # Write profile to JSON, creating the data/ directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)


def update_profile(
    profile: dict,
    restaurant_name: str,
    accepted: bool,
    cuisines: list[str] | None = None,
    foods: list[str] | None = None,
    price: int | None = None,
) -> dict:
    delta = 0.15 if accepted else -0.15

    # Nudge cuisine scores up or down based on feedback
    if cuisines:
        for cuisine in cuisines:
            current = profile["cuisine_scores"].get(cuisine, 0.0)
            profile["cuisine_scores"][cuisine] = round(max(-1.0, min(1.0, current + delta)), 3)

    # Nudge food scores up or down based on feedback
    if foods:
        for food in foods:
            current = profile["food_scores"].get(food, 0.0)
            profile["food_scores"][food] = round(max(-1.0, min(1.0, current + delta)), 3)

    # Track accepted and rejected restaurant names
    if accepted and restaurant_name not in profile["accepted"]:
        profile["accepted"].append(restaurant_name)
    elif not accepted and restaurant_name not in profile["rejected"]:
        profile["rejected"].append(restaurant_name)

    # Infer budget from price levels of accepted restaurants (EMA, alpha=0.3)
    # Write through to profile["budget"] so all downstream code benefits automatically
    if accepted and price is not None and 1 <= price <= 4:
        _BUDGET_REVERSE = {1: "budget", 2: "moderate", 3: "premium", 4: "premium+"}
        current = profile.get("inferred_budget_level")
        if current is None:
            profile["inferred_budget_level"] = float(price)
        else:
            profile["inferred_budget_level"] = round(0.3 * price + 0.7 * current, 2)
        profile["budget"] = _BUDGET_REVERSE.get(round(profile["inferred_budget_level"]), "moderate")
        profile["inferred_budget_count"] = profile.get("inferred_budget_count", 0) + 1

    profile["preferred_cuisines"] = [
        k for k, v in profile["cuisine_scores"].items() if v > 0.2
    ]
    profile["liked_foods"] = [
        k for k, v in profile["food_scores"].items() if v > 0.2
    ]
    profile["disliked_foods"] = [
        k for k, v in profile["food_scores"].items() if v < -0.2
    ]
 

    return profile


def profile_summary(profile: dict) -> str:
    # Serialize profile state into a readable string for the results file
    lines = [
        f"Preferred cuisines: {', '.join(profile['preferred_cuisines']) or 'none yet'}",
        f"Liked foods: {', '.join(profile['liked_foods']) or 'none yet'}",
        f"Disliked foods: {', '.join(profile['disliked_foods']) or 'none yet'}",
        f"Budget: {profile['budget']}",
        f"Online order: {profile['online_order']}",
        f"Accepted restaurants: {len(profile['accepted'])}",
        f"Rejected restaurants: {len(profile['rejected'])}",
    ]
    return "\n".join(lines)
