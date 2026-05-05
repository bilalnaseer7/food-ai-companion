from collections.abc import Iterable

from openai import OpenAI


CHAT_MODEL = "gpt-4o-mini"


def parse_ingredient_text(value) -> list[str]:
    """Normalize comma/newline separated ingredients while preserving order."""
    if value is None:
        return []

    if isinstance(value, str):
        raw_items = value.replace("\n", ",").replace(";", ",").split(",")
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        raw_items = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                raw_items.extend(item.replace("\n", ",").replace(";", ",").split(","))
            else:
                raw_items.append(str(item))
    else:
        raw_items = [str(value)]

    cleaned = []
    seen = set()
    for item in raw_items:
        normalized = " ".join(str(item).strip().split())
        key = normalized.lower()
        if normalized and key not in seen:
            cleaned.append(normalized)
            seen.add(key)
    return cleaned


def _profile_list(profile: dict, *keys: str) -> list[str]:
    for key in keys:
        values = parse_ingredient_text(profile.get(key))
        if values:
            return values
    return []


def _join_or_none(values: list[str]) -> str:
    return ", ".join(values) if values else "none provided"


def _meal_context_text(occasion: str, meal_type: str) -> str:
    parts = []
    for value in (occasion, meal_type):
        cleaned = str(value or "").strip()
        lower_cleaned = cleaned.lower()
        existing = {part.lower() for part in parts}
        overlaps_existing = any(lower_cleaned in part or part in lower_cleaned for part in existing)
        if cleaned and lower_cleaned not in existing and not overlaps_existing:
            parts.append(cleaned)
    return " / ".join(parts) if parts else "not specified"


def build_cook_context(
    craving: str,
    profile: dict,
    ingredients: list[str] | str | None = None,
) -> dict:
    profile = profile or {}
    available = parse_ingredient_text(
        ingredients if ingredients is not None else profile.get("pantry", [])
    )
    preferred_cuisines = _profile_list(profile, "preferred_cuisines", "preferred_cuisine")

    return {
        "craving": " ".join(str(craving or "").strip().split()),
        "available_ingredients": available,
        "preferred_cuisines": preferred_cuisines,
        "liked_foods": _profile_list(profile, "liked_foods"),
        "disliked_foods": _profile_list(profile, "disliked_foods"),
        "dietary_restrictions": _profile_list(
            profile,
            "dietary_restrictions",
            "restrictions",
            "allergies",
        ),
        "budget": str(profile.get("budget", "moderate") or "moderate"),
        "time_preference": str(
            profile.get("time_preference")
            or profile.get("time_budget")
            or profile.get("cook_time")
            or "not specified"
        ),
        "occasion": str(profile.get("occasion", "casual meal") or "casual meal"),
        "meal_type": str(profile.get("meal_type", "") or ""),
        "grounding_note": (
            "No recipe dataset is available in this repository, so this mode is "
            "LLM + taste-profile generation, not recipe RAG."
        ),
    }


def build_cook_prompt(context: dict) -> tuple[str, str]:
    meal_context = _meal_context_text(context["occasion"], context["meal_type"])

    system_prompt = (
        "You are Food AI Companion's Cook at Home mode. "
        "Generate practical home-cooking ideas using the user's available ingredients, "
        "craving, and taste profile. Be grounded and transparent: this repository does "
        "not currently include a recipe database, so do not claim recipe-dataset or "
        "RAG grounding. Do not invent exact nutrition facts. Avoid disliked foods and "
        "dietary restrictions. If a recipe depends on missing ingredients, say so clearly "
        "and suggest reasonable substitutions with flavor reasoning."
    )

    user_prompt = (
        "Cook at Home request\n"
        f"Craving: {context['craving'] or 'not specified'}\n"
        f"Available ingredients: {_join_or_none(context['available_ingredients'])}\n"
        f"Preferred cuisines: {_join_or_none(context['preferred_cuisines'])}\n"
        f"Liked foods/flavors: {_join_or_none(context['liked_foods'])}\n"
        f"Disliked foods: {_join_or_none(context['disliked_foods'])}\n"
        f"Dietary restrictions: {_join_or_none(context['dietary_restrictions'])}\n"
        f"Budget: {context['budget']}\n"
        f"Time preference: {context['time_preference']}\n"
        f"Occasion/meal type: {meal_context}\n"
        f"Grounding note: {context['grounding_note']}\n\n"
        "Return 2 to 3 recipe ideas. For each recipe, use this exact structure:\n"
        "RECIPE: <name>\n"
        "WHY IT FITS: <why it matches the craving and taste profile>\n"
        "USES FROM PANTRY: <specific available ingredients used>\n"
        "MISSING OR SUBSTITUTE INGREDIENTS: <missing ingredients and substitutes, or none>\n"
        "QUICK STEPS: <3 to 5 concise steps>\n"
        "CAUTION: <short note if the recipe depends on missing ingredients, otherwise 'None'>\n\n"
        "Keep the recommendations realistic for a home cook and avoid unsupported claims."
    )

    return system_prompt, user_prompt


def _chat(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.35,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def generate_cook_recommendations(
    client: OpenAI | None,
    craving: str,
    profile: dict,
    ingredients: list[str] | str | None = None,
) -> str:
    context = build_cook_context(craving=craving, profile=profile, ingredients=ingredients)

    if not context["available_ingredients"]:
        return "Your pantry is empty. Add available ingredients in the Cook tab first."

    client = client or OpenAI()
    system_prompt, user_prompt = build_cook_prompt(context)
    return _chat(client, system_prompt, user_prompt)
