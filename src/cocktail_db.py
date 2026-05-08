"""CocktailDB grounding layer — free API (v1), no key required."""

import json
import re
import requests
from functools import lru_cache
from typing import Optional

_BASE = "https://www.thecocktaildb.com/api/json/v1/1"

# Mixers that produce too many matches to be useful for ranking
_SKIP_INGREDIENTS = {
    "ice", "water", "sugar", "salt", "sugar syrup", "simple syrup",
    "soda water", "club soda", "tonic water", "cola", "ginger beer",
    "ginger ale", "cream", "milk", "egg white", "egg", "lemon juice",
    "lime juice", "orange juice", "cranberry juice", "pineapple juice",
    "grenadine", "bitters", "angostura bitters", "peach bitters",
}


def normalize_bar_inventory(bar: list[str], client) -> list[str]:
    """
    Use the LLM to interpret vague or shorthand bar descriptions and return
    a flat list of specific ingredient names suitable for CocktailDB queries.
    If all items are already specific (no inference needed), returns as-is.
    """
    if not bar:
        return bar

    raw = ", ".join(bar)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a bartender's assistant. Given a bar inventory description, "
                        "return a JSON array of specific spirit and mixer names. "
                        "Expand shorthand phrases like 'the basics', 'well stocked', or vague descriptions "
                        "into their constituent ingredients. Keep specific items as-is. "
                        "Use common cocktail ingredient names (e.g. 'bourbon' not 'american whiskey'). "
                        "Return only the JSON array, nothing else."
                    ),
                },
                {"role": "user", "content": f"Bar inventory: {raw}"},
            ],
        )
        text = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(i, str) for i in parsed):
            return [i.strip() for i in parsed if i.strip()]
    except Exception:
        pass
    return bar


@lru_cache(maxsize=128)
def _search_by_ingredient(ingredient: str) -> list:
    try:
        r = requests.get(f"{_BASE}/filter.php", params={"i": ingredient}, timeout=5)
        data = r.json()
        drinks = data.get("drinks")
        if not isinstance(drinks, list):
            return []
        return [d for d in drinks if isinstance(d, dict)]
    except Exception:
        return []


@lru_cache(maxsize=256)
def _lookup_cocktail(cocktail_id: str) -> Optional[dict]:
    try:
        r = requests.get(f"{_BASE}/lookup.php", params={"i": cocktail_id}, timeout=5)
        data = r.json()
        drinks = data.get("drinks")
        return drinks[0] if drinks else None
    except Exception:
        return None


def _extract_ingredients(drink: dict) -> list[str]:
    items = []
    for i in range(1, 16):
        ing = (drink.get(f"strIngredient{i}") or "").strip()
        if ing:
            items.append(ing)
    return items


def _extract_ingredients_with_measures(drink: dict) -> list[str]:
    items = []
    for i in range(1, 16):
        ing = (drink.get(f"strIngredient{i}") or "").strip()
        measure = (drink.get(f"strMeasure{i}") or "").strip()
        if ing:
            items.append(f"{measure} {ing}".strip() if measure else ing)
    return items


def _coverage(cocktail_ingredients: list[str], bar_set: set[str]) -> tuple[int, int]:
    total = len(cocktail_ingredients)
    have = 0
    for ing in cocktail_ingredients:
        ing_lower = ing.lower()
        for bar_item in bar_set:
            if bar_item in ing_lower or ing_lower in bar_item:
                have += 1
                break
    return have, total


def find_matching_cocktails(bar_inventory: list[str], top_k: int = 8) -> list[dict]:
    """
    Query CocktailDB for cocktails matchable from bar_inventory (already normalized).
    Returns up to top_k cocktails sorted by ingredient coverage (descending).
    """
    if not bar_inventory:
        return []

    bar_lower = {item.lower().strip() for item in bar_inventory}

    spirits = [
        item for item in bar_inventory
        if item.lower().strip() not in _SKIP_INGREDIENTS
    ][:8]

    id_to_meta: dict[str, dict] = {}
    for spirit in spirits:
        results = _search_by_ingredient(spirit)
        for r in results:
            cid = r["idDrink"]
            if cid not in id_to_meta:
                id_to_meta[cid] = {
                    "name": r["strDrink"],
                    "thumb": r.get("strDrinkThumb", ""),
                    "hit_count": 0,
                }
            id_to_meta[cid]["hit_count"] += 1

    if not id_to_meta:
        return []

    sorted_ids = sorted(id_to_meta, key=lambda x: id_to_meta[x]["hit_count"], reverse=True)

    scored: list[dict] = []
    for cid in sorted_ids[: top_k * 3]:
        details = _lookup_cocktail(cid)
        if not details:
            continue

        ingredients = _extract_ingredients(details)
        if not ingredients:
            continue

        have, total = _coverage(ingredients, bar_lower)
        scored.append({
            "id": cid,
            "name": details.get("strDrink", ""),
            "category": details.get("strCategory", ""),
            "glass": details.get("strGlass", ""),
            "instructions": (details.get("strInstructions") or "").strip(),
            "ingredients": ingredients,
            "ingredients_with_measures": _extract_ingredients_with_measures(details),
            "thumbnail": details.get("strDrinkThumb", ""),
            "have_count": have,
            "total_ingredients": total,
            "coverage": have / total if total > 0 else 0,
        })

    scored.sort(key=lambda x: x["coverage"], reverse=True)
    return scored[:top_k]


def format_for_prompt(cocktails: list[dict]) -> str:
    if not cocktails:
        return "No CocktailDB records retrieved."

    blocks = []
    for c in cocktails:
        ing_list = ", ".join(c["ingredients_with_measures"])
        have_note = f"{c['have_count']}/{c['total_ingredients']} ingredients available"
        instr = re.sub(r'\s+', ' ', c["instructions"])[:400]
        blocks.append(
            f"COCKTAIL: {c['name']}\n"
            f"Category: {c['category']} | Glass: {c['glass']} | {have_note}\n"
            f"Ingredients: {ing_list}\n"
            f"Instructions: {instr}"
        )
    return "\n\n".join(blocks)
