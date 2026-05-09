from openai import OpenAI
from src.cook_mode import generate_cook_recommendations
from src.retrieval import retrieve_restaurants
import json
import re


CHAT_MODEL = "gpt-4o-mini"


def _profile_to_text(user_profile: dict) -> str:
    return (
        f"Preferred cuisines: {', '.join(user_profile.get('preferred_cuisines', []))}\n"
        f"Liked foods: {', '.join(user_profile.get('liked_foods', []))}\n"
        f"Disliked foods: {', '.join(user_profile.get('disliked_foods', []))}\n"
        f"Budget: {user_profile.get('budget', '')}\n"
        f"Online order preference: {user_profile.get('online_order', '')}\n"
        f"Occasion: {user_profile.get('occasion', '')}\n"
        f"City: {user_profile.get('city', 'New York City')}"
    )


def _chat(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def baseline_recommend(client: OpenAI, query: str) -> str:
    system_prompt = (
        "You are a restaurant recommendation assistant for New York City. "
        "Give concise and realistic restaurant recommendations based only on the user's request. "
        "You do not have access to external restaurant records. "
        "Do not pretend you know live restaurant inventory or exact menu details."
    )

    user_prompt = (
        f"User request: {query}\n\n"
        "Recommend 5 NYC restaurant options or restaurant types. "
        "For each one, briefly explain why it may fit. "
        "Be honest that this is based on general knowledge only."
    )

    return _chat(client, system_prompt, user_prompt)


def profile_recommend(client: OpenAI, query: str, user_profile: dict) -> str:
    system_prompt = (
        "You are a restaurant recommendation assistant for New York City. "
        "Use the user's taste profile carefully. "
        "Recommend restaurants or restaurant types that fit the user's cuisine, food, budget, "
        "and ordering preferences. "
        "You do not have access to retrieved restaurant records, so do not invent evidence."
    )

    user_prompt = (
        f"User request: {query}\n\n"
        f"User taste profile:\n{_profile_to_text(user_profile)}\n\n"
        "Recommend 5 NYC restaurant options or restaurant types that best fit this profile when possible. "
        "For each one, explain why it matches the profile."
    )

    return _chat(client, system_prompt, user_prompt)


def rag_recommend(client: OpenAI, query: str, user_profile: dict, df, top_k: int = 5) -> tuple[str, list]:
    retrieved = retrieve_restaurants(
        query=query,
        user_profile=user_profile,
        df=df,
        client=client,
        top_k=top_k,
    )

    context_blocks = []
    for i, row in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[Restaurant {i}]\n"
            f"Name: {row['title']}\n"
            f"Category: {row['category']}\n"
            f"Popular food: {row['popular_food']}\n"
            f"Online order: {row['online_order']}\n"
            f"Number of reviews: {row['num_reviews']}\n"
            f"Retrieval score: {row['retrieval_score']}\n"
            f"Review snippets: {row['review_snippets'][:700]}\n"
        )

    context_text = "\n".join(context_blocks)

    system_prompt = (
        "You are a restaurant recommendation assistant for New York City. "
        "You must recommend only from the retrieved restaurant records provided. "
        "Use the user's request, the taste profile, and the retrieved evidence together. "
        "Do not invent restaurants or unsupported claims. "
        "Do not select a restaurant if its cuisine or featured food clearly conflicts with the user's request. "
        "If evidence is weak for a restaurant, say so briefly instead of making things up."
    )

    user_prompt = (
        f"User request: {query}\n\n"
        f"User taste profile:\n{_profile_to_text(user_profile)}\n\n"
        f"Retrieved restaurant evidence:\n{context_text}\n\n"
        "Pick the best 5 restaurants from the retrieved records when 5 viable options are available; "
        "return fewer only if the retrieved records do not contain 5 reasonable matches.\n\n"
        "For each recommendation, provide:\n"
        "1. Restaurant name\n"
        "2. Why it matches the user's request\n"
        "3. Why it matches the taste profile\n"
        "4. One short supporting evidence phrase from the retrieved review snippets\n"
        "5. One caution/uncertainty if the evidence is imperfect\n\n"
        "Then include one short final summary explaining why the top choice is strongest overall."
    )

    answer = _chat(client, system_prompt, user_prompt)
    return answer, retrieved

def recommend_recipe(craving: str, profile: dict, client: OpenAI | None = None, previous_response: str | None = None) -> str:
    return generate_cook_recommendations(client, craving, profile, previous_response=previous_response)


def explicit_cocktail_search_terms(vibe: str) -> list[str]:
    text = " ".join(str(vibe or "").lower().split())
    compact = re.sub(r"[^a-z0-9]", "", text)
    if compact in {"gt", "gandt", "ginandtonic", "gintonic"}:
        return ["Gin Tonic", "Gin and Tonic"]
    if "gin" in text and "tonic" in text:
        return ["Gin Tonic", "Gin and Tonic"]
    return []


def infer_cocktail_search_terms(client: OpenAI, vibe: str, bar_inventory: list[str]) -> list[str]:
    fallback = ["margarita", "daiquiri", "negroni", "old fashioned", "martini", "mojito"]
    vibe_text = " ".join(str(vibe or "").split())
    bar_text = ", ".join(bar_inventory or [])
    if not vibe_text:
        return fallback

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Infer CocktailDB cocktail name search targets from a user's mood/vibe. "
                        "Return JSON only: an array of 4 to 8 classic or common cocktail names. "
                        "Make the list specific to the vibe/request and vary style, base spirit, and texture. "
                        "Do not default to Margarita, Daiquiri, or Martini unless they genuinely fit the vibe. "
                        "Do not echo vague words from the user. Prefer likely CocktailDB names, e.g. "
                        "Negroni, Old Fashioned, Margarita, Daiquiri, Martini, Mojito, Manhattan, "
                        "Whiskey Sour, Tom Collins, Mai Tai, Cosmopolitan, Boulevardier, Rob Roy."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Vibe/request: {vibe_text}\n"
                        f"Available bar context: {bar_text or 'unspecified / flexible'}"
                    ),
                },
            ],
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            terms = []
            seen = set()
            for item in parsed:
                term = " ".join(str(item or "").split())
                key = term.lower()
                if term and key not in seen:
                    terms.append(term)
                    seen.add(key)
                if len(terms) >= 8:
                    break
            if terms:
                return terms
    except Exception:
        pass

    return fallback


def recommend_cocktail(vibe: str, profile: dict, previous_response: str | None = None) -> str:
    bar = profile.get("bar_inventory", [])
    if not bar:
        return "Your bar is empty. Add some spirits and mixers in the Cocktails tab.", []

    from src.cocktail_db import (
        find_cocktails_by_name,
        find_matching_cocktails,
        format_for_prompt,
        is_vague_bar_inventory,
        normalize_bar_inventory,
    )
    client_inst = OpenAI()
    bar = normalize_bar_inventory(bar, client_inst)
    matched = [] if is_vague_bar_inventory(bar) else find_matching_cocktails(bar, top_k=12)
    if not matched:
        search_terms = explicit_cocktail_search_terms(vibe)
        for term in infer_cocktail_search_terms(client_inst, vibe, bar):
            if term.lower() not in {existing.lower() for existing in search_terms}:
                search_terms.append(term)
        matched = find_cocktails_by_name(search_terms, top_k=12)
    grounding_block = format_for_prompt(matched)
    has_grounding = bool(matched)

    previous = str(previous_response or "").strip()
    previous_block = (
        f"\nPrevious suggestion (modify this based on the remix instruction):\n{previous}\n"
        if previous else ""
    )

    system_prompt = (
        "You are a thoughtful craft bartender with access to CocktailDB recipe data. "
        + (
            "Prefer recommending cocktails from the grounded records below; use their exact "
            "ingredient lists as the base recipe, but expand terse instructions into a complete, "
            "practical method without changing the drink. "
            if has_grounding else ""
        )
        + "Choose drinks that genuinely match the requested vibe and available bar. "
        "If a classic ingredient is missing, suggest a lateral substitute and explain the flavor logic briefly. "
        "Do not recommend a poor fit just because it shares one ingredient."
    )

    grounding_section = (
        f"\nCocktailDB grounded records (prefer these):\n{grounding_block}\n"
        if has_grounding else ""
    )

    user_prompt = (
        f"Vibe: {vibe}\n"
        f"Available bar: {', '.join(bar)}\n"
        f"{grounding_section}"
        f"{previous_block}\n"
        + (
            "Recommend exactly 1 revised cocktail, based only on the previous suggestion and remix instruction. "
            if previous else
            "Recommend 3 cocktails when the bar inventory and CocktailDB candidates can support 3 distinct, good-fit options; "
            "return fewer only if there are not 3 viable cocktails. "
        )
        +
        "For each, use this exact structure so the app can format it:\n"
        "COCKTAIL: <name>\n"
        "WHY IT FITS: <1 concise sentence for the card only; do not repeat this idea elsewhere>\n"
        "GLASS: <glassware>\n"
        "ICE: <ice style, or 'None'>\n"
        "INGREDIENTS:\n"
        "- <measure> <ingredient>\n"
        "METHOD:\n"
        "1. <complete bartender step, including chill/shake/stir/build, strain, and top if relevant>\n"
        "GARNISH: <optional garnish or 'None'>\n"
        "SUBSTITUTIONS: <missing ingredients and substitutes with brief flavor logic, or 'None'>\n"
        "NOTE: <one useful serving or balance note, or 'None'>\n\n"
        "Use exact measures. Make the recipe complete enough to mix from directly. "
        "Keep COCKTAIL and WHY IT FITS as plain parser labels only; the full recipe should stand on its own."
    )

    return _chat(client_inst, system_prompt, user_prompt), matched

def combined_recommend(client: OpenAI, query: str, user_profile: dict, csv_results: list, fsq_results: list) -> tuple[str, list]:
    csv_block = "\n".join([
        f"- {r['title']} | {r['category']} | Popular: {r['popular_food']}"
        for r in csv_results
    ])

    def open_status(r: dict) -> str:
        if r.get("open_now") is True:
            return "openNow=true"
        if r.get("open_now") is False:
            return "openNow=false"
        return "openNow=unknown"

    fsq_block = "\n".join([
        f"- {r['name']} | {', '.join(r.get('categories', [])[:2])} | "
        f"Rating: {r.get('rating', 'N/A')}/5 | {open_status(r)} | {r.get('address', '')}"
        for r in fsq_results
    ]) if fsq_results else "No live results available."

    system_prompt = (
        "You are a restaurant recommendation assistant for New York City. "
        "You have two sources of restaurant data: a curated dataset and live Google Places results. "
        "Use both sources together with the user's taste profile to select and rank the best 5 restaurants. "
        "Treat budget as general comfort context, not a hard filter; explicit user intent such as Michelin, tasting menu, splurge, cheap eats, or casual should override the stored budget. "
        "If the user asks for walking distance, interpret that as less than 1 mile from the requested location. "
        "If the user explicitly asks for places that are open now, only choose live results marked openNow=true. Otherwise, do not exclude options based on open status. "
        "Only recommend restaurants from the provided lists. Do not invent any."
    )

    user_prompt = (
        f"User request: {query}\n\n"
        f"User taste profile:\n{_profile_to_text(user_profile)}\n\n"
        f"Curated dataset results:\n{csv_block}\n\n"
        f"Live Google Places results:\n{fsq_block}\n\n"
        "Pick the best 5 restaurants from the live Google Places results. "
        "For each, write exactly in this format with no numbering or extra text:\n"
        "RESTAURANT: <exact name>\nBLURB: <1-2 sentence explanation why it fits>\n\n"
        "After all 5, add one sentence starting with BEST: naming the top pick and why. "
        "Use exact restaurant names as they appear in the list. Do not number the entries."
        "Do not mention exact ratings in the blurbs as this information will be included separately. Focus only on atmosphere, food, and fit with the request. "
        "Use the budget to understand the user's usual comfort zone, but do not let it override explicit price or occasion cues in the current request."
    )

    answer = _chat(client, system_prompt, user_prompt)

    blurbs = {}
    for match in re.finditer(
        r'RESTAURANT:\s*(.+?)\nBLURB:\s*(.+?)(?=\n\s*RESTAURANT:|\nBEST:|\Z)',
        answer, re.DOTALL
    ):
        name = match.group(1).strip().lstrip('0123456789. ')
        blurb = match.group(2).strip()
        blurbs[name] = blurb

    best_match = re.search(r'BEST:\s*(.+)', answer)
    best_line = best_match.group(1).strip() if best_match else ""

    def normalize(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())

    normalized_blurbs = {normalize(k): v for k, v in blurbs.items()}

    selected = []
    seen = set()
    for r in fsq_results:
        norm = normalize(r["name"])
        if norm in normalized_blurbs and norm not in seen:
            r["blurb"] = normalized_blurbs[norm]
            selected.append(r)
            seen.add(norm)
        if len(selected) == 5:
            break

    if not selected:
        blurb_list = list(blurbs.values())
        for i, r in enumerate(fsq_results[:5]):
            r["blurb"] = blurb_list[i] if i < len(blurb_list) else ""
            selected.append(r)

    return best_line, selected
