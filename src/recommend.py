from openai import OpenAI
from src.cook_mode import generate_cook_recommendations
from src.retrieval import retrieve_restaurants
import re


CHAT_MODEL = "gpt-4o-mini"


def _profile_to_text(user_profile: dict) -> str:
    return (
        f"Preferred cuisines: {', '.join(user_profile.get('preferred_cuisines', []))}\n"
        f"Liked foods: {', '.join(user_profile.get('liked_foods', []))}\n"
        f"Disliked foods: {', '.join(user_profile.get('disliked_foods', []))}\n"
        f"Removed foods: {', '.join(user_profile.get('removed_foods', []))}\n"
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
        "Recommend 3 NYC restaurant options or restaurant types that best fit this profile. "
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
        "Pick the best 3 restaurants from the retrieved records.\n\n"
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

def map_recommend(client: OpenAI, query: str, user_profile: dict, borough: str = "manhattan") -> tuple[str, list]:
    from src.places import search_restaurants, format_for_prompt
 
    restaurants = search_restaurants(
        query=query,
        borough=borough,
        price=None,
        limit=8,
    )
 
    restaurant_block = format_for_prompt(restaurants, fetch_tips=True)
 
    system_prompt = (
        "You are a restaurant recommendation assistant for New York City. "
        "You must recommend only from the live restaurant data provided below. "
        "Use the user's taste profile and the retrieved evidence together. "
        "Treat budget as general comfort context, not a hard filter; explicit user intent such as Michelin, tasting menu, splurge, cheap eats, or casual should override the stored budget. "
        "If the user asks for walking distance, interpret that as less than 1 mile from the requested location. "
        "Do not invent restaurants outside the retrieved list."
    )
 
    user_prompt = (
        f"User request: {query}\n\n"
        f"User taste profile:\n{_profile_to_text(user_profile)}\n\n"
        f"{restaurant_block}\n\n"
        "Pick the best 5 restaurants from the list above. "
        "For each recommendation, provide:\n"
        "1. Restaurant name\n"
        "2. Why it matches the user's request\n"
        "3. Why it matches the taste profile\n"
        "4. One short detail from the user tips if available\n\n"
        "Use the budget to understand the user's usual comfort zone, but do not let it override explicit price or occasion cues in the current request.\n\n"
        "Then include one short overall summary comparing why the top choice is strongest."
    )
 
    answer = _chat(client, system_prompt, user_prompt)
    return answer, restaurants

def recommend_recipe(craving: str, profile: dict, client: OpenAI | None = None) -> str:
    return generate_cook_recommendations(client, craving, profile)


def recommend_cocktail(vibe: str, profile: dict) -> str:
    bar = profile.get("bar_inventory", [])
    if not bar:
        return "Your bar is empty. Add some spirits and mixers in the Cocktails tab."

    system_prompt = (
        "You are a creative bartender. Generate a cocktail from the user's available spirits and mixers. "
        "If a classic ingredient is missing, find a lateral substitute and explain the flavor logic briefly."
    )

    user_prompt = (
        f"Vibe: {vibe}\n"
        f"Available: {', '.join(bar)}\n\n"
        "Create a drink with exact measurements, substitution rationale if needed, and an optional garnish."
    )

    return _chat(OpenAI(), system_prompt, user_prompt)

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
