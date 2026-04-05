from openai import OpenAI
from src.retrieval import retrieve_restaurants


CHAT_MODEL = "gpt-4o-mini"


def _profile_to_text(user_profile: dict) -> str:
    return (
        f"Preferred cuisines: {', '.join(user_profile.get('preferred_cuisines', []))}\n"
        f"Liked foods: {', '.join(user_profile.get('liked_foods', []))}\n"
        f"Disliked foods: {', '.join(user_profile.get('disliked_foods', []))}\n"
        f"Budget: {user_profile.get('budget', '')}\n"
        f"Online order preference: {user_profile.get('online_order', '')}\n"
        f"Occasion: {user_profile.get('occasion', '')}"
    )


def _chat(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def baseline_recommend(client: OpenAI, query: str) -> str:
    system_prompt = (
        "You are a restaurant recommendation assistant for New York City. "
        "Give concise, useful restaurant recommendations based only on the user's request. "
        "You do not have access to external restaurant data."
    )

    user_prompt = (
        f"User request: {query}\n\n"
        "Recommend 3 NYC restaurant options or restaurant types. "
        "For each, explain briefly why it may fit. "
        "Be honest that this is based on general knowledge, not retrieved data."
    )

    return _chat(client, system_prompt, user_prompt)


def profile_recommend(client: OpenAI, query: str, user_profile: dict) -> str:
    system_prompt = (
        "You are a restaurant recommendation assistant for New York City. "
        "Use the user's taste profile carefully. "
        "Recommend restaurants or restaurant types that fit the user's cuisine, food, budget, "
        "and ordering preferences. "
        "You do not have access to retrieved restaurant records."
    )

    user_prompt = (
        f"User request: {query}\n\n"
        f"User taste profile:\n{_profile_to_text(user_profile)}\n\n"
        "Recommend 3 NYC restaurant options or restaurant types that best fit this profile. "
        "Explain why each matches the user."
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
            f"Review excerpt: {row['review_text'][:500]}\n"
        )

    context_text = "\n".join(context_blocks)

    system_prompt = (
        "You are a restaurant recommendation assistant for New York City. "
        "You must recommend only from the retrieved restaurant records provided to you. "
        "Use the user's taste profile and the retrieved evidence together. "
        "Do not invent restaurants outside the retrieved list."
    )

    user_prompt = (
        f"User request: {query}\n\n"
        f"User taste profile:\n{_profile_to_text(user_profile)}\n\n"
        f"Retrieved restaurant evidence:\n{context_text}\n\n"
        "Pick the best 3 restaurants from the retrieved records. "
        "For each recommendation, provide: \n"
        "1. Restaurant name\n"
        "2. Why it matches the user's request\n"
        "3. Why it matches the taste profile\n"
        "4. One short evidence phrase from the retrieved review text\n\n"
        "Then include one short overall summary comparing why the top choice is strongest."
    )

    answer = _chat(client, system_prompt, user_prompt)
    return answer, retrieved

def foursquare_recommend(client: OpenAI, query: str, user_profile: dict, borough: str = "manhattan") -> tuple[str, list]:
    from src.foursquare_places import search_restaurants, format_for_prompt, price_sensitivity_to_tier
 
    price_tier = price_sensitivity_to_tier(user_profile.get("budget", "moderate"))
    restaurants = search_restaurants(
        query=query,
        borough=borough,
        price=price_tier,
        limit=8,
    )
 
    restaurant_block = format_for_prompt(restaurants, fetch_tips=True)
 
    system_prompt = (
        "You are a restaurant recommendation assistant for New York City. "
        "You must recommend only from the live restaurant data provided below. "
        "Use the user's taste profile and the retrieved evidence together. "
        "Do not invent restaurants outside the retrieved list."
    )
 
    user_prompt = (
        f"User request: {query}\n\n"
        f"User taste profile:\n{_profile_to_text(user_profile)}\n\n"
        f"{restaurant_block}\n\n"
        "Pick the best 3 restaurants from the list above. "
        "For each recommendation, provide:\n"
        "1. Restaurant name\n"
        "2. Why it matches the user's request\n"
        "3. Why it matches the taste profile\n"
        "4. One short detail from the user tips if available\n\n"
        "Then include one short overall summary comparing why the top choice is strongest."
    )
 
    answer = _chat(client, system_prompt, user_prompt)
    return answer, restaurants

def recommend_recipe(craving: str, profile: dict) -> str:
    pantry = profile.get("pantry", [])
    if not pantry:
        return "Your pantry is empty. Add some ingredients in the Cook tab."

    system_prompt = (
        "You are a creative home cooking assistant. "
        "Generate a recipe using primarily the user's available ingredients. "
        "If key ingredients are missing, suggest a specific substitution with a brief flavor explanation. "
        f"The user likes: {', '.join(profile.get('liked_foods', []) or ['varied flavors'])}. "
        f"Dislikes: {', '.join(profile.get('disliked_foods', []) or ['nothing noted'])}."
    )

    user_prompt = (
        f"Craving: {craving}\n"
        f"Available ingredients: {', '.join(pantry)}\n\n"
        "Generate a recipe that matches the craving, flags missing ingredients with substitutions, and gives clear steps."
    )

    return _chat(OpenAI(), system_prompt, user_prompt)


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

def combined_recommend(client: OpenAI, query: str, user_profile: dict, csv_results: list, fsq_results: list) -> str:
    csv_block = "\n".join([
        f"- {r['title']} | {r['category']} | Popular: {r['popular_food']}"
        for r in csv_results
    ])

    fsq_block = "\n".join([
        f"- {r['name']} | {', '.join(r.get('categories', [])[:2])} | Rating: {r.get('rating', 'N/A')}/10"
        for r in fsq_results
    ]) if fsq_results else "No live results available."

    system_prompt = (
        "You are a restaurant recommendation assistant for New York City. "
        "You have two sources of restaurant data: a curated dataset and live Foursquare results. "
        "Use both sources together to give the best recommendations. "
        "Do not invent restaurants outside the provided lists."
    )

    user_prompt = (
        f"User request: {query}\n\n"
        f"User taste profile:\n{_profile_to_text(user_profile)}\n\n"
        f"Curated dataset results:\n{csv_block}\n\n"
        f"Live Foursquare results:\n{fsq_block}\n\n"
        "Pick the best 3-5 restaurants across both sources. "
        "For each, give the name, why it fits the request, and why it fits the taste profile. "
        "End with one sentence summarizing the best overall pick."
    )

    return _chat(client, system_prompt, user_prompt)
 
