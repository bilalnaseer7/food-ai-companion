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
        "Recommend 3 NYC restaurant options or restaurant types. "
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
