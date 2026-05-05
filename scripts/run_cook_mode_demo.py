import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cook_mode import build_cook_context, build_cook_prompt, generate_cook_recommendations


def demo_cases() -> list[dict]:
    return [
        {
            "name": "Weeknight Korean comfort dinner",
            "craving": "quick Korean comfort dinner",
            "profile": {
                "pantry": ["eggs", "rice", "spinach", "soy sauce", "garlic"],
                "preferred_cuisines": ["Korean"],
                "liked_foods": ["rice", "spicy food", "savory sauces"],
                "disliked_foods": ["seafood"],
                "dietary_restrictions": [],
                "budget": "budget",
                "time_preference": "under 30 minutes",
                "occasion": "weeknight dinner",
                "meal_type": "dinner",
            },
        },
        {
            "name": "Vegetarian Italian pantry meal",
            "craving": "cozy Italian dinner without meat",
            "profile": {
                "pantry": ["pasta", "canned tomatoes", "onion", "spinach", "parmesan"],
                "preferred_cuisines": ["Italian"],
                "liked_foods": ["pasta", "tomato sauce", "cheese"],
                "disliked_foods": ["mushrooms"],
                "dietary_restrictions": ["vegetarian"],
                "budget": "moderate",
                "time_preference": "under 45 minutes",
                "occasion": "casual dinner",
                "meal_type": "dinner",
            },
        },
        {
            "name": "Low-effort lunch from leftovers",
            "craving": "healthy lunch that uses leftovers",
            "profile": {
                "pantry": ["chicken", "brown rice", "avocado", "lettuce", "lime"],
                "preferred_cuisines": ["Mexican", "Mediterranean"],
                "liked_foods": ["bowls", "fresh herbs", "citrus"],
                "disliked_foods": ["heavy cream"],
                "dietary_restrictions": [],
                "budget": "budget",
                "time_preference": "under 20 minutes",
                "occasion": "work lunch",
                "meal_type": "lunch",
            },
        },
    ]


def _format_values(values) -> str:
    if isinstance(values, list):
        return ", ".join(values) if values else "none"
    return str(values) if values else "none"


def render_preview_case(case: dict) -> str:
    context = build_cook_context(case["craving"], case["profile"])
    system_prompt, user_prompt = build_cook_prompt(context)

    lines = [
        f"## {case['name']}",
        "",
        "### Input Context",
        f"- Craving: {context['craving']}",
        f"- Available ingredients: {_format_values(context['available_ingredients'])}",
        f"- Preferred cuisines: {_format_values(context['preferred_cuisines'])}",
        f"- Liked foods/flavors: {_format_values(context['liked_foods'])}",
        f"- Disliked foods: {_format_values(context['disliked_foods'])}",
        f"- Dietary restrictions: {_format_values(context['dietary_restrictions'])}",
        f"- Budget: {context['budget']}",
        f"- Time preference: {context['time_preference']}",
        f"- Occasion: {context['occasion']}",
        "",
        "### Grounding",
        f"- {context['grounding_note']}",
        "",
        "### Prompt Contract",
        "The model must return 2 to 3 recipe ideas with recipe name, taste-profile fit, pantry ingredients used, missing/substitute ingredients, quick steps, and a caution.",
        "",
        "<details>",
        "<summary>Full prompt sent to the model</summary>",
        "",
        "```text",
        system_prompt,
        "",
        user_prompt,
        "```",
        "",
        "</details>",
        "",
    ]
    return "\n".join(lines)


def render_generated_case(case: dict, output: str) -> str:
    context = build_cook_context(case["craving"], case["profile"])
    lines = [
        f"## {case['name']}",
        "",
        f"- Craving: {context['craving']}",
        f"- Available ingredients: {_format_values(context['available_ingredients'])}",
        f"- Preferred cuisines: {_format_values(context['preferred_cuisines'])}",
        f"- Disliked foods/restrictions: {_format_values(context['disliked_foods'] + context['dietary_restrictions'])}",
        f"- Budget/time: {context['budget']} / {context['time_preference']}",
        "",
        "### Model Output",
        output,
        "",
    ]
    return "\n".join(lines)


def write_preview(cases: list[dict], output_path: Path) -> None:
    sections = [
        "# Cook at Home Mode Preview",
        "",
        "This file is a deterministic preview of Bilal's Cook at Home mode inputs and prompt contract. It is not a model-generated recipe result and does not contain fabricated scores or outputs.",
        "",
    ]
    sections.extend(render_preview_case(case) for case in cases)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")


def write_generated_outputs(cases: list[dict], output_path: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found. Add it to .env or run without --generate.")

    from openai import OpenAI

    client = OpenAI()
    sections = [
        "# Cook at Home Mode Generated Outputs",
        "",
        "These outputs were generated by the Cook at Home mode using the local OpenAI API key. They are LLM + taste-profile outputs, not recipe-dataset RAG outputs.",
        "",
    ]
    for case in cases:
        output = generate_cook_recommendations(client, case["craving"], case["profile"])
        sections.append(render_generated_case(case, output))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or preview Cook at Home Milestone 3 demo cases.")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Call the OpenAI API and write real generated recipe outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit the number of demo cases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to results/cook_mode_demo_preview.md or results/cook_mode_demo_outputs.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = demo_cases()
    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]

    if args.generate:
        output_path = args.output or ROOT / "results" / "cook_mode_demo_outputs.md"
        write_generated_outputs(cases, output_path)
    else:
        output_path = args.output or ROOT / "results" / "cook_mode_demo_preview.md"
        write_preview(cases, output_path)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
