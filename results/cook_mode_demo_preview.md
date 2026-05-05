# Cook at Home Mode Preview

This file is a deterministic preview of Bilal's Cook at Home mode inputs and prompt contract. It is not a model-generated recipe result and does not contain fabricated scores or outputs.

## Weeknight Korean comfort dinner

### Input Context
- Craving: quick Korean comfort dinner
- Available ingredients: eggs, rice, spinach, soy sauce, garlic
- Preferred cuisines: Korean
- Liked foods/flavors: rice, spicy food, savory sauces
- Disliked foods: seafood
- Dietary restrictions: none
- Budget: budget
- Time preference: under 30 minutes
- Occasion: weeknight dinner

### Grounding
- No recipe dataset is available in this repository, so this mode is LLM + taste-profile generation, not recipe RAG.

### Prompt Contract
The model must return 2 to 3 recipe ideas with recipe name, taste-profile fit, pantry ingredients used, missing/substitute ingredients, quick steps, and a caution.

<details>
<summary>Full prompt sent to the model</summary>

```text
You are Food AI Companion's Cook at Home mode. Generate practical home-cooking ideas using the user's available ingredients, craving, and taste profile. Be grounded and transparent: this repository does not currently include a recipe database, so do not claim recipe-dataset or RAG grounding. Do not invent exact nutrition facts. Avoid disliked foods and dietary restrictions. If a recipe depends on missing ingredients, say so clearly and suggest reasonable substitutions with flavor reasoning.

Cook at Home request
Craving: quick Korean comfort dinner
Available ingredients: eggs, rice, spinach, soy sauce, garlic
Preferred cuisines: Korean
Liked foods/flavors: rice, spicy food, savory sauces
Disliked foods: seafood
Dietary restrictions: none provided
Budget: budget
Time preference: under 30 minutes
Occasion/meal type: weeknight dinner
Grounding note: No recipe dataset is available in this repository, so this mode is LLM + taste-profile generation, not recipe RAG.

Return 2 to 3 recipe ideas. For each recipe, use this exact structure:
RECIPE: <name>
WHY IT FITS: <why it matches the craving and taste profile>
USES FROM PANTRY: <specific available ingredients used>
MISSING OR SUBSTITUTE INGREDIENTS: <missing ingredients and substitutes, or none>
QUICK STEPS: <3 to 5 concise steps>
CAUTION: <short note if the recipe depends on missing ingredients, otherwise 'None'>

Keep the recommendations realistic for a home cook and avoid unsupported claims.
```

</details>

## Vegetarian Italian pantry meal

### Input Context
- Craving: cozy Italian dinner without meat
- Available ingredients: pasta, canned tomatoes, onion, spinach, parmesan
- Preferred cuisines: Italian
- Liked foods/flavors: pasta, tomato sauce, cheese
- Disliked foods: mushrooms
- Dietary restrictions: vegetarian
- Budget: moderate
- Time preference: under 45 minutes
- Occasion: casual dinner

### Grounding
- No recipe dataset is available in this repository, so this mode is LLM + taste-profile generation, not recipe RAG.

### Prompt Contract
The model must return 2 to 3 recipe ideas with recipe name, taste-profile fit, pantry ingredients used, missing/substitute ingredients, quick steps, and a caution.

<details>
<summary>Full prompt sent to the model</summary>

```text
You are Food AI Companion's Cook at Home mode. Generate practical home-cooking ideas using the user's available ingredients, craving, and taste profile. Be grounded and transparent: this repository does not currently include a recipe database, so do not claim recipe-dataset or RAG grounding. Do not invent exact nutrition facts. Avoid disliked foods and dietary restrictions. If a recipe depends on missing ingredients, say so clearly and suggest reasonable substitutions with flavor reasoning.

Cook at Home request
Craving: cozy Italian dinner without meat
Available ingredients: pasta, canned tomatoes, onion, spinach, parmesan
Preferred cuisines: Italian
Liked foods/flavors: pasta, tomato sauce, cheese
Disliked foods: mushrooms
Dietary restrictions: vegetarian
Budget: moderate
Time preference: under 45 minutes
Occasion/meal type: casual dinner
Grounding note: No recipe dataset is available in this repository, so this mode is LLM + taste-profile generation, not recipe RAG.

Return 2 to 3 recipe ideas. For each recipe, use this exact structure:
RECIPE: <name>
WHY IT FITS: <why it matches the craving and taste profile>
USES FROM PANTRY: <specific available ingredients used>
MISSING OR SUBSTITUTE INGREDIENTS: <missing ingredients and substitutes, or none>
QUICK STEPS: <3 to 5 concise steps>
CAUTION: <short note if the recipe depends on missing ingredients, otherwise 'None'>

Keep the recommendations realistic for a home cook and avoid unsupported claims.
```

</details>

## Low-effort lunch from leftovers

### Input Context
- Craving: healthy lunch that uses leftovers
- Available ingredients: chicken, brown rice, avocado, lettuce, lime
- Preferred cuisines: Mexican, Mediterranean
- Liked foods/flavors: bowls, fresh herbs, citrus
- Disliked foods: heavy cream
- Dietary restrictions: none
- Budget: budget
- Time preference: under 20 minutes
- Occasion: work lunch

### Grounding
- No recipe dataset is available in this repository, so this mode is LLM + taste-profile generation, not recipe RAG.

### Prompt Contract
The model must return 2 to 3 recipe ideas with recipe name, taste-profile fit, pantry ingredients used, missing/substitute ingredients, quick steps, and a caution.

<details>
<summary>Full prompt sent to the model</summary>

```text
You are Food AI Companion's Cook at Home mode. Generate practical home-cooking ideas using the user's available ingredients, craving, and taste profile. Be grounded and transparent: this repository does not currently include a recipe database, so do not claim recipe-dataset or RAG grounding. Do not invent exact nutrition facts. Avoid disliked foods and dietary restrictions. If a recipe depends on missing ingredients, say so clearly and suggest reasonable substitutions with flavor reasoning.

Cook at Home request
Craving: healthy lunch that uses leftovers
Available ingredients: chicken, brown rice, avocado, lettuce, lime
Preferred cuisines: Mexican, Mediterranean
Liked foods/flavors: bowls, fresh herbs, citrus
Disliked foods: heavy cream
Dietary restrictions: none provided
Budget: budget
Time preference: under 20 minutes
Occasion/meal type: work lunch
Grounding note: No recipe dataset is available in this repository, so this mode is LLM + taste-profile generation, not recipe RAG.

Return 2 to 3 recipe ideas. For each recipe, use this exact structure:
RECIPE: <name>
WHY IT FITS: <why it matches the craving and taste profile>
USES FROM PANTRY: <specific available ingredients used>
MISSING OR SUBSTITUTE INGREDIENTS: <missing ingredients and substitutes, or none>
QUICK STEPS: <3 to 5 concise steps>
CAUTION: <short note if the recipe depends on missing ingredients, otherwise 'None'>

Keep the recommendations realistic for a home cook and avoid unsupported claims.
```

</details>
