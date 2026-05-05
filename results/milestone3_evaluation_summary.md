# Milestone 3 Evaluation Harness

This report supports Bilal's final contribution by creating a reproducible evaluation layer for the Food AI Companion recommendation pipeline.

Important limitation: the numeric metrics below are deterministic metadata-side checks over local restaurant records. They are not presented as human scores and they do not automatically grade generated language quality. The separate human-evaluation CSV should be filled using the actual outputs from `main.py` or the Streamlit demo.

## Compared Modes

- `baseline_query_only_proxy`: query-only recommendation behavior, used as the baseline reference.
- `taste_profile_proxy`: query plus persistent taste-profile behavior.
- `taste_profile_rag_proxy`: query plus taste profile plus explicit restaurant evidence quality, representing the RAG target behavior.

## Metric Definitions

- `relevance`: keyword and food/category match between the query and candidate restaurant metadata.
- `profile_alignment`: match to preferred cuisines and liked foods, with penalties for disliked foods.
- `category_diversity`: unique primary categories divided by recommendation count.
- `category_entropy`: normalized category spread.
- `novelty`: fraction of recommendations not repeated from a simulated accepted-history set.
- `grounding_quality`: completeness of the restaurant evidence available for RAG explanation.

## Average Metrics by Mode

| Mode | Relevance | Profile alignment | Diversity | Entropy | Novelty | Grounding quality |
|---|---:|---:|---:|---:|---:|---:|
| baseline_query_only_proxy | 0.6400 | 0.7325 | 0.3000 | 0.2162 | 0.8500 | 0.7585 |
| taste_profile_proxy | 0.5876 | 1.0000 | 0.2000 | 0.0000 | 0.7000 | 0.7900 |
| taste_profile_rag_proxy | 0.5683 | 1.0000 | 0.2500 | 0.1805 | 0.6500 | 0.9400 |

## How To Use This In The Final Report

Use the average-metrics table as system-side evidence that the project evaluates relevance, personalization, diversity, novelty, and grounding. Use `results/milestone3_human_eval_template.csv` for manual review of the actual generated outputs so the final report does not rely on invented evaluator scores.

## Candidate Recommendation Sets

## italian_pasta_budget

Query: I want cheap Italian food in NYC with good pasta and online ordering.

Profile: preferred=Italian, Pizza | liked=pasta, pizza | disliked=seafood | budget=cheap | occasion=casual dinner

### baseline_query_only_proxy

| Rank | Restaurant | Primary category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Amali NYC | Italian | pasta | 0.9134 |
| 2 | Tony's Di Napoli - Midtown | Italian | Pasta | 0.8446 |
| 3 | Red Rooster Harlem | Italian | pasta | 0.8274 |
| 4 | Sardi's Restaurant | Italian | pasta | 0.8252 |
| 5 | Amarone Scarlatto | Italian | pasta | 0.8246 |

### taste_profile_proxy

| Rank | Restaurant | Primary category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | 2 Bros Pizza | Italian | pizza | 0.8527 |
| 2 | Tony's Di Napoli - Midtown | Italian | Pasta | 0.8432 |
| 3 | Sardi's Restaurant | Italian | pasta | 0.8393 |
| 4 | Amarone Scarlatto | Italian | pasta | 0.8392 |
| 5 | Eataly Downtown | Italian | pasta | 0.839 |

### taste_profile_rag_proxy

| Rank | Restaurant | Primary category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Tony's Di Napoli - Midtown | Italian | Pasta | 0.9114 |
| 2 | Upland | Italian | pasta | 0.9114 |
| 3 | Amarone Scarlatto | Italian | pasta | 0.9114 |
| 4 | Bar Primi | Italian | pasta | 0.9114 |
| 5 | Trattoria Casa di Isacco | Italian | pasta | 0.9114 |

## korean_casual

Query: I am looking for a casual Korean restaurant in NYC with generous portions.

Profile: preferred=Korean | liked=bbq, noodles, rice | disliked=seafood | budget=moderate | occasion=casual dinner

### baseline_query_only_proxy

| Rank | Restaurant | Primary category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Miss Korea BBQ | Asian | korean bbq | 0.7116 |
| 2 | Jongro BBQ | Asian | korean bbq | 0.6038 |
| 3 | Gunbae Tribeca | Asian | korean bbq | 0.602 |
| 4 | KOBA Korean Bbq | Asian | korean bbq | 0.5891 |
| 5 | Madangsui Korean BBQ Restaurant | Asian | korean bbq | 0.5649 |

### taste_profile_proxy

| Rank | Restaurant | Primary category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Miss Korea BBQ | Asian | korean bbq | 0.818 |
| 2 | Gunbae Tribeca | Asian | korean bbq | 0.7675 |
| 3 | LOVE Korean BBQ | Asian | korean bbq | 0.75 |
| 4 | Woorijip | Asian | bbq | 0.7327 |
| 5 | Five Senses | Asian | bbq | 0.7326 |

### taste_profile_rag_proxy

| Rank | Restaurant | Primary category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Miss Korea BBQ | Asian | korean bbq | 0.8086 |
| 2 | Gyu-Kaku Japanese BBQ | Japanese | bbq | 0.7957 |
| 3 | Danji | Asian | kimchi | 0.7957 |
| 4 | KOBA Korean Bbq | Asian | korean bbq | 0.7757 |
| 5 | Gunbae Tribeca | Asian | korean bbq | 0.7657 |

## japanese_date_night

Query: Find me a date-night Japanese restaurant in NYC that is not too expensive.

Profile: preferred=Japanese, Sushi | liked=sushi, ramen | disliked=pizza | budget=moderate | occasion=date night

### baseline_query_only_proxy

| Rank | Restaurant | Primary category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Zuma | Japanese | sushi | 0.6375 |
| 2 | Sushi of Gari | Japanese | sushi | 0.6351 |
| 3 | Shabu-Tatsu | Japanese | sushi | 0.6343 |
| 4 | Mifune | Japanese | sushi | 0.6342 |
| 5 | Zaika NYC | Japanese | sushi | 0.5967 |

### taste_profile_proxy

| Rank | Restaurant | Primary category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Zuma | Japanese | sushi | 0.7832 |
| 2 | Zaika NYC | Japanese | sushi | 0.7651 |
| 3 | Bodhi Kosher Vegan Sushi & Dim Sum Restaurant | Japanese | sushi | 0.7351 |
| 4 | Hama Japanese Cuisine | Japanese | sushi | 0.7327 |
| 5 | Sushi of Gari | Japanese | sushi | 0.7327 |

### taste_profile_rag_proxy

| Rank | Restaurant | Primary category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Zuma | Japanese | sushi | 0.8536 |
| 2 | TAO Downtown Restaurant | Japanese | sushi | 0.7957 |
| 3 | Cha Pa's Noodles & Grill | Japanese | sushi | 0.7957 |
| 4 | Whole Foods Market | Japanese | sushi | 0.7957 |
| 5 | Hasaki | Japanese | omakase | 0.7957 |

## open_dinner_profile_comparison

Query: I want something good for dinner in NYC.

Profile: preferred=Mexican, Japanese, Korean | liked=tacos, sushi, rice, spicy food | disliked=heavy cream | budget=moderate | occasion=casual dinner

### baseline_query_only_proxy

| Rank | Restaurant | Primary category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Antique Garage | European | pastrami sandwich | 0.7019 |
| 2 | Carmine's Italian Restaurant - Times Square | Italian | tiramisu | 0.5059 |
| 3 | Bill's Bar & Burger | American | burger | 0.467 |
| 4 | Eleven Madison Park | American | salad | 0.4616 |
| 5 | Hard Rock Cafe | American | vegetarian | 0.4579 |

### taste_profile_proxy

| Rank | Restaurant | Primary category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | TAO Downtown Restaurant | Japanese | sushi | 0.698 |
| 2 | Zuma | Japanese | sushi | 0.6974 |
| 3 | Whole Foods Market | Japanese | sushi | 0.6968 |
| 4 | Bread & Butter | Japanese | sushi | 0.6968 |
| 5 | Flame Hibachi Downtown | Japanese | sushi | 0.6967 |

### taste_profile_rag_proxy

| Rank | Restaurant | Primary category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | TAO Downtown Restaurant | Japanese | sushi | 0.78 |
| 2 | Zuma | Japanese | sushi | 0.78 |
| 3 | Whole Foods Market | Japanese | sushi | 0.78 |
| 4 | Bread & Butter | Japanese | sushi | 0.705 |
| 5 | Flame Hibachi Downtown | Japanese | sushi | 0.705 |
