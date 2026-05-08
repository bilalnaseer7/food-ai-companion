# Reddit Grounding Artifact

This Milestone 3 artifact supports Bilal's external-grounding contribution by collecting public Reddit discussion signals for Food AI Companion restaurant queries.

It is a supplementary analysis artifact, not a live ranking layer in the Streamlit app. Reddit mentions are treated as noisy community signals and are not presented as ground truth.

## Collection Status

Live Reddit collection was not run.

- Missing Reddit credentials: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT. Add them to `.env` to run live collection.
- Expected `.env` keys: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT.

## Method

- Search public Reddit posts for fixed restaurant-intent queries aligned with the Milestone 2/3 evaluation cases.
- Store only post metadata, snippets, URLs, matched local restaurant names, and cuisine keywords.
- Match restaurant names against the local `data/restaurants.csv` inventory to avoid inventing unsupported restaurant entities.
- Use the artifact for final-report discussion of external community grounding, not as fabricated evaluation scores.

## Query Coverage

| Case | Posts | Query |
| --- | --- | --- |
| italian_pasta_budget | 0 | cheap Italian pasta restaurant NYC |
| korean_casual | 0 | casual Korean restaurant NYC generous portions |
| japanese_date_night | 0 | Japanese restaurant NYC date night not too expensive |
| open_dinner_recommendation | 0 | best dinner restaurant recommendation NYC |

## Top Cuisine Signals

No cuisine terms collected yet.

## Top Local Restaurant Mentions

No local restaurant matches collected yet.

## Subreddit Mix

No subreddit rows collected yet.

## Limitations

- Reddit discussions are informal, biased toward active posters, and may include outdated recommendations.
- Name matching is conservative and only counts restaurants already present in the local restaurant dataset.
- This artifact does not replace Google Places, RAG retrieval, or human evaluation.
- The CSV may be empty if Reddit credentials are not configured; that is intentional and avoids fake results.

## Milestone 3 Report Note

Bilal added a Reddit/PRAW grounding artifact that collects public community discussion signals, maps them back to local restaurant metadata, and produces report-ready CSV/Markdown outputs without changing the existing Eat Out or Cook at Home app flow.
