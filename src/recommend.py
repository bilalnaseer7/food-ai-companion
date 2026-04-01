from src.retrieval import retrieve_reviews

def recommend(query, df):
    retrieved = retrieve_reviews(query, df)

    if not retrieved:
        return "No matching reviews were found for this query."

    output = []
    output.append(f"User query: {query}\n")
    output.append("Top retrieved reviews:\n")

    for i, item in enumerate(retrieved, start=1):
        output.append(
            f"{i}. {item['business_name']} | Rating: {item['rating']}\n"
            f"   Review: {item['text']}\n"
        )

    return "\n".join(output)
