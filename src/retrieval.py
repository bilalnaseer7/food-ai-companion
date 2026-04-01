def retrieve_reviews(query, df, top_k=5):
    query_words = query.lower().split()
    results = []

    for _, row in df.iterrows():
        text = str(row["text"]).lower()
        if any(word in text for word in query_words):
            results.append({
                "business_name": row["business_name"],
                "rating": row["rating"],
                "text": row["text"]
            })

    return results[:top_k]
