def retrieve_reviews(query, df, top_k=5):
    query = query.lower()
    
    results = []
    
    for text in df["text"]:
        if any(word in text.lower() for word in query.split()):
            results.append(text)
    
    return results[:top_k]
