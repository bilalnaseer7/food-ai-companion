from retrieval import retrieve_reviews

def recommend(query, df):
    retrieved = retrieve_reviews(query, df)
    
    context = "\n".join(retrieved)
    
    prompt = f"""
User wants: {query}

Here are similar reviews:
{context}

Recommend a restaurant or food choice based on this.
Explain briefly.
"""
    
    return prompt  # for now just return prompt
