import pandas as pd

def load_reviews(path="data/reviews.csv"):
    df = pd.read_csv(path)
    df = df[["text", "rating", "business_name"]]
    df = df.dropna(subset=["text"])
    return df
  
