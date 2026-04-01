from data_loader import load_reviews
from recommend import recommend

df = load_reviews()

query = "cheap pizza with good taste"
result = recommend(query, df)

print(result)
