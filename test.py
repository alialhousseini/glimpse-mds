import pandas as pd
import re



uniqueSum = pd.read_csv("pegasus-arxiv_glimpseSpeaker_extractive_sentences-_-all_reviews_2017-_-none-_-2024-12-24-08-25-55.csv")

print(uniqueSum.count())

for index, row in uniqueSum.iterrows():
    cleaned_summary = re.sub(r'\s+', ' ', row['summary']).strip()
    print(f"Summary: {cleaned_summary}")
    cleaned_text = re.sub(r'\s+', ' ', row['text']).strip()
    print(f"Text: {cleaned_text}")
    
    print("\n-----------------------\n")
    break