import pandas as pd
import os

data_glimpse = "data/processed/"
if not os.path.exists(data_glimpse):
    os.makedirs(data_glimpse)

for year in range (2017, 2018):
    dataset = pd.read_csv(f"D:/Universita/Progetto NLP/model_evaluation/NLP-Project/data/all_merged_filtered_4_no_unique.csv")
    #sub_dataset = dataset[['id','review', 'metareview']]
    #sub_dataset.rename(columns={"review": "text", "metareview": "gold"}, inplace=True)
    
    sub_dataset = dataset[['id','text_cleaned', 'gold_cleaned']]
    sub_dataset.rename(columns={"text_cleaned": "text", "gold_cleaned": "gold"}, inplace=True)

    sub_dataset.to_csv(f"D:/Universita/Progetto NLP/model_evaluation/NLP-Project/{data_glimpse}all_merged_filtered_4_no_unique.csv", index=False)
    
    