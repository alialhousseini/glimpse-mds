import pandas as pd
import pickle

def create_and_save_dataframes(data_dict):
    """
    Creates DataFrames from a dictionary of data and saves them to a pickle file.

    Args:
        data_dict: A dictionary where keys are IDs and values are dictionaries 
                   representing data for each DataFrame.
                   Each inner dictionary has the same structure as the original 
                   example (keys: 'Summary A', 'Summary B', 'Summary C'; 
                   values: dictionaries with likelihood scores for each text).
    """
    results_dict = {}
    for id, data in data_dict.items():
        df = pd.DataFrame(data)
        results_dict[id] = df

    with open('results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)

# Example usage:
data_dict = {
    'id1': {
        'Summary A': {'Text 1': 0.8, 'Text 2': 0.2, 'Text 3': 0.5},
        'Summary B': {'Text 1': 0.1, 'Text 2': 0.7, 'Text 3': 0.3},
        'Summary C': {'Text 1': 0.3, 'Text 2': 0.1, 'Text 3': 0.9}
    },
    'id2': {
        'Summary A': {'Text 1': 0.6, 'Text 2': 0.4, 'Text 3': 0.1},
        'Summary B': {'Text 1': 0.2, 'Text 2': 0.5, 'Text 3': 0.8},
        'Summary C': {'Text 1': 0.9, 'Text 2': 0.1, 'Text 3': 0.2}
    }
}


def testSaveFile():
    with open(f'preCompProb/extractive_sentences-_-all_merged_filtered_4_no_unique-_-none-_-2024-12-26-18-06-23.pk', 'rb') as f:
        results_dict = pickle.load(f)
    print(results_dict['id1'])


"""import pandas as pd

# Prepare toy CSV data based on the structure of the pk file
toy_data = []
with open(f'preCompProb/extractive_sentences-_-all_merged_filtered_4_no_unique-_-none-_-2024-12-26-18-06-23.pk', 'rb') as f:
        results_dict = pickle.load(f)
for id_key, df in results_dict.items():
    for text in df.index:
        for summary in df.columns:
            toy_data.append({
                "index": len(toy_data),  # Unique index for each row
                "id": id_key,
                "text": text,
                "gold": "invented_gold_value",  # Placeholder gold value
                "summary": summary,
                "id_candidate": f"{id_key}_{summary}"
            })

# Convert to a pandas DataFrame
toy_df = pd.DataFrame(toy_data)

# Save to a CSV file for inspection
toy_csv_path = "toy_data.csv"
toy_df.to_csv(toy_csv_path, index=False)

toy_df.head(), toy_csv_path"""


"""
import pickle
import pandas as pd

# Load the file
file_path = 'D:/Universita/Progetto NLP/model_evaluation/NLP-Project/preCompProb/all_merged_226_-_extr-_-BART.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Check the top-level structure
print("Top-level type:", type(data))

# If the top-level structure is a dictionary, examine its keys
if isinstance(data, dict):
    print("Keys in dictionary:", list(data.keys()))
    for key, value in data.items():
        print(f"Key: {key}, Type: {type(value)}")
        if key == 'results' and isinstance(value, list):
            # Analyze the results list
            print(f"'results' contains {len(value)} elements.")
            if len(value) > 0:
                print("Type of first element in 'results':", type(value[0]))

                # Check the structure of the first element
                first_result = value[0]
                if isinstance(first_result, dict):
                    print("Keys in the first element of 'results':", list(first_result.keys()))
                    for sub_key, sub_value in first_result.items():
                        print(f"Sub-key: {sub_key}, Type: {type(sub_value)}")
                        if isinstance(sub_value, pd.DataFrame):
                            print(f"Columns in DataFrame for sub-key '{sub_key}': {sub_value.columns.tolist()}")
                            print(f"DataFrame shape for sub-key '{sub_key}': {sub_value.shape}")
                elif isinstance(first_result, pd.DataFrame):
                    print("Columns in first DataFrame in 'results':", first_result.columns.tolist())
                    print("Shape of first DataFrame in 'results':", first_result.shape)


results_by_id = {result['id'][0]: result['language_model_proba_df'] for result in data['results']}
print(len(results_by_id))"""




import pandas as pd

# Load the CSV file
csv_file = "D:\\Universita\\Progetto NLP\\model_evaluation\\NLP-Project\\allSummariesGSpeaker\\all_merged_226_-_extr-_-BART_glimpseSpeaker_all_merged_226_-_extr.csv"  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Count the number of unique IDs
unique_ids_count = data['Id'].nunique()

print(f"The number of unique IDs is: {unique_ids_count}")