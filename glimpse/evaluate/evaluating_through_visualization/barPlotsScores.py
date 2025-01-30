import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def process_files_for_prefixes(directory_path, names_directory):
    prefixes = []
    for file_name in os.listdir(names_directory):
        if file_name.endswith(".csv"): 
            prefix = file_name.replace(".csv", "") 
            prefixes.append(prefix)

    for probability_prefix in prefixes:
        results = []
        for file_name in os.listdir(directory_path):
            if file_name.startswith(probability_prefix) and file_name.endswith(".csv"):
                model_name = file_name.replace(probability_prefix, "").replace(".csv", "").strip("_")
                model_name = (model_name.split('_')[0] + "-" + model_name.split('_')[1]).strip('-')

                file_path = os.path.join(directory_path, file_name)
                try:
                    data = pd.read_csv(file_path)
                    if "BERTScore" in data.columns:
                        avg_score = data["BERTScore"].mean()
                        results.append({"Model": model_name, "AverageScore": avg_score})
                    else:
                        print(f"Skipping {file_name}: 'BERTScore' column not found.")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

        if results:
            results_df = pd.DataFrame(results)
            print(f"\nResults for prefix: {probability_prefix}")
            print(results_df.head())
            results_df.sort_values(by="AverageScore", ascending=True, inplace=True)

            plt.figure(figsize=(10, 6))
            plt.barh(results_df["Model"], results_df["AverageScore"], color="skyblue")
            plt.xlabel("Average BERTScore")
            plt.ylabel("Model")
            plt.title(f"Average BERTScore for Models ({probability_prefix})")
            plt.tight_layout()

            save_path = os.path.join(directory_path, f"{probability_prefix}_BERTScore_barplot.png")
            plt.savefig(save_path)
            print(f"Saved BERTScore bar plot for {probability_prefix} at {save_path}")
            plt.close()
        else:
            print(f"No valid data found to process for prefix: {probability_prefix}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files for multiple prefixes.")
    parser.add_argument("--directory_path", type=str, help="Path to the directory containing the evaluation files.")
    parser.add_argument("--candidates_directory", type=str, help="Path to the directory containing the prefix files.")
    args = parser.parse_args()

    process_files_for_prefixes(args.directory_path, args.candidates_directory)
