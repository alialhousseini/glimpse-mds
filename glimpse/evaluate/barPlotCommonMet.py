import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def process_files(directory_path, names_directory):
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
                data = pd.read_csv(file_path)
                
                avg_rouge1 = data['rouge1'].mean()
                avg_rouge2 = data['rouge2'].mean()
                avg_rougeL = data['rougeL'].mean()
                avg_rougeLsum = data['rougeLsum'].mean()
                
                results.append({
                    "Model": model_name,
                    "rouge1": avg_rouge1,
                    "rouge2": avg_rouge2,
                    "rougeL": avg_rougeL,
                    "rougeLsum": avg_rougeLsum
                })

        results_df = pd.DataFrame(results)

        metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            results_df.sort_values(by=metric, ascending=True, inplace=True)
            plt.barh(results_df["Model"], results_df[metric], color="lightgreen")
            plt.xlabel(f"Average {metric}")
            plt.ylabel("Model")
            plt.title(f"Average {metric} for Models ({probability_prefix})")
            plt.tight_layout()

            save_path = os.path.join(directory_path, f"{probability_prefix}_{metric}_barplot.png")
            plt.savefig(save_path)  # Save the figure
            print(f"Saved {metric} bar plot at {save_path}")

            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process evaluation files based on prefixes.")
    parser.add_argument("--directory_path", type=str, help="Path to the directory containing the evaluation files.")
    parser.add_argument("--candidates_directory", type=str, help="Path to the directory containing the prefix files.")

    args = parser.parse_args()
    process_files(args.directory_path, args.candidates_directory)
