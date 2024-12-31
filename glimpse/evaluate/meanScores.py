import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt

def extract_model_name(filename: str) -> str:
    # Extract the model name: take the part after the second "_-_"
    parts = filename.split('_-_')
    if len(parts) >= 3:
        model_part = parts[1].split('_')  # Split the model part by '_'
        model_name = model_part[1]
        return model_name
    return None

def compute_mean_scores(files: list, score_field: str):
    model_scores = {}

    for file in files:
        model_name = extract_model_name(file)  
        if model_name:
            df = pd.read_csv(file)
            
            if score_field in df.columns:
                score_mean = df[score_field].mean() 

                if model_name in model_scores:
                    model_scores[model_name].append(score_mean)
                else:
                    model_scores[model_name] = [score_mean]
    
    model_mean_scores = {model: sum(scores) / len(scores) for model, scores in model_scores.items()}
    
    return model_mean_scores

def parse_args():
    parser = argparse.ArgumentParser(description="Compute the mean of a specified field for each model and plot it.")
    parser.add_argument('--directory_path', type=str, help="Path to the folder containing the CSV files.")
    parser.add_argument('--score_field', type=str, help="The field/column name for which to compute the mean.")
    return parser.parse_args()

def plot_bar_chart(mean_scores, score_field, output_path):
    models = list(mean_scores.keys())
    scores = list(mean_scores.values())

    plt.figure(figsize=(10, 6))
    plt.barh(models, scores, color='skyblue')  # Use barh() for horizontal bars
    plt.ylabel('Model')
    plt.xlabel(f'Mean {score_field}')
    plt.title(f'Mean {score_field} by Model')
    plt.tight_layout()

    # Save the plot
    plt.savefig(str(output_path) + "/_mean_" + str(score_field) + ".png")  # Save as PNG or other format
    plt.close()

def main():
    args = parse_args()
    
    files = [os.path.join(args.directory_path, f) for f in os.listdir(args.directory_path) if f.endswith(".csv")]

    mean_scores = compute_mean_scores(files, args.score_field)
    plot_bar_chart(mean_scores, args.score_field, args.directory_path)

    print(f"Bar plot saved at: {args.directory_path}")

if __name__ == '__main__':
    main()
