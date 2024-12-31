import argparse
from pathlib import Path
import pandas as pd
from bert_score import BERTScorer
from tqdm import tqdm


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries_folder", type=Path,
                        required=True, help="Folder containing summary files.")
    parser.add_argument("--output_folder", type=Path,
                        required=True, help="Folder to save the output metrics.")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def parse_summaries(path: Path):
    df = pd.read_csv(path).dropna()
    if not all([col in df.columns for col in ["gold", "summary"]]):
        raise ValueError(
            "The csv file must have the columns 'text' and 'summary'.")
    return df


def evaluate_bartbert(df, scorer):
    texts = df.gold.tolist()
    summaries = df.summary.tolist()
    metrics = {'BERTScore': []}
    for i in range(len(texts)):
        texts[i] = texts[i].replace("\n", " ")
        summaries[i] = summaries[i].replace("\n", " ")
        P, R, F1 = scorer.score([summaries[i]], [texts[i]])
        metrics['BERTScore'].append(F1.mean().item())
    return metrics


def process_files_in_folder(input_folder: Path, output_folder: Path, scorer):
    output_folder.mkdir(parents=True, exist_ok=True)
    for summary_file in tqdm(input_folder.glob("*.csv")):
        print(f"Processing file: {summary_file.name}")
        df = parse_summaries(summary_file)
        metrics = evaluate_bartbert(df, scorer)
        df['BERTScore'] = metrics['BERTScore']
        df.to_csv(summary_file, index=False)


def main():
    args = parse_args()
    scorer = BERTScorer(
        lang="en", rescale_with_baseline=True, device=args.device)
    process_files_in_folder(args.summaries_folder, args.output_folder, scorer)


if __name__ == "__main__":
    main()
