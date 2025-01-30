"""
Used to evaluate the performance of the models on the common metrics dataset. (ROUGE 1, ROUGE 2, ROUGE L, ROUGE Lsum)
It reads the summary files from the input folder, evaluates the metrics and saves the results in the output folder.
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from rouge_score import rouge_scorer


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name to be used as a folder name.
    @param model_name: The model name
    @return: The sanitized model name
    """
    return model_name.replace("/", "_")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries_folder", type=Path,
                        required=True, help="Folder containing summary files.")
    parser.add_argument("--output_folder", type=Path,
                        required=True, help="Folder to save the output metrics.")

    args = parser.parse_args()
    return args


def parse_summaries(path: Path):
    """
    :return: a pandas dataframe with at least the columns 'text' and 'summary'
    """
    # read csv file

    df = pd.read_csv(path).dropna()

    # check if the csv file has the correct columns
    if not all([col in df.columns for col in ["gold", "summary"]]):
        raise ValueError(
            "The csv file must have the columns 'text' and 'summary'.")

    return df


def evaluate_rouge(
    df,
):
    # make a list of the tuples (text, summary)

    texts = df.gold.tolist()
    summaries = df.summary.tolist()

    # rouges
    metrics = {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []}

    rouges = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )

    metrics["rouge1"].extend(
        [
            rouges.score(summary, text)["rouge1"].fmeasure
            for summary, text in zip(summaries, texts)
        ]
    )
    metrics["rouge2"].extend(
        [
            rouges.score(summary, text)["rouge2"].fmeasure
            for summary, text in zip(summaries, texts)
        ]
    )
    metrics["rougeL"].extend(
        [
            rouges.score(summary, text)["rougeL"].fmeasure
            for summary, text in zip(summaries, texts)
        ]
    )
    metrics["rougeLsum"].extend(
        [
            rouges.score(summary, text)["rougeLsum"].fmeasure
            for summary, text in zip(summaries, texts)
        ]
    )

    # compute the mean of the metrics
    # metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

    return metrics


def process_files_in_folder(input_folder: Path, output_folder: Path):
    """
    Process all summary files in a folder.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    for summary_file in tqdm(input_folder.glob("*.csv"), desc=f"Processing"):
        print(f"Processing file: {summary_file.name}")

        # Parse the summaries
        df = parse_summaries(summary_file)

        # Evaluate metrics
        metrics = evaluate_rouge(df)
        for k, v in metrics.items():
            df[k] = v

        # Save results to a new file
        output_path = output_folder / f"{summary_file.stem}_metrics.csv"
        df.to_csv(output_path, index=False)


def main():
    args = parse_args()

    input_folder = args.summaries_folder
    output_folder = args.output_folder

    process_files_in_folder(input_folder, output_folder)


main()
