"""
This script is responsible to generate the dsicriminative score, "Proba_of_success" for each summary in the dataset.
This differs from `discriminative_classification_sumy` as this is used for the datasets of extractive & abstractive summaries.
The reason behind separating the two scripts. Is that here our focus is to generate the discriminative score for the summaries while iterating over the dataset in a folder.
While in the other case, different summarization techniques are used in a single csv file.

The results obtained from this script are added in the column "Proba_of_success" in the dataset.
Link: https://drive.google.com/drive/folders/1JyfLRWvLf0AW7dTBKZ6JtCBHJo9_JW0C?usp=drive_link
"""

from typing import Tuple
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
# Helper function for x * log(x), handling the case of x == 0


def xlogx(x):
    if x == 0:
        return 0
    else:
        return x * torch.log(x)


def parse_summaries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'id' not in df.columns:
        raise ValueError('Id column not found in the summaries file')
    if 'text' not in df.columns:
        raise ValueError('text column not found in the summaries file')
    if 'summary' not in df.columns:
        raise ValueError('summary column not found in the summaries file')
    return df


def embed_text_and_summaries(df: pd.DataFrame, model: SentenceTransformer) -> Tuple[torch.Tensor, torch.Tensor]:
    text_embeddings = model.encode(df.text.tolist(), convert_to_tensor=True)
    summary_embeddings = model.encode(
        df.summary.tolist(), convert_to_tensor=True)
    return text_embeddings, summary_embeddings


def compute_dot_products(df: pd.DataFrame, text_embeddings: torch.Tensor, summary_embeddings: torch.Tensor):
    metrics = {'proba_of_success': []}
    for idx, row in df.iterrows():
        text_embedding = text_embeddings[idx]
        summary_embedding = summary_embeddings[idx]
        dot_product = torch.dot(text_embedding, summary_embedding)
        metrics['proba_of_success'].append(dot_product.item())
    df['proba_of_success'] = metrics['proba_of_success']
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_folder', type=Path, required=True,
                        help="Folder containing the summary CSV files.")
    parser.add_argument('--model', type=str, default='paraphrase-MiniLM-L6-v2',
                        help="Model to use for embedding.")
    parser.add_argument('--output_folder', type=Path, required=True,
                        help="Folder to save the output CSV files.")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to run the model on (e.g., 'cuda' or 'cpu').")
    args = parser.parse_args()
    return args


def process_files_in_folder(input_folder: Path, output_folder: Path, model: SentenceTransformer):
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    for summary_file in tqdm(input_folder.glob("*.csv")):
        print(f"Processing file: {summary_file.name}")

        df = parse_summaries(summary_file)

        text_embeddings, summary_embeddings = embed_text_and_summaries(
            df, model)

        df = compute_dot_products(df, text_embeddings, summary_embeddings)

        output_path = output_folder / f"{summary_file.stem}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved metrics to: {output_path}")


def main():
    args = parse_args()
    model = SentenceTransformer(args.model, device=args.device)
    input_folder = args.summaries_folder
    output_folder = args.output_folder

    print(f"Processing files in {input_folder}")
    process_files_in_folder(input_folder, output_folder, model)


if __name__ == '__main__':
    main()
