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
    return df


def embed_text_and_summaries(df: pd.DataFrame, model: SentenceTransformer, col_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    text_embeddings = model.encode(df.text.tolist(), convert_to_tensor=True)
    summary_embeddings = model.encode(
        df[col_name].tolist(), convert_to_tensor=True)
    return text_embeddings, summary_embeddings


def compute_dot_products(df: pd.DataFrame, text_embeddings: torch.Tensor, summary_embeddings: torch.Tensor, method: str):
    metrics = {'proba_of_success': []}
    for idx, row in df.iterrows():
        text_embedding = text_embeddings[idx]
        summary_embedding = summary_embeddings[idx]
        dot_product = torch.dot(text_embedding, summary_embedding)
        metrics['proba_of_success'].append(dot_product.item())
    df[f'proba_of_success_{method}'] = metrics['proba_of_success']
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
        method_summ = [col for col in df.columns if 'summary' in col]
        method_summ = [col.split('_')[-1] for col in method_summ]
        method_summ = list(set(method_summ))

        for method in method_summ:
            text_embeddings, summary_embeddings = embed_text_and_summaries(
                df, model, f'summary_{method}')

            df = compute_dot_products(
                df, text_embeddings, summary_embeddings, method)

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
