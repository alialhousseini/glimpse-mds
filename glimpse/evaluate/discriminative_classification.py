from typing import Tuple
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

def xlogx(x):
    if x == 0:
        return 0
    else:
        return x * torch.log(x)

def parse_summaries(path : Path):
    df = pd.read_csv(path)
    if 'Id' not in df.columns:
        raise ValueError('Id column not found in the summaries file')
    if 'text' not in df.columns:
        raise ValueError('text column not found in the summaries file')
    if 'summary' not in df.columns:
        raise ValueError('summary column not found in the summaries file')
    return df

def embed_text_and_summaries(df : pd.DataFrame, model : SentenceTransformer) -> Tuple[torch.Tensor, torch.Tensor]:
    text_embeddings = model.encode(df.text.tolist(), convert_to_tensor=True)
    summary_embeddings = model.encode(df.summary.tolist(), convert_to_tensor=True)
    return text_embeddings, summary_embeddings

def compute_dot_products(df : pd.DataFrame, text_embeddings : torch.Tensor, summary_embeddings : torch.Tensor):
    metrics = {'proba_of_success' : []}
    for Idx, row in df.iterrows():
        text_embedding = text_embeddings[idx]
        summary_embedding = summary_embeddings[idx]
        dot_product = torch.matmul(text_embedding.unsqueeze(0), summary_embedding.unsqueeze(0).T)
        log_softmax = torch.nn.functional.log_softmax(dot_product, dim=0)
        log_proba_of_success = log_softmax.squeeze().item()
        metrics['proba_of_success'].append(log_proba_of_success)
    df['proba_of_success'] = metrics['proba_of_success']
    return df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries', type=Path, required=True)
    parser.add_argument('--model', type=str, default='paraphrase-MiniLM-L6-v2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model = SentenceTransformer(args.model, device=args.device)
    df = parse_summaries(args.summaries)
    text_embeddings, summary_embeddings = embed_text_and_summaries(df, model)
    df = compute_dot_products(df, text_embeddings, summary_embeddings)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.summaries.stem}.csv"
    df.to_csv(path, index=False)

if __name__ == '__main__':
    main()
