from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer
import argparse
from tqdm import tqdm

from pickle import dump

import sys, os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rsasumm.rsa_reranker import RSAReranking


DESC = """
Compute the RSA matrices for all the set of multi-document samples and dump these along with additional information in a pickle file.
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/pegasus-arxiv")
    parser.add_argument("--summaries_folder", type=Path, default="")
    parser.add_argument("--output_dir", type=str, default="output")

    parser.add_argument("--filter", type=str, default=None)
    
    # if ran in a scripted way, the output path will be printed
    parser.add_argument("--scripted-run", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def parse_summaries(path: Path) -> pd.DataFrame:
    
    try:
        summaries = pd.read_csv(path)
    except:
        raise ValueError(f"Unknown dataset {path}")

    # check if the dataframe has the right columns
    if not all(
        col in summaries.columns for col in ["index", "id", "text", "gold", "summary", "id_candidate"]
    ):
        raise ValueError(
            "The dataframe must have columns ['index', 'id', 'text', 'gold', 'summary', 'id_candidate']"+str(path)
        )

    return summaries


def consensus_scores_based_summaries(sample, n_consensus=3, n_dissensus=3):
    consensus_samples = sample['consensuality_scores'].sort_values(ascending=True).head(n_consensus).index.tolist()
    disensus_samples = sample['consensuality_scores'].sort_values(ascending=False).head(n_dissensus).index.tolist()
    
    consensus = ".".join(consensus_samples)
    disensus = ".".join(disensus_samples)
    
    return consensus + "\n\n" + disensus
    
    
def rsa_scores_based_summaries(sample, n_consensus=3, n_rsa_speaker=3):
    consensus_samples = sample['consensuality_scores'].sort_values(ascending=True).head(n_consensus).index.tolist()
    rsa = sample['best_rsa'].tolist()[:n_rsa_speaker]
    
    consensus = ".".join(consensus_samples)
    rsa = ".".join(rsa)
    
    return consensus + "\n\n" + rsa

def compute_rsa(summaries: pd.DataFrame, model, tokenizer, device,modName,datasetName):
    results = []
    gliUn_data = []
    gliSp_data = []
    i=0
    for name, group in tqdm(summaries.groupby(["id"])):
        if(len(group)>70):
            i+=1
            continue
        rsa_reranker = RSAReranking(
            model,
            tokenizer,
            device=device,
            candidates=group.summary.unique().tolist(),
            source_texts=group.text.unique().tolist(),
            batch_size=32,
            rationality=3,
        )
        (
            best_rsa,
            best_base,
            speaker_df,
            listener_df,
            initial_listener,
            language_model_proba_df,
            initial_consensuality_scores,
            consensuality_scores,
        ) = rsa_reranker.rerank(t=2)

        gold = group['gold'].tolist()[0]

        results.append(
            {
                "id": name,
                "best_rsa": best_rsa,
                "best_base": best_base,
                "speaker_df": speaker_df,
                "listener_df": listener_df,
                "initial_listener": initial_listener,
                "language_model_proba_df": language_model_proba_df,
                "initial_consensuality_scores": initial_consensuality_scores,
                "consensuality_scores": consensuality_scores,
                "gold": gold,
                "rationality": 3,
                "text_candidates" : group
            }
        )
        unique_texts = group.text.unique().tolist()
        concatenated_string = " ".join(unique_texts) 
        gliUn = consensus_scores_based_summaries(results[-1])
        gliSp = rsa_scores_based_summaries(results[-1])

        gliUn_data.append({"Id": name, "summary": gliUn,"text":concatenated_string,"gold":gold})
        gliSp_data.append({"Id": name, "summary": gliSp,"text":concatenated_string,"gold":gold})

    gliUn_df = pd.DataFrame(gliUn_data)
    gliSp_df = pd.DataFrame(gliSp_data)    
    gliUn_df.to_csv(f"producedSum/{modName.split('/')[1]}_glimpseUnique_{datasetName.stem}.csv", index=False)
    gliSp_df.to_csv(f"producedSum/{modName.split('/')[1]}_glimpseSpeaker_{datasetName.stem}.csv", index=False)

    return results


def main():
    args = parse_args()
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if "pegasus" in args.model_name: 
        tokenizer = PegasusTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = model.to(args.device)

    for summary_file in os.listdir(args.summaries_folder):
        if summary_file.endswith(".csv"):
            summaries_path = Path(args.summaries_folder) / summary_file

            summaries = parse_summaries(summaries_path)

            results = compute_rsa(summaries, model, tokenizer, args.device, args.model_name, summaries_path)

            results = {"results": results}

            results["metadata/reranking_model"] = args.model_name
            results["metadata/rsa_iterations"] = 3

            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            output_path = Path(args.output_dir) / f"{summaries_path.stem}-_-r3-_-rsa_reranked-{args.model_name.replace('/', '-')}.pk"
            output_path_base = (
                Path(args.output_dir) / f"{summaries_path.stem}-_-base_reranked.pk"
            )

            with open(output_path, "wb") as f:
                dump(results, f)

            if args.scripted_run: 
                print(output_path)


if __name__ == "__main__":
    main()
