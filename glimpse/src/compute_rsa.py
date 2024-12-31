from rsasumm.rsa_reranker import RSAReranking
import pickle
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer
import argparse
from tqdm import tqdm

from pickle import dump

import sys
import os.path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


DESC = """
Compute the RSA matrices for all the set of multi-document samples and dump these along with additional information in a pickle file.
"""


def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--model_name", type=str, default="google/pegasus-arxiv")facebook/bart-large-cn
    parser.add_argument("--model_name", type=str,
                        default="facebook/bart-large-cn")
    parser.add_argument("--summaries_folder", type=Path, default="")

    parser.add_argument("--output_dir", type=str, default="output")

    parser.add_argument("--filter", type=str, default=None)

    # if ran in a scripted way, the output path will be printed
    parser.add_argument(
        "--scripted-run", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def parse_summaries(path: Path) -> pd.DataFrame:

    try:
        summaries = pd.read_csv(path)
    except:
        raise ValueError(f"Unknown dataset {path}")

    # check if the dataframe has the right columns
    if not all(
        col in summaries.columns for col in ["id", "text", "gold", "summary", "id_candidate"]
    ):
        raise ValueError(
            "The dataframe must have columns ['index', 'id', 'text', 'gold', 'summary', 'id_candidate']"+str(
                path)
        )

    return summaries


def consensus_scores_based_summaries(sample, n_consensus=3, n_dissensus=3):
    consensus_samples = sample['consensuality_scores'].sort_values(
        ascending=True).head(n_consensus).index.tolist()
    disensus_samples = sample['consensuality_scores'].sort_values(
        ascending=False).head(n_dissensus).index.tolist()

    consensus = ".".join(consensus_samples)
    disensus = ".".join(disensus_samples)

    return consensus + "\n\n" + disensus


def rsa_scores_based_summaries(sample, n_consensus=3, n_rsa_speaker=3):
    consensus_samples = sample['consensuality_scores'].sort_values(
        ascending=True).head(n_consensus).index.tolist()
    rsa = sample['best_rsa'].tolist()[:n_rsa_speaker]

    consensus = ".".join(consensus_samples)
    rsa = ".".join(rsa)

    return consensus + "\n\n" + rsa


def compute_rsa(summaries: pd.DataFrame, model, tokenizer, device, modName, datasetName):
    probas_dir = Path("data/lm_probas")

    # Create a log file to capture errors
    error_log_path = "error_log.txt"

    # Iterate over all the probas files
    for probas_file in probas_dir.glob("*.pkl"):
        # Initialize for each computation the following lists
        results = []
        gliUn_data = []
        gliSp_data = []

        # Check if the dataset name is in the probas file
        # e.g. PEGASUS-Arxiv_all_merged_226_-_abstr in PEGASUS-Arxiv_all_merged_226_-_abstr-_-BART.pkl
        if datasetName in probas_file.stem:
            # Load the probas file
            results_dict = pd.read_pickle(probas_file)

            # Our interest is only 'id' and 'language_model_proba_df' columns
            # We extract both as key-value pair
            results_by_id = {result['id'][0]: result['language_model_proba_df']
                             for result in results_dict['results']}

            # Iterate over each group of summaries
            for name, group in tqdm(summaries.groupby(["id"]), desc=f"Processing {probas_file}"):
                try:

                    try:
                        likelihoodPreComp = results_by_id.get(
                            group["id"].iloc[0])
                    except KeyError:
                        raise ValueError(
                            f"Could not find {group['id'].iloc[0]} in {probas_file}")

                    rsa_reranker = RSAReranking(
                        model,
                        tokenizer,
                        device=device,
                        candidates=group.summary.unique().tolist(),
                        source_texts=group.text.unique().tolist(),
                        batch_size=32,
                        rationality=1,
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
                    ) = rsa_reranker.rerank(t=3, likelihoodMatrixPre=likelihoodPreComp)

                    gold = group['gold'].tolist()[0]

                    # Collect results
                    results.append(
                        {
                            "id": name[0],
                            "best_rsa": best_rsa,
                            "best_base": best_base,
                            "speaker_df": speaker_df,
                            "listener_df": listener_df,
                            "initial_listener": initial_listener,
                            "language_model_proba_df": language_model_proba_df,
                            "initial_consensuality_scores": initial_consensuality_scores,
                            "consensuality_scores": consensuality_scores,
                            "gold": gold,
                            "rationality": 1,
                            "text_candidates": group
                        }
                    )

                    unique_texts = group.text.unique().tolist()
                    unique_texts = [text.replace("\n", " ")
                                    for text in unique_texts]
                    concatenated_string = " ".join(unique_texts)

                    gliUn = consensus_scores_based_summaries(results[-1])
                    gliSp = rsa_scores_based_summaries(results[-1])

                    gliUn_data.append(
                        {"id": name[0], "summary": gliUn, "text": concatenated_string, "gold": gold})
                    gliSp_data.append(
                        {"id": name[0], "summary": gliSp, "text": concatenated_string, "gold": gold})

                except Exception as e:
                    with open(error_log_path, 'a') as error_log:
                        error_log.write(
                            f"Error processing {name} in file {probas_file}: {str(e)}\n")
                    print(
                        f"Error processing {name} in file {probas_file}: {str(e)}")
                    continue

            gliUn_df = pd.DataFrame(gliUn_data)
            gliSp_df = pd.DataFrame(gliSp_data)

            gliUn_dir = "allSummariesGUnique"
            gliSp_dir = "allSummariesGSpeaker"

            Path.mkdir(Path(gliUn_dir), exist_ok=True)
            Path.mkdir(Path(gliSp_dir), exist_ok=True)
            gliUn_df.to_csv(
                f"{Path(gliUn_dir) / probas_file.stem}_glimpseUnique.csv", index=False)
            gliSp_df.to_csv(
                f"{Path(gliSp_dir) / probas_file.stem}_glimpseSpeaker.csv", index=False)

    return results


def main():
    args = parse_args()
    """model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    model = model.to(args.device)


    for summary_file in os.listdir(args.summaries_folder):
        if summary_file.endswith(".csv"):
            summaries_path = Path(args.summaries_folder) / summary_file


            summaries = parse_summaries(summaries_path)

            results = compute_rsa(summaries, model, tokenizer, args.device, "facebook/bart-large-cnn", summaries_path)

            results = {"results": results}

            results["metadata/reranking_model"] = "facebook/bart-large-cnn"
            results["metadata/rsa_iterations"] = 3

            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            output_path = Path(args.output_dir) / f"{summaries_path.stem}-_-r3-_-rsa_reranked-facebook-bart-large-cnn.pk"
            output_path_base = Path(args.output_dir) / f"{summaries_path.stem}-_-base_reranked.pk"

            with open(output_path, "wb") as f:
                dump(results, f)

            if args.scripted_run: 
                print(output_path)"""
    model = None
    tokenizer = None
    for summary_file in os.listdir(args.summaries_folder):
        if summary_file.endswith(".csv"):
            summaries_path = Path(args.summaries_folder) / summary_file

            summaries = parse_summaries(summaries_path)

            results = compute_rsa(
                summaries, model, tokenizer, args.device, None, summary_file.split('.')[0])


if __name__ == "__main__":
    main()
