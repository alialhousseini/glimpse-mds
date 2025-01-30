"""
It is recommended to read this script from the main notebook as data was generated in the notebook and then saved in the pkl files.

This script is used to generate the pkl files that contain the language model probabilities for each model and each dataset.

It also iterates over the set of folder provided and do the process automatically.

"""
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import datetime
import torch
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import List, Dict
import nltk
from rsasumm.rsa_reranker import RSAReranking
import pickle
import re
from functools import reduce
import operator
from torch.cuda.amp import autocast
from transformers import set_seed
import random
import numpy as np
import json
import os
########################################################################


def set_random_seed(seed: int):
    random.seed(seed)  # For Python's random
    np.random.seed(seed)  # For NumPy
    torch.manual_seed(seed)  # For PyTorch on CPU
    torch.cuda.manual_seed(seed)  # For PyTorch on GPU


set_random_seed(42)
########################################################################


class RSAReranking2:
    """
    Rerank a list of candidates according to the RSA model.
    """

    def __init__(
            self,
            model,
            model_type: str,
            batch_size: int = 32,
            rationality: int = 1,
            device="cuda",
            tokenizer=None,
    ):
        """
        :param candidates: list of candidates summaries
        :param source_texts: list of source texts
        :param batch_size: batch size used to compute the likelihoods (can be high since we don't need gradients and
        it's a single forward pass)
        :param rationality: rationality parameter of the RSA model
        :param device: device used to compute the likelihoods
        """
        self.device = device

        self.batch_size = batch_size
        self.rationality = rationality

        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        assert self.model_type in ["ENC-DEC", "ENC",
                                   "Others", "LongContext"], "Invalid model type."

        # Generalized dictionary for model-to-function mapping
        self.model_handlers: Dict[str, Dict] = {
            "ENC-DEC": {
                # BART: Different version
                # PEGASUS: Different version (XSUM)
                # T5: Different version (C4)

                "models": ["BART", "PEGASUS", "T5", "mBART", "Flan-T5"],
                "function": "compute_likelihood_enc_dec",
            },
            "ENC": {
                "models": ["BERTSUM", "SciBERT"],
                "function": "compute_similarity_enc",
            },
            "Others": {
                "models": ["ALL-MPNET", "XLM-RoBERTa"],
                "function": "compute_custom_others",
            },
            "LongContext": {
                "models": ["LED", "BigBird-Pegasus"],
                "function": "compute_custom_long_context",
            },
        }

    def score(self, model_name: str, x: List[str], y: List[str], **kwargs):
        """
        Compute the likelihood of a summary given a source text.
        """
        for model_type in self.model_handlers.keys():
            if model_name in self.model_handlers[model_type]["models"]:
                function_name = self.model_handlers[model_name]["function"]
                function = getattr(self, function_name)

        # Compute the likelihood
        return function(x, y, **kwargs)

    def compute_likelihood_enc_dec(self, x: List[str], y: List[str], mean: bool = True) -> torch.Tensor:
        """
        Compute likelihood for Encoder-Decoder models (e.g., BART, T5).
        """
        assert len(x) == len(
            y), "Source and summary lists must have the same length."

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        # Tokenize source and summary
        # Tokenize the source texts (x) and summaries (y)
        x_enc = self.tokenizer(x, return_tensors="pt", padding=True,
                               truncation=True).to(self.device)
        y_enc = self.tokenizer(y, return_tensors="pt", padding=True,
                               truncation=True).to(self.device)
        # Move to the correct device
        x_enc = {k: v.to(self.device) for k, v in x_enc.items()}
        y_enc = {k: v.to(self.device) for k, v in y_enc.items()}

        # Compute logits
        logits = self.model(
            input_ids=x_enc["input_ids"],
            decoder_input_ids=y_enc["input_ids"],
            attention_mask=x_enc["attention_mask"],
            decoder_attention_mask=y_enc["attention_mask"],
        ).logits

        # Compute token-level likelihood
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = y_enc["input_ids"][..., 1:].contiguous()

        likelihood = -loss_fn(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
        )

        likelihood = likelihood.view(len(x['input_ids']), -1).sum(dim=-1)
        if mean:
            likelihood /= (y_enc["input_ids"] !=
                           self.tokenizer.pad_token_id).float().sum(dim=-1)
        return likelihood

    def compute_similarity_enc(self, x: List[str], y: List[str]) -> torch.Tensor:
        """
        Compute semantic similarity for Encoder-only models (e.g., BERTSUM, SciBERT).
        """
        assert len(x) == len(
            y), "Source and summary lists must have the same length."

        # Tokenize and encode
        x_enc = self.tokenizer(x, return_tensors="pt",
                               padding=True, truncation=True)
        y_enc = self.tokenizer(y, return_tensors="pt",
                               padding=True, truncation=True)

        x_enc = {k: v.to(self.device) for k, v in x_enc.items()}
        y_enc = {k: v.to(self.device) for k, v in y_enc.items()}

        # Generate embeddings
        x_embeddings = self.model(**x_enc).last_hidden_state.mean(dim=1)
        y_embeddings = self.model(**y_enc).last_hidden_state.mean(dim=1)

        # Compute cosine similarity
        return torch.nn.functional.cosine_similarity(x_embeddings, y_embeddings)

    def compute_custom_long_context(self, x: List[str], y: List[str]) -> torch.Tensor:
        """
        Handle long-context models like LongFormer or BigBird.
        """
        assert len(x) == len(
            y), "Source and summary lists must have the same length."

        # Concatenate source and summary for joint tokenization
        inputs = [f"{src} {summ}" for src, summ in zip(x, y)]
        encodings = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True)
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        # Compute logits
        logits = self.model(**encodings).logits

        # Convert logits to probabilities (assume binary classification task)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        # Adjust based on the model's task definition
        return probabilities[:, 1]

    def compute_custom_others(self, x: List[str], y: List[str]) -> torch.Tensor:
        """
        Handle special cases like ALL-MPNET or XLM-RoBERTa.
        """
        assert len(x) == len(
            y), "Source and summary lists must have the same length."

        # Generate embeddings
        x_embeddings = self.model.encode(x, convert_to_tensor=True)
        y_embeddings = self.model.encode(y, convert_to_tensor=True)

        # Compute cosine similarity
        return torch.nn.functional.cosine_similarity(x_embeddings, y_embeddings)


########################################################################

class SummaryGenerator:

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    def __init__(self, model_name: str = 'BART', model=model, tokenizer=tokenizer, device="cuda"):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generation_config = {
            "max_new_tokens": 200,
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 1.0,
            "num_return_sequences": 8,
            "num_beams": 1,
            "early_stopping": True,
            "min_length": 0,
        }

    def evaluate_summarizer(self, dataset_path: Path, batch_size: int, trimming: bool, checkpoint_path: str) -> Dataset:
        """
        @param model: The model used to generate the summaries
        @param tokenizer: The tokenizer used to tokenize the text and the summary
        @param dataset: A dataset with the text
        @param decoding_config: Dictionary with the decoding config
        @param batch_size: The batch size used to generate the summaries
        @return: The same dataset with the summaries added
        """
        try:
            dataset = pd.read_csv(dataset_path)
        except:
            raise ValueError(f"Unknown dataset {dataset_path}")

        # make a dataset from the dataframe
        dataset = Dataset.from_pandas(dataset)

        # create a dataloader
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=trimming)

        # Checkpoint file to save progress
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
        else:
            checkpoint = {"processed_batches": 0, "summaries": []}

        summaries = checkpoint["summaries"]
        print("Generating summaries...")

        self.model.to(self.device)

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # Skip already processed batches
            if batch_idx < checkpoint["processed_batches"]:
                continue

            text = batch["text"]

            inputs = self.tokenizer(
                text,
                max_length=min(self.tokenizer.model_max_length,
                               768),  # Adjust max_length
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # move inputs to device
            inputs = {key: value.to(self.device)
                      for key, value in inputs.items()}

            # generate summaries
            try:
                with torch.amp.autocast('cuda'):
                    outputs = self.model.generate(
                        **inputs, **self.generation_config)
            except RuntimeError as e:
                print(f"Error during generation: {e}")
                print(f"Input shape: {inputs['input_ids'].shape}")
                # Save progress before raising an error
                checkpoint["processed_batches"] = batch_idx
                checkpoint["summaries"] = summaries
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint, f)
                raise

            total_size = outputs.numel()  # Total number of elements in the tensor
            # Target size of the last dimension
            target_size = batch_size * outputs.shape[-1]
            # Calculate the required padding size to make the total number of elements divisible by the target size
            pad_size = (target_size - (total_size % target_size)) % target_size

            # Pad the tensor with zeros to make the total number of elements divisible by the target size
            if not trimming and pad_size != 0:
                outputs = torch.nn.functional.pad(
                    outputs, (0, 0, 0, pad_size // outputs.shape[-1]))

            # output : (batch_size * num_return_sequences, max_length)
            try:
                outputs = outputs.reshape(batch_size, -1, outputs.shape[-1])
            except Exception as e:
                print(f"Error reshaping outputs: {e}")
                raise ValueError(f"Cannot reshape tensor of size {outputs.numel()} into shape "
                                 f"({batch_size}, -1, {outputs.shape[-1]}).")

            # decode summaries
            for b in range(batch_size):
                summaries.append(
                    [
                        self.tokenizer.decode(
                            outputs[b, i],
                            skip_special_tokens=True,
                        )
                        for i in range(outputs.shape[1])
                    ]
                )

            # Save progress after processing each batch
            checkpoint["processed_batches"] = batch_idx + 1
            checkpoint["summaries"] = summaries
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f)

            # if trimming the last batch, remove them from the dataset
            if trimming:
                dataset = dataset.select(range(len(summaries)))

        # add summaries to the huggingface dataset
        dataset = dataset.map(lambda example: {"summary": summaries.pop(0)})

        # Clean up the checkpoint file after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        return dataset

    def generate_abstractive_summary(self, dataset_path: Path, batch_size: int, trimming: bool):
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        dataset = self.evaluate_summarizer(
            dataset_path, batch_size, trimming, checkpoint_path="summarizer_checkpoint.json"
        )

        df_dataset = dataset.to_pandas()
        df_dataset = df_dataset.explode('summary')
        df_dataset = df_dataset.reset_index()
        # add an idx with  the id of the summary for each example
        df_dataset['id_candidate'] = df_dataset.groupby(['index']).cumcount()

        output_path = Path(
            f"data/candidates/{self.model_name}_{dataset_path.stem}_-_abstr.csv")
        # create output dir if it doesn't exist
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        df_dataset.to_csv(output_path, index=False, encoding="utf-8")
        print('done')

    def generate_extractive_summary(self, dataset_path: Path):
        try:
            dataset = pd.read_csv(dataset_path)
        except:
            raise ValueError(f"Unknown dataset {dataset_path}")

        # make a dataset from the dataframe
        dataset = Dataset.from_pandas(dataset)

        summaries = []

        # (tqdm library for progress bar)
        for sample in tqdm(dataset):
            text = sample["text"]

            # Replace any set of successive dashes (e.g., --, ----, -----) with a newline
            text = re.sub(r'-{2,}', '\n', text)

            # Remove patterns like ".2-" or isolated numerics with hyphens
            text = re.sub(r'\.\d+-', '', text)

            # Replace multiple newlines or spaces with a single newline or space
            # Replace multiple newlines with one
            text = re.sub(r'\n+', '\n', text)
            # Replace multiple spaces with one
            text = re.sub(r'\s+', ' ', text)

            # Remove any remaining unwanted characters (e.g., control characters)
            # Remove non-ASCII characters
            text = re.sub(r'[^\x00-\x7F]+', '', text)

            # To be discussed
            text = text.replace("\n", " ")

            sentences = nltk.sent_tokenize(text)

            # remove empty sentences
            sentences = [sentence for sentence in sentences if sentence != ""]

            # Filter out short or meaningless sentences
            sentences = [sent for sent in sentences if len(sent) > 8]

            summaries.append(sentences)

        # add summaries to the huggingface dataset
        dataset = dataset.map(lambda example: {"summary": summaries.pop(0)})

        df_dataset = dataset.to_pandas()
        df_dataset = df_dataset.explode("summary")
        df_dataset = df_dataset.reset_index()
        # add an idx with  the id of the summary for each example
        df_dataset["id_candidate"] = df_dataset.groupby(["index"]).cumcount()

        output_path = f"data/candidates/{dataset_path.stem}_-_extr.csv"
        output_path = Path(output_path)
        # create output dir if it doesn't exist
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        df_dataset.to_csv(output_path, index=False, encoding="utf-8")
        print('done')

    def change_model(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
########################################################################


if __name__ == "__main__":
    # # Iterate over the dataset and generate (extractive) summaries first
    # path_original_dataset = Path("data/data_to_process")

    # model_for_extractive = {
    #     # 'model_1':
    #     #     {
    #     #         'model_id': "facebook/bart-large-cnn",
    #     #         'model_name': "BART"

    #     #     },
    #     'model_1':
    #         {
    #             'model_id': "google/pegasus-arxiv",
    #             'model_name': "PEGASUS"
    #         },
    # }

    # # Load the model and tokenizer
    # model = AutoModelForSeq2SeqLM.from_pretrained(
    #     model_for_extractive["model_1"]['model_id'])
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_for_extractive["model_1"]['model_id'])
    # model_name = model_for_extractive["model_1"]['model_name']

    # # Initialize the SummaryGenerator
    # sg = SummaryGenerator()
    # for model_count, model_info in model_for_extractive.items():
    #     model_id = model_info['model_id']
    #     model_name = model_info['model_name']

    #     # Adjust SG params
    #     model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    #     tokenizer = AutoTokenizer.from_pretrained(model_id)
    #     sg.change_model(model_name, model, tokenizer)

    #     # Iterate over the datasets
    #     for file in path_original_dataset.glob('*.csv'):

    #         # Generate extractive summaries for this dataset
    #         sg.generate_abstractive_summary(file, 16, False)

    # # Once all extractive summaries are generated, generate abstractive summaries
    # for file in path_original_dataset.glob('*.csv'):
    #     sg.generate_extractive_summary(file)

    # # By this we generated all abstractive and extractive summaries for all our datasets
    # ########################################################################
    # print("GENERATION OF ABS/EXT SUMMARIES DONE")

    # Now we can generate for each dataset LM_probas
    # Iterate over the generated summaries and compute the likelihoods
    # Save data in appropriate format

    model_for_likelihood_computation = {
        'model_1':
            {
                'model_id': "facebook/bart-large-cnn",
                'model_name': "BART"
            },
        'model_2':
            {
                'model_id': "google/pegasus-xsum",
                'model_name': "PEGASUS"
            },
    }

    path_candidates = Path("data/candidates")

    for model_count, model_info in model_for_likelihood_computation.items():
        # Load the model and tokenizer
        model_id = model_info['model_id']
        model_name = model_info['model_name']
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        for file in path_candidates.glob('*.csv'):
            # For each dataset we want to do the following
            # Extract LM_probas for each dataframe
            # Save them in a pkl file for each 'paper' (id)
            # Save the pkl files in a folder named after the dataset and the model used

            results = []
            curr_ds = pd.read_csv(file)

            # Name is a tuple e.g. ('id_name',)
            # group is a GroupedByKey DataFrame

            for name, group in tqdm(curr_ds.groupby(["id"])):

                rsa_reranker = RSAReranking(
                    model,  # model on which we want to compute the RSA
                    tokenizer,  # tokenizer for the model
                    device='cuda',
                    candidates=group.summary.unique().tolist(),
                    source_texts=group.text.unique().tolist(),
                    batch_size=32,
                    rationality=1,
                )
                # print(len(group.summary.unique().tolist()))
                # print(len(group.text.unique().tolist()))
                lm_probas = rsa_reranker.likelihood_matrix()
                # print(lm_probas.shape)
                lm_probas = lm_probas.cpu().numpy()
                lm_probas_df = pd.DataFrame(lm_probas)
                lm_probas_df.index = group.text.unique().tolist()
                lm_probas_df.columns = group.summary.unique().tolist()
                gold = group['gold'].tolist()[0]

                results.append(
                    {
                        "id": name,
                        "language_model_proba_df": lm_probas_df,
                        "gold": gold,
                        "rationality": 1,  # hyperparameter
                        "text_candidates": group
                    }
                )

            # Save the results
            opt_dir = Path(f'data/lm_probas/')
            if not opt_dir.exists():
                opt_dir.mkdir(parents=True, exist_ok=True)

            opt_path = Path(f"data/lm_probas/{file.stem}-_-{model_name}.pkl")
            results = {"results": results}
            with open(opt_path, 'wb') as f:
                pickle.dump(results, f)

    ########################################################################

    def elementwise_max(dfs):
        """
        dfs: list of DataFrames (same index/columns)
        """
        return reduce(lambda x, y: x.combine(y, func=max), dfs)

    # Now we can write a script that takes the set of LM_probas for each dataset and (set) of models
    # and aggregate them to get the final ranking

    # We define a set of model names, this set represents the set of models we want to aggregate their results
    # In addition we define a methodology of aggregation(e.g. mean, max, weighted_avg, etc.)

    model_names = ["BART", "PEGASUS"]

    # We need to find for each set of common datasets, the models we are looking for:
    lm_probas_path = Path("data/lm_probas")
    lm_probas_files = list(lm_probas_path.glob("*.pkl"))
    # Filter out the files that do not contain the models we are looking for
    # So we keep only the files that contain the models we are looking for
    lm_probas_files = [file for file in lm_probas_files if any(
        model_name in file.stem.split('-_-')[-1] for model_name in model_names)]

    # Now for each file, we collect filenames together to be processed
    files_and_pickles = {}
    for file in lm_probas_files:
        filename = file.stem.split('-_-')[0]
        if filename not in files_and_pickles:
            files_and_pickles[filename] = [file]
        else:
            files_and_pickles[filename].append(file)

    method = "mean"

    # Now we can aggregate the results
    # We will aggregate the results for each dataset
    for filename, files in files_and_pickles.items():
        # We iterate over the dict
        # filename is the name of the dataset
        # files is a list of paths to the pkl files

        # Load the results for each model
        pkls = [pd.read_pickle(f) for f in files]
        # Go to results
        pkls = [f['results'] for f in pkls]
        # Now pkls is a list of lists of dictionaries [ [{},{},{}], [{},{},{}], ...]
        # We want to access the language_model_proba_df for each dictionary in parallel
        # i.e. [ [{a1},{b1},{c1}], [{a2},{b2},{c2}], ...] -> [ {a_i}, {b_i}, {c_i} ]

        # Results
        results = []
        for i in range(len(pkls[0])):  # iterate over the dictionaries
            # index 'i' is shared
            set_of_dicts = [pkls[j][i] for j in range(len(pkls))]
            # set_of_dicts is a list of dictionaries that share the same index
            # [{a1}, {a2}, {a3}, ...]
            # Now we want to aggregate the language_model_proba_df for each dictionary
            new_dict = {}
            new_dict['id'] = set_of_dicts[0]['id']
            new_dict['gold'] = set_of_dicts[0]['gold']
            new_dict['rationality'] = set_of_dicts[0]['rationality']
            new_dict['text_candidates'] = set_of_dicts[0]['text_candidates']
            # Now we want to aggregate the language_model_proba_df
            # THIS HAS TO BE DONE ACCORDING TO A METHOD (max, weighted_avg, etc.)
            set_of_dfs = [d['language_model_proba_df'] for d in set_of_dicts]

            # Additional check of consistency
            ref_index = set_of_dfs[0].index
            ref_columns = set_of_dfs[0].columns

            for t, df in enumerate(set_of_dfs[1:], start=2):
                # Compare sets OR compare ordered lists
                if not df.index.equals(ref_index):
                    raise ValueError(
                        f"DataFrame #{i} index does not match the reference. "
                        f"Expected {list(ref_index)}, got {list(df.index)}."
                    )
                if not df.columns.equals(ref_columns):
                    raise ValueError(
                        f"DataFrame #{i} columns do not match the reference. "
                        f"Expected {list(ref_columns)}, got {list(df.columns)}."
                    )

            if method == "mean":

                # To aggregation safely
                df_sum = reduce(operator.add, set_of_dfs)
                df_agg = df_sum / len(set_of_dfs)

            if method == "max":
                df_agg = elementwise_max(set_of_dfs)

            # Save it!
            new_dict['language_model_proba_df'] = df_agg

            # Save model names used as well
            new_dict['model_names'] = model_names

            results.append(new_dict)

        results = {"results": results}
        # Save the results
        opt_dir = Path(f'data/agg_lms/')
        if not opt_dir.exists():
            opt_dir.mkdir(parents=True, exist_ok=True)
        opt_path = Path(f"data/agg_lms/{filename}.pkl")
        with open(opt_path, 'wb') as f:
            pickle.dump(results, f)
    ########################################################################
