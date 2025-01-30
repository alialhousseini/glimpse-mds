"""
This script is used to generate summaries for the datasets under study.

The script contains a class SummaryGenerator that is used to generate summaries for the dataset(s) under study. The class has the following methods:
- evaluate_summarizer: This method is used to evaluate the summarizer model on the dataset provided. The method takes the dataset path, batch size, trimming, and checkpoint path as input and returns the dataset with summaries added.
- generate_abstractive_summary: This method is used to generate abstractive summaries for the dataset provided. The method takes the dataset path, batch size, and trimming as input and saves the summaries in a CSV file.
- generate_extractive_summary: This method is used to generate extractive summaries for the dataset provided. The method takes the dataset path as input and saves the summaries in a CSV file.
- change_model: This method is used to change the model used for generating summaries. The method takes the model name, model, and tokenizer as input and updates the model, tokenizer, and model name.

This script takes a folder (Path) and iterates over all the files:
- For each file, it generates an EXTRACTIVE summary and saves it in a corresponding directory.
- For each file, it generates a set of ABSTRACTIVE summaries according to the list of models provided.

Resulting data can be found in the 'data/candidates' directory.
"""


# Import libraries
from pathlib import Path
import os
import re
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import json
import torch
from tqdm import tqdm
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


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


if __name__ == '__main__':
    # Iterate over the dataset and generate (extractive) summaries first
    path_original_dataset = Path("/content/drive/MyDrive/original_data")

    model_for_extractive = {
        'model_1':
            {
                'model_id': "facebook/bart-large-cnn",
                'model_name': "BART"

            },
        'model_2':
            {
                'model_id': "google/pegasus-arxiv",
                'model_name': "PEGASUS-Arxiv"
            },
        'model_3':
            {
                'model_id': "google/pegasus-large",
                'model_name': "PEGASUS-Large"
            },
    }

    # Load the model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_for_extractive["model_1"]['model_id'])
    tokenizer = AutoTokenizer.from_pretrained(
        model_for_extractive["model_1"]['model_id'])
    model_name = model_for_extractive["model_1"]['model_name']

    # Initialize the SummaryGenerator
    sg = SummaryGenerator()
    for model_count, model_info in model_for_extractive.items():
        model_id = model_info['model_id']
        model_name = model_info['model_name']

        # Adjust SG params
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        sg.change_model(model_name, model, tokenizer)

        # Iterate over the datasets
        for file in path_original_dataset.glob('*.csv'):

            # Generate extractive summaries for this dataset
            sg.generate_abstractive_summary(file, 16, False)

    # Once all extractive summaries are generated, generate abstractive summaries
    for file in path_original_dataset.glob('*.csv'):
        sg.generate_extractive_summary(file)

    # By this we generated all abstractive and extractive summaries for all our datasets
    ########################################################################
    print("GENERATION OF ABS/EXT SUMMARIES DONE")
