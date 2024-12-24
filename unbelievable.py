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
########################################################################


class RSAReranking2:
    """
    Rerank a list of candidates according to the RSA model.
    """

    def __init__(
            self,
            model,
            model_type: str,
            candidates: List[str],
            source_texts: List[str],
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

        self.candidates = candidates
        self.source_texts = source_texts

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

    def __init__(self, model_name: str, model, tokenizer, device="cuda"):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generation_config = {
            "top_p_sampling": {
                "max_new_tokens": 200,
                "do_sample": True,
                "top_p": 0.95,
                "temperature": 1.0,
                "num_return_sequences": 8,
                "num_beams": 1,
                "early_stopping": True,
                "min_length": 0,
            }
        }

    def evaluate_summarizer(self, dataset_path: Path, batch_size: int, trimming: bool
                            ) -> Dataset:
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

        # generate summaries
        summaries = []
        print("Generating summaries...")

        for batch in tqdm(dataloader):
            text = batch["text"]

            inputs = self.tokenizer(
                text,
                max_length=1024,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # move inputs to device
            inputs = {key: value.to(self.device)
                      for key, value in inputs.items()}

            # generate summaries
            outputs = self.model.generate(
                **inputs,
                **self.generation_config,
            )

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

        # if trimming the last batch, remove them from the dataset
        if trimming:
            dataset = dataset.select(range(len(summaries)))

        # add summaries to the huggingface dataset
        dataset = dataset.map(lambda example: {"summary": summaries.pop(0)})

        return dataset

    def generate_abstractive_summary(self, dataset_path: Path, batch_size: int, trimming: bool):
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        dataset = self.evaluate_summarizer(
            dataset_path, batch_size, trimming
        )

        df_dataset = dataset.to_pandas()
        df_dataset = df_dataset.explode('summary')
        df_dataset = df_dataset.reset_index()
        # add an idx with  the id of the summary for each example
        df_dataset['id_candidate'] = df_dataset.groupby(['index']).cumcount()
        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d-%H-%M-%S")
        output_path = f"data/candidates/{self.model_name}_{dataset_path.stem}_{date}_extr.csv"
        # create output dir if it doesn't exist
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        df_dataset.to_csv(output_path, index=False, encoding="utf-8")
        print('done')

    def generate_extractive_summary(self, dataset_path: Path, batch_size: int, trimming: bool):
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

            text = text.replace('-----', '\n')
            sentences = nltk.sent_tokenize(text)
            # remove empty sentences
            sentences = [sentence for sentence in sentences if sentence != ""]

            summaries.append(sentences)

        # add summaries to the huggingface dataset
        dataset = dataset.map(lambda example: {"summary": summaries.pop(0)})

        df_dataset = dataset.to_pandas()
        df_dataset = df_dataset.explode("summary")
        df_dataset = df_dataset.reset_index()
        # add an idx with  the id of the summary for each example
        df_dataset["id_candidate"] = df_dataset.groupby(["index"]).cumcount()

        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d-%H-%M-%S")
        output_path = f"data/candidates/{self.model_name}_{dataset_path.stem}_{date}_abstr.csv"
        # create output dir if it doesn't exist
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        df_dataset.to_csv(output_path, index=False, encoding="utf-8")
        print('done')
########################################################################
