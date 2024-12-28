from functools import cache
from typing import List
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

class VectorRSAReranking:
    def __init__(
            self,
            models,  # List of models
            tokenizers,  # List of tokenizers corresponding to the models
            model_types,  # List of model types: "encoder-decoder", "encoder-only", "long-context"
            candidates: List[str],
            source_texts: List[str],
            batch_size: int = 32,
            rationality: int = 1,
            device="cuda",
    ):
        assert len(models) == len(tokenizers) == len(model_types), (
            "Each model must have a corresponding tokenizer and model type."
        )
        self.models = models
        self.num_models = len(models)
        self.tokenizers = tokenizers
        self.model_types = model_types
        self.device = device

        self.candidates = candidates
        self.source_texts = source_texts

        self.batch_size = batch_size
        self.rationality = rationality

        # Move all models to the correct device
        for model in self.models:
            model.to(self.device)

    def compute_conditionned_likelihood(self, model, tokenizer, x, y, mean):
        """
        Encoder-decoder specific likelihood computation.
        """
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        batch_size = len(x)

        x = tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        y = tokenizer(y, return_tensors="pt", padding=True, truncation=True)

        x_ids = x.input_ids.to(self.device)
        y_ids = y.input_ids.to(self.device)

        logits = model(
            input_ids=x_ids,
            decoder_input_ids=y_ids,
            attention_mask=x.attention_mask.to(self.device),
            decoder_attention_mask=y.attention_mask.to(self.device),
        ).logits

        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_ids = y_ids[..., 1:].contiguous()

        likelihood = -loss_fn(
            shifted_logits.view(-1, shifted_logits.size(-1)), shifted_ids.view(-1)
        )
        likelihood = likelihood.view(batch_size, -1).sum(-1)

        if mean:
            likelihood /= (y_ids != tokenizer.pad_token_id).float().sum(-1)

        return likelihood

    def compute_conditionned_likelihoodEO(self, model, tokenizer, x, y, mean):
        """
        Encoder-only specific likelihood computation.
        """
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        batch_size = len(x)

        x = tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        y = tokenizer(y, return_tensors="pt", padding=True, truncation=True)

        x_ids = x.input_ids.to(self.device)
        y_ids = y.input_ids.to(self.device)

        logits = model(
            input_ids=x_ids,
            attention_mask=x.attention_mask.to(self.device),
        ).logits

        shifted_logits = logits[:, -y_ids.shape[1]:, :].contiguous()  # focusing on the summary part
        shifted_ids = y_ids.contiguous()

        likelihood = -loss_fn(
            shifted_logits.view(-1, shifted_logits.size(-1)), shifted_ids.view(-1)
        )
        likelihood = likelihood.view(batch_size, -1).sum(-1)

        if mean:
            likelihood /= (y_ids != tokenizer.pad_token_id).float().sum(-1)

        return likelihood

    def compute_conditionned_likelihoodLC(self, model, tokenizer, x, y, mean):
        """
        Long-context specific likelihood computation.
        """
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        batch_size = len(x)

        # Tokenize both source texts and summaries
        x = tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        y = tokenizer(y, return_tensors="pt", padding=True, truncation=True)

        x_ids = x.input_ids.to(self.device)
        y_ids = y.input_ids.to(self.device)

        # Concatenate the source and target tokens (for long-context models)
        input_ids = torch.cat([x_ids, y_ids], dim=1)
        attention_mask = torch.cat([x.attention_mask, y.attention_mask], dim=1)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Focus on the summary part of the logits
        summary_length = y_ids.shape[1]
        shifted_logits = logits[:, -summary_length:, :].contiguous()
        shifted_ids = y_ids.contiguous()

        # Compute likelihood for the summary
        likelihood = -loss_fn(
            shifted_logits.view(-1, shifted_logits.size(-1)), shifted_ids.view(-1)
        )

        likelihood = likelihood.view(batch_size, -1).sum(-1)

        if mean:
            likelihood /= (y_ids != tokenizer.pad_token_id).float().sum(-1)

        return likelihood

    def compute_conditionned_likelihood_t5(self, model, tokenizer, x, y, mean):
        """
        T5 Encoder-decoder specific likelihood computation for summarization tasks.
        This computes the conditional likelihood of summaries (y) given source texts (x).
        """
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        batch_size = len(x)

        # Add task-specific prefix (e.g., "summarize: ") to the source text
        x = ["summarize: " + text for text in x]
        
        # Tokenize the source and candidate summary texts
        x = tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        y = tokenizer(y, return_tensors="pt", padding=True, truncation=True)

        x_ids = x.input_ids.to(self.device)
        y_ids = y.input_ids.to(self.device)

        # Get logits from T5 model
        logits = model(
            input_ids=x_ids,
            attention_mask=x.attention_mask.to(self.device),
            decoder_input_ids=y_ids,
            decoder_attention_mask=y.attention_mask.to(self.device),
        ).logits

        # Shift logits to match the labels (ignores padding token)
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_ids = y_ids[..., 1:].contiguous()

        # Compute the likelihood (cross-entropy loss)
        likelihood = -loss_fn(
            shifted_logits.view(-1, shifted_logits.size(-1)), shifted_ids.view(-1)
        )
        likelihood = likelihood.view(batch_size, -1).sum(-1)

        # Optionally compute the mean likelihood over non-padding tokens
        if mean:
            likelihood /= (y_ids != tokenizer.pad_token_id).float().sum(-1)

        return likelihood

    def compute_likelihood(self, model, tokenizer, model_type, x, y, mean):
        """
        Dispatch to the correct likelihood computation function based on the model type.
        """
        if model_type == "encoder-decoder":
            return self.compute_conditionned_likelihood(model, tokenizer, x, y, mean)
        elif model_type == "encoder-only":
            return self.compute_conditionned_likelihoodEO(model, tokenizer, x, y, mean)
        elif model_type == "long-context":
            return self.compute_conditionned_likelihoodLC(model, tokenizer, x, y, mean)
        elif model_type == "t5":
            return self.compute_conditionned_likelihood_t5(model, tokenizer, x, y, mean)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def score(self, x: List[str], y: List[str], mean: bool = True):
        """
        Compute the average likelihoods from all models.
        """
        all_likelihoods = []

        for model, tokenizer, model_type in zip(self.models, self.tokenizers, self.model_types):
            likelihoods = self.compute_likelihood(
                model=model,
                tokenizer=tokenizer,
                model_type=model_type,
                x=x,
                y=y,
                mean=mean
            )
            all_likelihoods.append(likelihoods)

        # Compute the average likelihoods across all models
        average_likelihoods = torch.stack(all_likelihoods).mean(0)
        return average_likelihoods

    def likelihood_matrix(self) -> torch.Tensor:
        """
        :return: likelihood matrix : (world_size, num_candidates), likelihood[i, j] is the likelihood of
        candidate j being a summary for source text i.
        """
        likelihood_matrix = torch.zeros(
            (len(self.source_texts), len(self.candidates))
        ).to(self.device)

        pairs = []
        for i, source_text in enumerate(self.source_texts):
            for j, candidate in enumerate(self.candidates):
                pairs.append((i, j, source_text, candidate))

        # split the pairs into batches
        batches = [
            pairs[i: i + self.batch_size]
            for i in range(0, len(pairs), self.batch_size)
        ]

        for batch in tqdm(batches):
            # get the source texts and candidates
            source_texts = [pair[2] for pair in batch]
            candidates = [pair[3] for pair in batch]

            # compute the likelihoods
            with torch.no_grad():
                likelihoods = self.score(
                    source_texts, candidates, mean=True
                )

            # fill the matrix
            for k, (i, j, _, _) in enumerate(batch):
                likelihood_matrix[i, j] = likelihoods[k].detach()

        return likelihood_matrix

    @cache
    def S(self, t):
        if t == 0:
            return self.initial_speaker_probas
        else:
            listener = self.L(t - 1)
            # Process each element in the lists separately
            result = []
            for i in range(self.num_models):
                # Extract i-th elements from each list in the matrix
                current_layer = listener[..., i]
                # Perform operation on the current layer
                prod = current_layer * self.rationality
                processed_layer = torch.log_softmax(prod, dim=-1)
                result.append(processed_layer)
            
            # Stack the processed layers into a single tensor
            # Shape will be [batch_size, num_classes, list_length]
            return torch.stack(result, dim=-1)

    @cache
    def L(self, t):
        speaker = self.S(t)
        result = []
        for i in range(self.num_models):
            # Extract i-th elements from each list in the matrix
            current_layer = speaker[..., i]
            processed_layer = torch.log_softmax(current_layer, dim=-2)
            result.append(processed_layer)
        
        return torch.stack(result, dim=-1)

    def mk_listener_dataframe(self, t):
        self.initial_speaker_probas = self.likelihood_matrix()

        # TODO: consider other aggreggation methods (e.g. max, weighted average, etc.)        
        initial_listener_probas = self.L(0).mean(dim=-1)

        # Compute and return `initial_listener_probas` and other necessary components
        initial_listener_probas = initial_listener_probas.cpu().numpy()
        initial_listener_probas = pd.DataFrame(initial_listener_probas)
        initial_listener_probas.index = self.source_texts
        initial_listener_probas.columns = self.candidates

        # TODO: consider other aggreggation methods (e.g. max, weighted average, etc.)        
        initial_speaker_probas = self.S(0).mean(dim=-1)
        initial_speaker_probas = initial_speaker_probas.numpy()
        initial_speaker_probas = pd.DataFrame(initial_speaker_probas)
        initial_speaker_probas.index = self.source_texts
        initial_speaker_probas.columns = self.candidates

        # compute consensus
        uniform_distribution_over_source_texts = torch.ones_like(
            initial_listener_probas
        ) / len(self.source_texts)

        listener_df = pd.DataFrame(self.L(t).cpu().numpy().tolist())

        consensuality_scores = (
            (
                torch.exp(self.L(t))
                * (self.L(t) - torch.log(uniform_distribution_over_source_texts))
            )
            .sum(0).cpu().numpy()
        )

        # Averaging the score over the models
        # TODO: consider other aggreggation methods (e.g. max, weighted average, etc.)
        consensuality_scores = consensuality_scores.mean(axis=1)
        consensuality_scores = pd.Series(consensuality_scores, index=self.candidates)

        # TODO: consider other aggreggation methods (e.g. max, weighted average, etc.)
        speaker_scores = self.S(t).mean(dim=-1)
        speaker_df = pd.DataFrame(speaker_scores.cpu().numpy().tolist())

        listener_df.index = self.source_texts
        speaker_df.index = self.source_texts

        listener_df.columns = self.candidates
        speaker_df.columns = self.candidates

        return listener_df, speaker_df, initial_listener_probas, initial_speaker_probas, None, consensuality_scores

    def rerank(self, t=1):
        """
        return the best summary (according to rsa) for each text
        """
        (
            listener_df,
            speaker_df,
            initial_listener_proba,
            initial_speaker_proba,
            initital_consensuality_score,
            consensuality_scores,
        ) = self.mk_listener_dataframe(t=t)
        best_rsa = speaker_df.idxmax(axis=1).values
        best_base = initial_listener_proba.idxmax(axis=1).values

        return (
            best_rsa,
            best_base,
            speaker_df,
            listener_df,
            initial_listener_proba,
            initial_speaker_proba,
            initital_consensuality_score,
            consensuality_scores,
        )