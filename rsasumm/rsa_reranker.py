""" Main RSA Reranker script that is responsible for reranking the summaries using the RSA model.
Iterates over the source texts and candidates to compute the likelihood matrix.
The RSA model is then used to rerank the candidates and compute the speaker and listener probabilities.
The RSA model is defined in the RSAReranking class.
"""

from functools import cache
from typing import List

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm


class RSAReranking:
    """
    Rerank a list of candidates according to the RSA model.
    """

    def __init__(
            self,
            model,
            tokenizer,
            candidates: List[str],
            source_texts: List[str],
            batch_size: int = 32,
            rationality: int = 1,
            device="cuda",
    ):
        """
        :param model: hf model used to compute the likelihoods (supposed to be a seq2seq model), is S0 in the RSA model
        :param tokenizer:
        :param candidates: list of candidates summaries
        :param source_texts: list of source texts
        :param batch_size: batch size used to compute the likelihoods (can be high since we don't need gradients and
        it's a single forward pass)
        :param rationality: rationality parameter of the RSA model
        :param device: device used to compute the likelihoods
        """
        self.model = model
        self.device = device
        self.tokenizer = tokenizer

        self.candidates = candidates
        self.source_texts = source_texts

        self.batch_size = batch_size
        self.rationality = rationality
        self.likelihood_matrixPreComp = None
        if self.model is not None:
            self.model.to(self.device)

    def compute_conditionned_likelihood(
            self, x: List[str], y: List[str], mean: bool = True
    ) -> torch.Tensor:
        """
        Compute the likelihood of y given x

        :param x: list of source texts len(x) = batch_size
        :param y: list of candidates summaries len(y) = batch_size
        :param mean: average the likelihoods over the tokens of y or take the sum
        :return: tensor of shape (batch_size) containing the likelihoods of y given x
        """
        # Dummy inputs
        # source_texts = ["The paper is interesting."] -> 7 tokens
        # candidate_summaries = ["Well-written summary."] -> 7 tokens not necessary to have the same number of tokens
        assert len(x) == len(y)

        # Define the loss function (cross-entropy for token-level predictions)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        # Tokenize the source texts (x) and summaries (y)
        x = self.tokenizer(x, return_tensors="pt", padding=True,
                           truncation=True).to(self.device)
        y = self.tokenizer(y, return_tensors="pt", padding=True,
                           truncation=True).to(self.device)

        # Extract token IDs for input and output
        x_ids = x.input_ids.to(self.device)
        y_ids = y.input_ids.to(self.device)
        x_attention_mask = x.attention_mask.to(self.device)
        y_attention_mask = y.attention_mask.to(self.device)
        # print(x_ids.shape, y_ids.shape) -> (1, 7) (1, 7)
        # print(x_ids) -> tensor([[0,133,2225,16,2679,4,2]])

        # Pass the inputs through the model
        logits = self.model(
            input_ids=x_ids,
            decoder_input_ids=y_ids,
            attention_mask=x_attention_mask,
            decoder_attention_mask=y_attention_mask,
        ).logits

        # print(logits.shape) -> (1, 7, 50265)

        # Shift logits and token IDs for loss computation
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_ids = y_ids[..., 1:].contiguous()

        # print(shifted_logits.shape, shifted_ids.shape)
        # Result: (1, 6, 50265) (1, 6)

        # Compute token-level negative log-likelihood
        # shifted logits has a size (batch_size, sequence_length, vocab_size)
        # WE FLATTEN IT TO (batch_size x sequence_length, vocab_size)
        likelihood = -loss_fn(
            shifted_logits.view(-1, shifted_logits.size(-1)  # (1x6, 50265)
                                ), shifted_ids.view(-1)  # (1x6,)
        )

        # print(likelihood.shape) -> [6] == (6,)

        # Reshape the likelihood to match the batch
        # Reshape back to (batch_size, sequence_length) then sum(-1) -> (batch_size,)
        likelihood = likelihood.view(len(x["input_ids"]), -1).sum(-1)

        # print(likelihood.shape) -> [1] == (1,)

        # Normalize likelihood by the number of tokens if `mean=True`
        if mean:
            likelihood /= (y_ids !=
                           self.tokenizer.pad_token_id).float().sum(-1)

        # print(likelihood) = tensor([-6.6653])
        return likelihood

    def score(self, x: List[str], y: List[str], **kwargs):
        return self.compute_conditionned_likelihood(x, y, **kwargs)

    def likelihood_matrix(self) -> torch.Tensor:
        """
        Compute a likelihood matrix where entry (i, j) is the likelihood of
        candidate j summarizing source text i.

        Returns:
            torch.Tensor: Likelihood matrix of shape (len(source_texts), len(candidates)).
        """

        # initialize the likelihood matrix of size (len(source_texts), len(candidates))
        likelihood_matrix = torch.zeros(
            (len(self.source_texts), len(self.candidates))
        ).to(self.device)

        # create a list of pairs (i: index source, j: index candidate, source_text, candidate)
        pairs = []
        for i, source_text in enumerate(self.source_texts):
            for j, candidate in enumerate(self.candidates):
                pairs.append((i, j, source_text, candidate))

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
            # update the likelihood matrix with the likelihoods
            for k, (i, j, _, _) in enumerate(batch):
                likelihood_matrix[i, j] = likelihoods[k].detach()

        # return the likelihood matrix
        return likelihood_matrix

    @cache
    def S(self, t):
        if t == 0:
            return self.initial_speaker_probas
        else:
            listener = self.L(t - 1)
            # + self.initial_speaker_probas.sum(0, keepdim=True)
            prod = listener * self.rationality
            # Higher rationality focuses on selecting the most relevant candidates.
            return torch.log_softmax(prod, dim=-1)

    @cache
    def L(self, t):
        speaker = self.S(t)
        return torch.log_softmax(speaker, dim=-2)

    def mk_listener_dataframe(self, t):
        self.initial_speaker_probas = torch.tensor(
            self.likelihood_matrixPreComp.values)

        initial_listener_probas = self.L(0)

        # compute consensus
        uniform_distribution_over_source_texts = torch.ones_like(
            initial_listener_probas
        ) / len(self.source_texts)

        initital_consensuality_score = (
            torch.exp(initial_listener_probas)
            * (
                initial_listener_probas -
                torch.log(uniform_distribution_over_source_texts)
            )
        ).sum(0).cpu().numpy()

        initital_consensuality_score = pd.Series(
            initital_consensuality_score, index=self.candidates)

        initial_listener_probas = initial_listener_probas.cpu().numpy()

        initial_listener_probas = pd.DataFrame(initial_listener_probas)
        initial_listener_probas.index = self.source_texts
        initial_listener_probas.columns = self.candidates

        initial_speaker_probas = self.S(0).cpu().numpy()
        initial_speaker_probas = pd.DataFrame(initial_speaker_probas)
        initial_speaker_probas.index = self.source_texts
        initial_speaker_probas.columns = self.candidates

        listener_df = pd.DataFrame(self.L(t).cpu().numpy())

        consensuality_scores = (
            torch.exp(self.L(t))
            * (self.L(t) - torch.log(uniform_distribution_over_source_texts))
        ).sum(0).cpu().numpy()

        consensuality_scores = pd.Series(
            consensuality_scores, index=self.candidates)

        S = self.S(t).cpu().numpy()
        speaker_df = pd.DataFrame(S)

        # add the source texts and candidates as index

        listener_df.index = self.source_texts
        speaker_df.index = self.source_texts

        listener_df.columns = self.candidates
        speaker_df.columns = self.candidates

        return listener_df, speaker_df, initial_listener_probas, initial_speaker_probas, initital_consensuality_score, consensuality_scores

    def rerank(self, t=1, likelihoodMatrixPre=None):
        """
        Rerank candidates after t iterations of RSA.

        Args:
            t (int): Number of RSA iterations.

        Returns:
            Tuple: Best RSA summary, speaker/listener probabilities, and consensuality scores.
        """
        self.likelihood_matrixPreComp = likelihoodMatrixPre
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
