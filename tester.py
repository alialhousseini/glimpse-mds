'''Script used to perform basic tests'''
from transformers import AutoTokenizer, BartForConditionalGeneration
from functools import cache
from typing import List
import nltk
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from rsasumm.rsa_reranker import kl_divergence, jensen_shannon_divergence, RSAReranking, RSARerankingEmbedder
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
from rsasumm.rsa_reranker import RSAReranking
import math
from typing import List, Tuple
import nltk
import numpy as np
import seaborn as sns
# EXAMPLES = [
#     "The paper gives really interesting insights on the topic of transfer learning. It is well presented and the experiment are extensive. I believe the authors missed Jane and al 2021. In addition, I think, there is a mistake in the math.",
#     "The paper gives really interesting insights on the topic of transfer learning. It is well presented and the experiment are extensive. Some parts remain really unclear and I would like to see a more detailed explanation of the proposed method.",
#     "The paper gives really interesting insights on the topic of transfer learning. It is not well presented and lack experiments. Some parts remain really unclear and I would like to see a more detailed explanation of the proposed method.",
# ]

# text1_sentences = nltk.sent_tokenize(EXAMPLES[0])
# text2_sentences = nltk.sent_tokenize(EXAMPLES[1])
# text3_sentences = nltk.sent_tokenize(EXAMPLES[2])

# # remove empty sentences
# text1_sentences = [
#     sentence for sentence in text1_sentences if sentence != ""]
# text2_sentences = [
#     sentence for sentence in text2_sentences if sentence != ""]
# text3_sentences = [
#     sentence for sentence in text3_sentences if sentence != ""]

# sentences = list(set(text1_sentences + text2_sentences + text3_sentences))

# # Initialize model and tokenizer
# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# # Dummy inputs
# source_texts = ["The paper is interesting."]
# candidate_summaries = ["Well-written summary."]

# # Create instance
# model_id = "facebook/bart-large-cnn"
# model = BartForConditionalGeneration.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# reranker = RSAReranking(model, tokenizer, source_texts=source_texts, candidates=candidate_summaries, device="cpu")

# # Test likelihood
# likelihood = reranker.compute_conditionned_likelihood(
#     source_texts, candidate_summaries)
# print("Likelihood:", likelihood)

# # Compute likelihood matrix
# matrix = reranker.likelihood_matrix()
# print("Likelihood Matrix:")
# print(matrix)

def sample_from_probs(
    logits: torch.Tensor, num_beams: torch.Tensor, do_sample: bool, K: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample or select top tokens from probabilities.

    Args:
        logits (torch.Tensor): (num_beams, vocab_size).
            Log probabilities of tokens for the next step.
        num_beams (torch.Tensor): Number of beams to sample.
        do_sample (bool): Whether to sample tokens or use argmax.
        K (int): Number of samples to draw per beam.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - idx_beam: Indices of sampled beams.
            - idx_token: Indices of sampled tokens.
            - tokens_scores: Log probabilities of sampled tokens.
    """
    vocab_size = logits.shape[-1]
    if do_sample:
        # sample from the probability distribution
        logits = logits.view(num_beams * logits.shape[-1])
        probs = torch.softmax(logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=K * num_beams)

        # get the indices of the sampled tokens
        idx_beam, idx_token = samples // vocab_size, samples % vocab_size

        logits = logits.view(num_beams * vocab_size)

        tokens_scores = logits.gather(dim=-1, index=samples).squeeze(-1)

        return idx_beam, idx_token, tokens_scores

    else:
        # get the indices of the most probable tokens
        num_beams = logits.shape[0]
        vocab_size = logits.shape[-1]

        logits = logits.view(num_beams * vocab_size)
        scores, samples = logits.topk(2 * num_beams, dim=-1)

        idx_beam, idx_token = samples // vocab_size, samples % vocab_size

        tokens_scores = scores.squeeze(-1)

        return idx_beam, idx_token, tokens_scores


logits = torch.tensor([[2.0, 0.5, 0.2], [1.0, 1.5, 0.5]])
num_beams = 2
do_sample = False
K = 2

idx_beam, idx_token, tokens_scores = sample_from_probs(logits, num_beams, do_sample, K)
print("Sampled Beams:", idx_beam)
print("Sampled Tokens:", idx_token)
print("Token Scores:", tokens_scores)