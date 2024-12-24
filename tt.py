

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class T5LikelihoodScorer:
    def __init__(self, model_name="t5-small", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name).to(self.device)

    def compute_likelihood(self, x: list[str], y: list[str], mean: bool = True) -> torch.Tensor:
        """
        Compute likelihood for T5 (or other encoder-decoder models).

        :param x: List of "source" texts (e.g., prompts).
        :param y: List of "target" texts (e.g., summaries).
        :param mean: If True, return average token log likelihood. 
                     If False, return the total log likelihood.
        :return: A tensor containing likelihoods for each input pair.
        """
        assert len(x) == len(
            y), "Source and target lists must have the same length."

        # CrossEntropyLoss without reduction; we will handle reduction manually
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        # Tokenize inputs and outputs
        x_enc = self.tokenizer(x, return_tensors="pt",
                               padding=True, truncation=True).to(self.device)
        y_enc = self.tokenizer(y, return_tensors="pt",
                               padding=True, truncation=True).to(self.device)

        # Forward pass through the model
        outputs = self.model(
            input_ids=x_enc["input_ids"],
            attention_mask=x_enc["attention_mask"],
            decoder_input_ids=y_enc["input_ids"],
            decoder_attention_mask=y_enc["attention_mask"]
        )
        logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]

        # Shift logits and labels for causal prediction
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = y_enc["input_ids"][..., 1:].contiguous()

        # Compute token-level negative log likelihood
        likelihood = -loss_fn(
            shifted_logits.view(-1, shifted_logits.size(-1)),  # Flatten logits
            shifted_labels.view(-1)                           # Flatten labels
        )

        # Reshape to [batch_size, seq_len - 1] to match sequence structure
        likelihood = likelihood.view(len(x), -1)

        # Sum over tokens in each sequence
        likelihood_per_sequence = likelihood.sum(dim=-1)

        if mean:
            # Normalize by the number of non-padding tokens in the target
            target_lengths = (y_enc["input_ids"] !=
                              self.tokenizer.pad_token_id).sum(dim=-1)
            likelihood_per_sequence /= target_lengths.float()

        return likelihood_per_sequence

    def compute_summarization_likelihood(self, x: list[str], y: list[str], mean: bool = True):
        """
        Compute the likelihood of a reference summary (y) given an input (x) using T5,
        while optionally generating a model-generated summary for comparison.

        :param x: List of "source" texts.
        :param y: List of "target" reference summaries.
        :param mean: If True, average the likelihood by the number of target tokens.
        :return: A dictionary containing:
                - likelihoods: Likelihood for each reference summary.
                - generated: Generated summaries for each input (optional).
        """
        assert len(x) == len(
            y), "Source and target lists must have the same length."

        x = ['Summarize: ' + x[i] for i in range(len(x))]
        print(x)
        # Generate summaries for comparison
        generated_summaries = self.tokenizer.batch_decode(
            self.model.generate(
                self.tokenizer(x, return_tensors="pt", padding=True,
                               truncation=True).input_ids.to(self.device),
                max_length=50,  # Adjust max length as needed
                num_beams=4,    # Beam search for better summaries
                early_stopping=True
            ),
            skip_special_tokens=True
        )
        print(generated_summaries)
        # Compute likelihoods for reference summaries (y)
        likelihoods = self.compute_likelihood(generated_summaries, y, mean)

        return likelihoods


# Example usage
if __name__ == "__main__":
    scorer = T5LikelihoodScorer(model_name="t5-small")
    x = ["The capital of France is Paris, and I don't know if I have to go to another city, what do you suggest?", "The largest ocean is"]
    y = ["Paris.", "the Pacific Ocean."]

    likelihoods = scorer.compute_summarization_likelihood(x, y)
    print("Likelihoods:", likelihoods)
