from sumy.parsers.plaintext import PlaintextParser
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from tqdm import tqdm
import argparse

import pandas as pd
from pathlib import Path

import nltk


def summarize(method, language, sentence_count, input_type, input_):
    if method == 'LSA':
        from sumy.summarizers.lsa import LsaSummarizer as Summarizer
    if method == 'text-rank':
        from sumy.summarizers.text_rank import TextRankSummarizer as Summarizer
    if method == 'lex-rank':
        from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer
    if method == 'edmundson':
        from sumy.summarizers.edmundson import EdmundsonSummarizer as Summarizer
    if method == 'luhn':
        from sumy.summarizers.luhn import LuhnSummarizer as Summarizer
    if method == 'kl-sum':
        from sumy.summarizers.kl import KLSummarizer as Summarizer
    if method == 'random':
        from sumy.summarizers.random import RandomSummarizer as Summarizer
    if method == 'reduction':
        from sumy.summarizers.reduction import ReductionSummarizer as Summarizer

    if input_type == "URL":
        parser = HtmlParser.from_url(input_, Tokenizer(language))
    if input_type == "text":
        parser = PlaintextParser.from_string(input_, Tokenizer(language))

    stemmer = Stemmer(language)
    summarizer = Summarizer(stemmer)
    stop_words = get_stop_words(language)

    if method == 'edmundson':
        summarizer.null_words = stop_words
        summarizer.bonus_words = parser.significant_words
        summarizer.stigma_words = parser.stigma_words
    else:
        summarizer.stop_words = stop_words

    summary_sentences = summarizer(parser.document, sentence_count)
    summary = ' '.join([str(sentence) for sentence in summary_sentences])

    return summary


# methods = ['LSA', 'text-rank', 'lex-rank', 'edmundson', 'luhn', 'kl-sum', 'random', 'reduction']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=Path,  default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_folder", type=Path, default="")

    args = parser.parse_args()
    return args


# group text by sample id and concatenate text

def group_text_by_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group the text by the sample id and concatenate the text.
    :param df: The dataframe
    :return: The dataframe with the text grouped by the sample id
    """
    # Was written wrong! this way will repeat each text as many sentences we have
    # texts = df.groupby("id")["text"].apply(lambda x: " ".join(x))

    texts = df.groupby('id')['text'].unique().apply(lambda x: " ".join(x))

    # retrieve first gold by id
    gold = df.groupby("id")["gold"].first()

    # create new dataframe
    df = pd.DataFrame({"text": texts, "gold": gold}, index=texts.index)

    return df


def main():
    args = parse_args()

    methods = ['LSA', 'text-rank', 'lex-rank', 'edmundson',
               'luhn', 'kl-sum', 'random', 'reduction']

    # Iterate over files in the input folder
    for file in tqdm(args.input_folder.glob("*.csv")):
        # For each, apply all methods and save summaries

        # Read current dataset
        dataset = pd.read_csv(file)
        # Process it
        # Concat all reviews , and return gold -> (sum_rev_i, gold)
        dataset = group_text_by_id(dataset)

        for method in tqdm(methods):
            summaries = []
            for text in dataset.text:
                summary = summarize(method, "english", 1, "text", text)
                summaries.append(summary)

            dataset[f'summary_{method}'] = summaries

        # Save the dataset
        # create folder if not exists
        if not Path(args.output_folder).exists():
            Path(args.output_folder).mkdir(parents=True, exist_ok=True)
        path_to_save = Path(args.output_folder) / \
            Path(file.stem + "_summy.csv")
        dataset.to_csv(path_to_save, index=True)


main()
