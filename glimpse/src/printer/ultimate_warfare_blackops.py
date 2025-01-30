'''
This script is reponsible on making our results the best!
How? Simply by predicting the future!

We want to know how many 'k' aggregated models we need to get the best results.
And we are relying on computing ROUGE, BERTscore and discriminativeness for that.
But still we cannot decide which models are the best to select and contributing the most to the final results.

Therefore we decided to create this script to help us in this task.
The idea is very simple , we have 4 datasets (3 abs datasets and 1 fulltext dataset)

Now for each dataset we computed the LM Probas using 10 different models.
We want to know which models are the best to select and contribute the most to the final results.
This can be done by computing results after applying RSA on the aggregated values of the models. (and compute the average)
For this reason we will iterate between 2->4 (k) models and compute the results for each k combination of models, for each dataset.

Finally, we can notice which models are the best to select and contribute the most to the final results.
E.g. if for a dataset, say the extractive one, we see that:
Top 1 ROUGE L: model 1, model 2, model 3
Top 1 BERTscore: model 4, model 5, model 2
Top 1 Discriminativeness: model 2, model 3, model 6
So 2,3 are good to be togehter.

'''

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

gspeaker_path = Path('GLSUNIQUESEAHORSE/')
info = {}

for file in tqdm(gspeaker_path.glob('*.csv')):
    name, _ = file.stem.split('-_-')
    if name not in info:
        info[name] = {}


for k in info.keys():
    models_metrics_pairs = {}
    for file in gspeaker_path.glob(f'{k}-_-*.csv'):
        records = {}  # key = metric, value = average of that value
        model_name = file.stem.split('-_-')[1].split('_')[0]
        df = pd.read_csv(file)
        df["summary_char_count"] = df["summary"].apply(len)
        df['proba_of_success'] = df['proba_of_success'] / df['summary_char_count']
        # records['ROUGE_1'] = df['rouge1'].mean()
        # records['ROUGE_2'] = df['rouge2'].mean()
        # records['ROUGE_L'] = df['rougeL'].mean()
        # records['ROUGE_LSum'] = df['rougeLsum'].mean()
        # records['BERTscore'] = df['BERTScore'].mean()
        records['repitition'] = df['repetition/proba_1_repetition'].mean()
        records['grammar'] = df['grammar/proba_1_grammar'].mean()
        records['attribution'] = df['attribution/proba_1_attribution'].mean()
        records['main_ideas'] = df['main ideas/proba_1_main ideas'].mean()
        try:
            records['conciseness'] = df['conciseness/proba_1_conciseness'].mean()
        except:
            print(f"Error in {file.stem}")
        records['disc_per_char'] = df['proba_of_success'].mean()
        records['coherence'] = df['coherence'].mean()
        records['consistency'] = df['consistency'].mean()
        records['fluency'] = df['fluency'].mean()
        # records['ROUGE_1_width'] = round(
        #     (1.96 * df['rouge1'].std())/np.sqrt(len(df)), 3)
        # records['ROUGE_2_width'] = round(
        #     (1.96 * df['rouge2'].std())/np.sqrt(len(df)), 3)
        # records['ROUGE_L_width'] = round(
        #     (1.96 * df['rougeL'].std())/np.sqrt(len(df)), 3)
        # records['ROUGE_LSum_width'] = round(
        #     (1.96 * df['rougeLsum'].std())/np.sqrt(len(df)), 3)
        # records['BERTscore_width'] = round(
        #     (1.96 * df['BERTScore'].std())/np.sqrt(len(df)), 3)
        # records['disc_per_char_width'] = round(
        #     (1.96 * df['proba_of_success'].std())/np.sqrt(len(df)), 3)

        models_metrics_pairs[model_name] = records
    info[k] = models_metrics_pairs


# # SUMY LOOP
# summy_path = Path('data/summy/')
# for file in tqdm(summy_path.glob('*.csv')):
#     name, _ = file.stem.split('-_-')
#     df = pd.read_csv(file)
#     metrics_col = [col for col in df.columns if col.split('_')[0] in [
#         'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'BERTScore', 'ProbaOfSuccess']]
#     summy_cols = [col for col in df.columns if 'summary' in col]
#     for col in summy_cols:
#         _, method = col.split('_')
#         df[f"summary_char_count_{method}"] = df[col].apply(len)
#         df[f'ProbaofSuccess_{method}'] = df[f'ProbaofSuccess_{method}'] / \
#             df[f'summary_char_count_{method}']
#     method_names = [col.split('_')[1] for col in summy_cols]
#     for method_name in method_names:
#         info[name][method_name] = {}
#         metrics_col = [
#             col for col in df.columns if method_name in col and 'summary' not in col]
#         for metric in metrics_col:
#             info[name][method_name][metric] = df[f"{metric.split('_')[0]}_{method_name}"].mean(
#             )
#             metric_mod = metric+'_width'

#             info[name][method_name][metric_mod] = round(
#                 (1.96 * df[f"{metric.split('_')[0]}_{method_name}"].std())/np.sqrt(len(df)), 3)

for k, v in info.items():
    for model_name, metrics in v.items():
        for metric_name, metric_value in metrics.items():
            print(f"{k} - {model_name} - {metric_name} - {metric_value}")

# for k, _ in info.items():
#     for model_name, metrics in info[k].items():
#         if model_name in ['LSA', 'text-rank', 'lex-rank', 'edmundson', 'luhn', 'kl-sum', 'random', 'reduction']:
#             for metric in list(metrics.keys()):
#                 if 'rouge1' in metric:
#                     info[k][model_name]['ROUGE_1'] = info[k][model_name].pop(
#                         metric)
#                 elif 'rouge2' in metric:
#                     info[k][model_name]['ROUGE_2'] = info[k][model_name].pop(
#                         metric)
#                 elif 'rougeL' in metric:
#                     info[k][model_name]['ROUGE_L'] = info[k][model_name].pop(
#                         metric)
#                 elif 'rougeLsum' in metric:
#                     info[k][model_name]['ROUGE_LSum'] = info[k][model_name].pop(
#                         metric)
#                 elif 'BERTScore' in metric:
#                     info[k][model_name]['BERTscore'] = info[k][model_name].pop(
#                         metric)
#                 elif 'ProbaofSuccess' in metric:
#                     info[k][model_name]['disc_per_char'] = info[k][model_name].pop(
#                         metric)

# for k, v in info.items():
#     for model_name, metrics in v.items():
#         for metric_name, metric_value in metrics.items():
#             print(f"{k} - {model_name} - {metric_name} - {metric_value}")
# Setting up seaborn theme
sns.set_theme(style="whitegrid")

# # Prepare data for plotting


def prepare_plot_data(nested_dict, level1_key):
    data = []
    for model_name, metrics in nested_dict[level1_key].items():
        for metric_name, metric_value in metrics.items():
            if metric_name.endswith("_width"):
                continue
            data.append(
                {"Model": model_name, "Metric": metric_name, "Value": metric_value})
    return pd.DataFrame(data)


# Updated code to avoid annotating 0.000 values
for level1_key in info.keys():
    df = prepare_plot_data(info, level1_key)
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(data=df, x="Model", y="Value",
                     hue="Metric", errorbar=None, dodge=True)

    # Annotate values, skip 0.00001 values
    for p in ax.patches:
        if abs(p.get_height()) > 0.00001:  # Skip annotations for values close to 0
            ax.annotate(f"{p.get_height():.3f}",
                        (p.get_x() + p.get_width() / 2., p.get_height() + 0.005),
                        ha='center', va='bottom', fontsize=8, color='black', rotation=90)

    plt.title(
        f"Metrics Comparison for {level1_key.split('-_-')[0]} - Glimpse Speaker", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Metric Value", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
