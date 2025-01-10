from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm
import operator
from functools import reduce
from itertools import combinations
########################################################################

DESC = """
    Here after saving all data in aggs_lm that are 120*4 = 480 files
    We want to find out what is the best combination of models that leads to
    the best performance in terms of our metrics (ROUGE, BLEU, ...)
"""

aggs_lm_path = Path("data/aggs_lm")
aggs_lm_files = list(aggs_lm_path.glob("*.pkl"))
datasets = {}
for lm_file in aggs_lm_files:
    dataset_name , list_of_models , method = lm_file.stem.split('-_-')
    if dataset_name not in datasets:
        datasets[dataset_name] = [lm_file]
    else:
        datasets[dataset_name].append(lm_file)

