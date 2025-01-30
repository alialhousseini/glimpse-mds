from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm
import operator
from functools import reduce
from itertools import combinations
########################################################################


def elementwise_max(dfs):
    """
    dfs: list of DataFrames (same index/columns)
    """
    return reduce(lambda x, y: x.combine(y, func=lambda a, b: pd.Series([max(a_, b_) for a_, b_ in zip(a, b)])), dfs)

# Now we can write a script that takes the set of LM_probas for each dataset and (set) of models
# and aggregate them to get the final ranking


# We define a set of model names, this set represents the set of models we want to aggregate their results
# In addition we define a methodology of aggregation(e.g. mean, max, weighted_avg, etc.)
k = 5
model_names = ["LED-Large", "PEGASUS-Large",
               "PEGASUS-BigBird-Arxiv", "PEGASUS-XSUM", "FlanT5-Large-FT-OAI"]

# lm_probas_path = Path("/content/drive/MyDrive/lm_probas")
lm_probas_path = Path("data/lm_probas")
lm_probas_files = list(lm_probas_path.glob("*.pkl"))
comb_list = list(combinations(model_names, k))

for comb in comb_list:

    # We need to find for each set of common datasets, the models we are looking for:
    # Filter out the files that do not contain the models we are looking for
    # So we keep only the files that contain the models we are looking for
    lm_probas_path = Path("data/lm_probas")
    lm_probas_files = list(lm_probas_path.glob("*.pkl"))

    lm_probas_files = [file for file in lm_probas_files if any(
        model_name in file.stem.split('-_-')[-1] for model_name in comb)]

    # if len(lm_probas_files) // 4 <= 2:
    #     raise ValueError("Not enough files to aggregate")

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
    for filename, files in tqdm(files_and_pickles.items(), desc="Processing datasets"):
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

            assert all(set_of_dicts[i]['id'] == set_of_dicts[0]['id'] for i in range(len(set_of_dicts))), \
                f"Some IDs are not equal {set_of_dicts[i]['id']} != {set_of_dicts[0]['id']}"

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

            for idx, df in enumerate(set_of_dfs):
                if df.shape != set_of_dfs[0].shape:
                    raise ValueError(
                        f"Shape mismatch: DataFrame #{idx} has shape {df.shape}, expected {set_of_dfs[0].shape}")
                if not df.index.equals(set_of_dfs[0].index):
                    raise ValueError(
                        f"Index mismatch: DataFrame #{idx} index does not match the reference.")
                if not df.columns.equals(set_of_dfs[0].columns):
                    raise ValueError(
                        f"Column mismatch: DataFrame #{idx} columns do not match the reference.")

            # Combine data into a vector and save it in a new DF where each element
            # Is a list (vector)
            # Create a DataFrame where each cell is a list of integers

            df_concat = pd.DataFrame(
                [[list(row) for row in zip(*[df.iloc[i].values for df in set_of_dfs])]
                 for i in range(set_of_dfs[0].shape[0])],
                index=ref_index,
                columns=ref_columns
            )

            # Save it
            new_dict['lm_probas_concat'] = df_concat

            if method == "mean":

                # To aggregation safely
                df_sum = reduce(operator.add, set_of_dfs)
                df_agg = df_sum / len(set_of_dfs)

            if method == "max":
                df_agg = elementwise_max(set_of_dfs)

            # Save it!
            new_dict['language_model_proba_df'] = df_agg

            # Save model names used as well
            new_dict['model_names'] = list(comb)

            results.append(new_dict)

        results = {"results": results}
        # Save the results
        # CHANGE PATH BEFORE RUNNING

        opt_dir = Path(f'data/aggs_lm_vectorized/')
        if not opt_dir.exists():
            opt_dir.mkdir(parents=True, exist_ok=True)
        str_list_model_names = "["+str("_".join(list(comb)))+"]"
        # e.g. all_merged_226_-_extr-_-[BART_PEGASUS-Arxiv_Falcon]-_-mean.pkl
        new_filename = filename + "-_-" + str_list_model_names + "-_-" + method

        # CHANGE PATH BEFORE RUNNING
        opt_path = Path(f"data/aggs_lm_vectorized/{new_filename}.pkl")
        with open(opt_path, 'wb') as f:
            pickle.dump(results, f)

    print(f"Combination: {comb} is done!")

    ########################################################################
