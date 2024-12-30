import os
import subprocess
import sys

# Set default dataset path
DEFAULT_DATASET_PATH = "data/processed/all_reviews_2017.csv"

def main(dataset_path=None):
    # Verify dataset path
    if dataset_path is None or not os.path.isfile(dataset_path):
        print(f"Couldn't find a valid path. Using default path: {DEFAULT_DATASET_PATH}")
        dataset_path = DEFAULT_DATASET_PATH

    # Load necessary modules (adapt for your system or remove if not applicable)
    """try:
        subprocess.run(["module", "--quiet", "load", "miniconda/3"], check=True)
        subprocess.run(["module", "--quiet", "load", "cuda/12.1.1"], check=True)
    except FileNotFoundError:
        print("`module` command not found. Ensure you have the required modules loaded.")
        sys.exit(1)

    # Activate the conda environment
    try:
        subprocess.run(["conda", "activate", "glimpse"], shell=True, check=True)
    except subprocess.CalledProcessError:
        print("Failed to activate the conda environment. Ensure `glimpse` is installed.")
        sys.exit(1)"""

    # Generate extractive summaries
    """try:
        extractive_command = [
            "python",
            "glimpse/data_loading/generate_extractive_candidates.py",
            "--dataset_path",
            dataset_path,
            "--scripted-run",
        ]
        candidates = subprocess.check_output(extractive_command).decode("utf-8").strip().split("\n")[-1]
        print(f"Generated candidates: {candidates}")
    except subprocess.CalledProcessError:
        print("Failed to generate extractive summaries.")
        sys.exit(1)"""
    candidates = "D:\\Universita\\Progetto NLP\\model_evaluation\\NLP-Project\\finalCandidates"

    # Compute RSA scores
    try:
        rsa_command = [
            "python",
            "glimpse/src/compute_rsa.py",
            "--summaries_folder",
            candidates,
            "--model_name",
            "prova/BART"
        ]
        rsa_scores = subprocess.check_output(rsa_command).decode("utf-8").strip().split("\n")[-1]
        print(f"Computed RSA scores: {rsa_scores}")
    except subprocess.CalledProcessError:
        print("Failed to compute RSA scores.")
        sys.exit(1)

if __name__ == "__main__":
    dataset_path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(dataset_path_arg)
