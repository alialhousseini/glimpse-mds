import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

map_questionnumber_to_question = {
        "repetition": "2",
        "grammar": "3",
        "attribution": "4",
        "main ideas": "5",
        "conciseness": "6"
    }


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name to be used as a folder name.
    @param model_name: The model name
    @return: The sanitized model name
    """
    return model_name.replace("/", "_")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        type=str,
        default="repetition",
    )
    parser.add_argument("--summaries_folder", type=Path, required=True, help="Folder containing summary files.")
    parser.add_argument("--output_folder", type=Path, required=True, help="Folder to save the output metrics.")
    parser.add_argument("--select", type=str, default="*")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args

def parse_summaries(path: Path):
    """
    :return: a pandas dataframe with at least the columns 'text' and 'summary'
    """

    df = pd.read_csv(path).dropna()

    if not all([col in df.columns for col in ["text", "summary"]]):
        raise ValueError("The csv file must have the columns 'text' and 'summary'.")

    return df

def evaluate_classification_task(model, tokenizer, question, df, batch_size):
    batch_size = 8
    texts = df.text.tolist()
    summaries = df.summary.tolist()

    template = "premise: {premise} hypothesis: {hypothesis}"
    ds = [template.format(premise=text[:20*1024], hypothesis=summary) for text, summary in zip(texts, summaries)]

    eval_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    metrics = {f"{question}/proba_1": [], f"{question}/proba_0": [], f"{question}/guess": []}

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            N_inputs = inputs["input_ids"].shape[0]
            decoder_input_ids = torch.full((N_inputs, 1), tokenizer.pad_token_id, dtype=torch.long, device=model.device)

            outputs = model(**inputs, decoder_input_ids=decoder_input_ids)

            del inputs, decoder_input_ids
            torch.cuda.empty_cache()

            logits = outputs.logits
            del outputs

            filtered_logits = logits[:, -1, [497, 333]]
            del logits 

            probs = F.softmax(filtered_logits, dim=-1)
            del filtered_logits

            guess = probs.argmax(dim=-1)

            metrics[f"{question}/proba_1"].extend(probs[:, 1].tolist())
            metrics[f"{question}/proba_0"].extend(probs[:, 0].tolist())
            metrics[f"{question}/guess"].extend(guess.tolist())

            del probs, guess
            torch.cuda.empty_cache()

    return metrics



def process_files_in_folder(input_folder: Path, output_folder: Path, model, tokenizer, question, batch_size, metricName):
    """
    Process all summary files in a folder.
    """
    # Combine output_folder and metricName into a single path
    combined_output_folder = output_folder / metricName
    combined_output_folder.mkdir(parents=True, exist_ok=True)

    for summary_file in input_folder.glob("*.csv"):
        print(f"Processing file: {summary_file.name}")

    

        df = parse_summaries(summary_file)

        print(str(summary_file.stem)+".csv")


        metrics = evaluate_classification_task(model, tokenizer, question, df, batch_size)
        metrics_df = pd.DataFrame(metrics)

        df = pd.concat([df, metrics_df], axis=1)
        output_path = combined_output_folder / f"{summary_file.stem}_metrics.csv"
        
        df.to_csv(output_path, index=False)
        print(f"Saved metrics to: {output_path}")


def main():
    args = parse_args()

    for question, question_number in map_questionnumber_to_question.items():
        
        if question != "conciseness":
            continue
        
        model_name = f"google/seahorse-large-q{question_number}"

        print(f"Processing question: {question} with model {model_name}")

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        question_output_folder = args.output_folder / question
        process_files_in_folder(args.summaries_folder, question_output_folder, model, tokenizer, question, args.batch_size,question )

        del model, tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
