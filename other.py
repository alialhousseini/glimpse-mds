from utils import convert_to_json
from metric.evaluator import get_evaluator

task = 'summarization'

gspeaker_path = Path("/content/drive/MyDrive/allSummariesGSpeaker")
gunique_path = Path("/content/drive/MyDrive/allSummariesGUnique")
gsummy_path = Path("/content/drive/MyDrive/summy")


for path in [gspeaker_path, gunique_path, gsummy_path]:

    if path == gsummy_path:
        for file in tqdm(path.glob("*.csv")):
            df = pd.read_csv(file)
            summary_cols = [col for col in df.columns if "summary" in col]
            src_list = df.text.tolist()
            for col in summary_cols:
                output_list = df[col].values.tolist()
                model_name = col.split("_")[1]
                data = convert_to_json(output_list=output_list,
                                       src_list=src_list)

                evaluator = get_evaluator(task)

                eval_scores = evaluator.evaluate(data, dims=['coherence', 'consistency', 'fluency'],
                                                 overall=False, print_result=False)

                coherence = [eval_scores[i]['coherence']
                             for i in range(len(eval_scores))]
                consistency = [eval_scores[i]['consistency']
                               for i in range(len(eval_scores))]
                fluency = [eval_scores[i]['fluency']
                           for i in range(len(eval_scores))]

                df[f'coherence_{model_name}'] = coherence
                df[f'consistency_{model_name}'] = consistency
                df[f'fluency_{model_name}'] = fluency

            df.to_csv(file)

    else:
        for file in tqdm(path.glob("*.csv")):
            # open the file
            df = pd.read_csv(file)
            src_list = df.text.tolist()
            output_list = df.summary.tolist()

            data = convert_to_json(output_list=output_list,
                                   src_list=src_list)

            evaluator = get_evaluator(task)

            eval_scores = evaluator.evaluate(data, dims=['coherence', 'consistency', 'fluency'],
                                             overall=False, print_result=False)

            coherence = [eval_scores[i]['coherence']
                         for i in range(len(eval_scores))]
            consistency = [eval_scores[i]['consistency']
                           for i in range(len(eval_scores))]
            fluency = [eval_scores[i]['fluency']
                       for i in range(len(eval_scores))]

            df['coherence'] = coherence
            df['consistency'] = consistency
            df['fluency'] = fluency

            df.to_csv(file)
