import json
from tqdm import tqdm
import numpy as np
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from aisafetylab.models import load_model
from aisafetylab.evaluation.scorers import LlamaGuard3Scorer, HarmBenchScorer
from aisafetylab.models import LocalModel

def load_file(path):
    if path.endswith('.jsonl'):
        with open(path) as f:
            return [json.loads(line) for line in f]
    
    if path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    
    raise ValueError(f'Unsupported file format: {path}')

def evaluate_asr(path, scorer, batch_size=8, save_path=None):
    data = load_file(path)
    all_scores = []
    try:
        responses = [d['output'] for d in data]
    except:
        responses = [d["response"] for d in data]
    with open("../data/harmbench_sorted.json", "r") as f:
        raw_queries = json.load(f)
    raw_queries = [d['goal'] for d in raw_queries]
    score_results = scorer.batch_score(raw_queries, responses, batch_size=batch_size)
    all_scores = [r['score'] for r in score_results]
    for i in range(len(data)):
        data[i]["question"] = raw_queries[i]
        data[i]['final_response'] = responses[i]
        data[i]["final_score"] = all_scores[i]

    print(f'ASR: {np.mean(all_scores)}')
    save_folder = os.path.dirname(save_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    device = torch.device('cuda:0')
    name = "file name of generated file from model"
    print(name)
    scorer = HarmBenchScorer(
        model_path="cais/HarmBench-Llama-2-13b-cls",
        tokenizer_path="cais/HarmBench-Llama-2-13b-cls",
        device=device
    )
    path = f"file path of generated file from model"
    evaluate_asr(
        path,
        scorer,
        batch_size=4,
        save_path=f"eval_results/gjo/{name}",
    )
