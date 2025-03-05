import os
import json, re
from utils import build_chat
from construct_data import get_universal_manual_prompt

if __name__ == "__main__":
    suffix_dir = '../data/prompts/gjo'
    file_name = "data_20_len_100_llama3_sys_False_2_token.json"
    target_file = '../data/harmbench_sorted.json'
    
    output_dir = '../data/harmbench/gjo'
    output_filename = file_name
    prompt_template = "gjo"
    output_filepath = os.path.join(output_dir, output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(target_file) as f:
        target_datas = json.load(f)

    output = []
    sum = 0
    with open(os.path.join(suffix_dir, file_name), 'r') as f:
        data = json.load(f)
    prompt = data[-1].get("prompt", "")
    prefix = data[-1].get("prefix", False)
    count = data[-1].get("length", "")
    for id, target_data in enumerate(target_datas):
        question = target_data.get('goal', "")
        if prefix:
            v, _ = get_universal_manual_prompt(prompt_template, target_data["target"], question)
            combined = f"{prompt} {v}"
        else:
            v, _ = get_universal_manual_prompt(prompt_template, target_data["target"], question)
            combined = f"{v} {prompt}"
        output.append({
            "id": sum,
            "prompt": combined,
            "question_idx": id,
            "suffix_length": count
            })
        sum += 1

    with open(output_filepath, 'w') as out_file:
        json.dump(output, out_file, ensure_ascii=False, indent=4)
        
    for model in ['llama3', 'llama2', 'qwen2', 'yi-1.5-9b', 'gemma-7b-it',"vicuna"]:
        model_output = []
        model_output_dir = os.path.join(output_dir, model)
        model_output_filepath = os.path.join(model_output_dir, output_filename)
        sum = 0
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        for id, target_data in enumerate(target_datas):
            question = target_data.get('goal', "")
            # combined = f"{question}"
            if prefix:
                v, _ = get_universal_manual_prompt(prompt_template, target_data["target"], question)
                combined = f"{prompt} {v}"
            else:
                v, _ = get_universal_manual_prompt(prompt_template, target_data["target"], question)
                combined = f"{v} {prompt}"
            combined = build_chat(combined, model)
            model_output.append({
                "id": sum,
                "prompt": combined,
                "question_idx": id,
                "suffix_length": count
                })
            sum += 1
            
        with open(model_output_filepath, 'w') as out_file:
            json.dump(model_output, out_file, ensure_ascii=False, indent=4)