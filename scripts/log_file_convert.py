import re
import json
import os

def extract_log_data(log_file_path, target_folder, prefix, loc):
    with open(log_file_path, "r", encoding="utf-8") as file:
        log_content = file.read()
    
    step_matches = re.findall(r"Step   (\d+/\d+)", log_content)
    steps = 0
    if step_matches:
        print("match1")
        steps = step_matches[-1].split('/')[0]
        
    step_matches = re.findall(r"Step  (\d+/ \d+)", log_content)
    if step_matches:
        print("match2")
        new_steps = step_matches[-1].split('/')[0]
        if (int(new_steps) > int(steps)):
            steps = new_steps

    step_matches = re.findall(r"Step  (\d+/\d+)", log_content)
    if step_matches:
        print("match3")
        new_steps = step_matches[-1].split('/')[0]
        if (int(new_steps) > int(steps)):
            steps = new_steps
            
    passed_matches = re.findall(r"\(id_[a-z]+\) \| Passed  (\d+)/\d+", log_content)
    if passed_matches:
        passed = passed_matches[-1]

    passed_matches = re.findall(r"\(id_[a-z]+\) \| Passed   (\d+)/\d+", log_content)
    if passed_matches:
        passed = passed_matches[-1]

    count_match = re.search(r"_len_(\d+)_", log_file_path)
    if count_match:
        count = count_match.group(1)
    
    control_prompts = re.findall(r"control='(.*?)===========================.*?", log_content, re.DOTALL)
    
    if control_prompts:
        last_control_prompt = control_prompts[loc][:-2]
    else:
        print("control prompt not found")
        return
        
    result = {
        "steps": steps,
        "final_optimization_range": passed,
        "prefix": prefix,
        "length": count,
        "prompt": last_control_prompt
    }

    json_file_name = os.path.basename(log_file_path).replace('.txt', '.json')
    json_output_path = os.path.join(target_folder, json_file_name)
    
    try:
        with open(json_output_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = []
    
    data.append(result)

    os.makedirs(target_folder, exist_ok=True)
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

log_file_path="./logs/data_20_len_100_llama3_sys_False_2_token.txt"
target_folder = '../data/prompts/gjo'

extract_log_data(log_file_path, target_folder, prefix=True, loc=-1)