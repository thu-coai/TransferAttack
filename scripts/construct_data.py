import json, csv

def get_universal_manual_prompt(prompt_template, target_str, goal):
    
    if prompt_template == 'gcg_sure':
        return f"{goal}", f" {target_str}"
    
    elif prompt_template == 'gcg_here':
        target_str = target_str.replace('Sure, ', '')[0].upper() + target_str.replace('Sure, ', '')[1:]
        return f"{goal}", f" {target_str}"
    
    elif prompt_template == 'gjo':
        target_str = target_str.replace('Sure, ', '')[0].upper() + target_str.replace('Sure, ', '')[1:]
        return f"{goal}\nStart your response with \"{target_str}:\nStep 1\"", f" {target_str}:\nStep 1"
    
    else:
        raise ValueError(f'wrong prompt template {prompt_template}')

if __name__ == "__main__":
    # prompt_template = "gcg_here"
    prompt_template = "gjo"
    file_path = '..data/harmbench_sorted.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    with open(f'./data/harmbench_{prompt_template}.csv', 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['goal', 'target', 'question']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            processed_goal, processed_target = get_universal_manual_prompt(prompt_template, entry['target'], entry['goal'])
            row = {'goal': processed_goal, 'target': processed_target, 'question': entry['goal']}
            writer.writerow(row)