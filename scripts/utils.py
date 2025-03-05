def build_chat(query, model_name):
    if model_name == 'llama2':
        # no safety prompt
        opt = f'[INST] {query} [/INST] '
    elif model_name == 'llama2_system': # official prompt
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        opt = f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{query} [/INST]'
    elif model_name == 'mistral':
        opt = f'[INST] {query} [/INST]'
    elif model_name == 'mistral_system':
        opt = f'[INST] You are an Al assistant. Always assist with care, respect, and truth. Please respond with utmost utility yet securely and avoid harmful, unethical, prejudiced, or negative content. Also ensure replies promote fairness and positivity.\n\n{query} [/INST]'
    elif model_name == 'llama3_system':
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        opt = f'<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
    elif model_name == 'llama3':
        opt = f'<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    elif model_name == 'gemma-7b-it':
        opt = f'<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n'
    elif model_name == 'vicuna':
        opt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {query} ASSISTANT:"
    elif model_name == 'qwen2':
        opt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    elif model_name == 'yi-1.5-9b':
        opt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    else:
        raise Exception(f"Model name {model_name} not supported")
    
    return opt