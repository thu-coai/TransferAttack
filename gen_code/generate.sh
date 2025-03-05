model_name="llama2"  # llama2, vicuna, yi-1.5-9b, qwen2, gemma-7b-it, llama3
base_model=mlsys/llama-2-7b-chat-hf
tokenizer_path=mlsys/llama-2-7b-chat-hf
input_file= # input file path
output_file= # output file path

CUDA_VISIBLE_DEVICES=3 python generate.py --base_model ${base_model} --tokenizer_path ${tokenizer_path} --input_file ${input_file} --output_file ${output_file} --limit 0 --regen 1 --batchsize 8