# TransferAttack

![method](<imgs/main_figure.png>)

This is the codebase for our paper [Guiding not Forcing: Enhancing the Transferability of Jailbreaking Attacks on LLMs via Removing Superfluous Constraints](https://arxiv.org/abs/2503.01865).

We introduce a novel conceptual framework to elucidate transferability and identify superfluous constraints—specifically, the response pattern constraint and the token tail constraint—as significant barriers to improved transferability. Our method, Guided Jailbreaking Optimization, increases the overall Transfer Attack Success Rate (T-ASR) across a set of target models with varying safety levels **from 18.4\% to 50.3\%**, while also improving the stability and controllability of jailbreak behaviors on both source and target models. Please refer to our [paper](https://arxiv.org/abs/2503.01865) for more details.


## Quick Start

### Setup

AISafetyLab is required since we utilize it for quick evaluation, and newest Fastchat is required for searching on Llama3.
```shell
git clone git@github.com:thu-coai/AISafetyLab.git
cd AISafetyLab
pip install -e .

git clone git@github.com:lm-sys/FastChat.git
cd FastChat
pip install -e .
```
And then install the others
```shell
git clone git@github.com:thu-coai/TransferAttack.git
cd TransferAttack
pip install -e .
```

### Searching code

#### Data Construction
We provide a script to construct searching data for GCG-Adaptive and Ours.
```shell
cd scripts
python construct_data.py
```

#### Search Script
We adopt the attack framework of GCG attack, and add our method. Here's how to run the searching code:
```shell
bash run_gjo.sh llama2
bash run_gjo.sh llama3
```
Remember to change the running config in `./scripts/configs`

#### Post Process
We provide quick scripts to extract the adversarial prompt from log file and combine it with test questions and model chat template.
```shell
python log_file_convert.py
bash gcg_combine.py
```

### Generation
After searching for the adversarial attack prompt, we provide a script supporting batch generation to get the response of the model. Remember to change the model/tokenizer path and the input/output path. 
```shell
cd gen_code
bash generate.sh
```

### Evaluation
We provide evaluation script supported by `AISafetyLab`.
```shell
cd evaluation
python harmbench_eval.py
```

## Citation

```
@misc{yang2025guidingforcingenhancingtransferability,
      title={Guiding not Forcing: Enhancing the Transferability of Jailbreaking Attacks on LLMs via Removing Superfluous Constraints}, 
      author={Junxiao Yang and Zhexin Zhang and Shiyao Cui and Hongning Wang and Minlie Huang},
      year={2025},
      eprint={2503.01865},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.01865}, 
}
```