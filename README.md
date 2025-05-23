# TransferAttack

![method](<imgs/main_figure.png>)

This is the codebase for our paper [Guiding not Forcing: Enhancing the Transferability of Jailbreaking Attacks on LLMs via Removing Superfluous Constraints](https://arxiv.org/abs/2503.01865).

We introduce a novel conceptual framework to elucidate transferability and identify superfluous constraints—specifically, the response pattern constraint and the token tail constraint—as significant barriers to improved transferability. Our method, Guided Jailbreaking Optimization, increases the overall Transfer Attack Success Rate (T-ASR) across a set of target models with varying safety levels **from 18.4\% to 50.3\%**, while also improving the stability and controllability of jailbreak behaviors on both source and target models. Please refer to our [paper](https://arxiv.org/abs/2503.01865) for more details.

## News

- 🎉🎉🎉 Our Paper `Guiding not Forcing: Enhancing the Transferability of Jailbreaking Attacks on LLMs via Removing Superfluous Constraints` is accepted by ACL 2025 Main! Please feel free to contact us if you have any questions about our work.

## Quick Start

### Setup

We require [AISafetyLab](https://arxiv.org/abs/2502.16776) for quick evaluation and the latest FastChat for searching on Llama3. Follow these steps to set up the required dependencies:
```shell
# Clone and install AISafetyLab
git clone git@github.com:thu-coai/AISafetyLab.git
cd AISafetyLab
pip install -e .

# Clone and install FastChat
git clone git@github.com:lm-sys/FastChat.git
cd FastChat
pip install -e .
```

Then, install the remaining dependencies:
```shell
git clone git@github.com:thu-coai/TransferAttack.git
cd TransferAttack
pip install -e .
```

### Searching code

#### Data Construction
We provide a script to construct search data for GCG-Adaptive and our method:
```shell
cd scripts
python construct_data.py
```

#### Running the Search Script
We build upon the GCG attack framework and integrate our method. Use the following commands to run the search:
```shell
bash run_gjo.sh llama2
bash run_gjo.sh llama3
```
Remember to change the running config in `./scripts/configs`

#### Post Processing
We provide scripts to extract adversarial prompts from log files and combine them with test questions and model chat templates:
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
We support evaluation using `AISafetyLab`. Run the evaluation script as follows:
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
