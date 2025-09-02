# LM-Harmony

![train-before-test](assets/banner.png)

LM-Harmony is an automatic evaluation tool for large language models. Unlike popular direct evaluations for LLMs, LM-Harmony uses a train-before-test evaluation paradigm. Each model is fine-tuned on the corresponding training set before evaluation. Our results demonstrate that LM-Harmony provides significantly more consistent model rankings across various tasks, revealing the true general capabilities of each model.

## Dependencies

```bash
conda env create -f environment.yml
```

## Example Usage

```python
python main.py --exp_name arc_easy/EleutherAI/pythia-70m-deduped --base_model EleutherAI/pythia-70m-deduped --task_name arc_easy --train_param.num_train_epochs 5 --eval_before_train --eval_every_epoch --eval_after_train --train_param.greater_is_better --train_param.metric_for_best_model eval_acc_norm,none --dataset_param.max_num_train 50000 --dataset_param.max_num_valid 1000 --dataset_param.max_num_test 10000 --no-use_git
```

You can specify the model by *--base_model*. We use the following models in our paper.

- openai-community/gpt2
- openai-community/gpt2-medium
- openai-community/gpt2-large
- openai-community/gpt2-xl
- meta-llama/Meta-Llama-3-8B
- meta-llama/Llama-3.1-8B
- meta-llama/Llama-3.2-1B
- meta-llama/Llama-3.2-3B
- meta-llama/Meta-Llama-3-8B-Instruct
- meta-llama/Llama-3.1-8B-Instruct
- meta-llama/Llama-3.2-1B-Instruct
- meta-llama/Llama-3.2-3B-Instruct
- Qwen/Qwen1.5-0.5B
- Qwen/Qwen1.5-1.8B
- Qwen/Qwen1.5-4B
- Qwen/Qwen1.5-7B
- Qwen/Qwen1.5-14B
- Qwen/Qwen1.5-0.5B-Chat
- Qwen/Qwen1.5-1.8B-Chat
- Qwen/Qwen1.5-4B-Chat
- Qwen/Qwen1.5-7B-Chat
- Qwen/Qwen1.5-14B-Chat
- Qwen/Qwen2-0.5B
- Qwen/Qwen2-1.5B
- Qwen/Qwen2-7B
- Qwen/Qwen2-0.5B-Instruct
- Qwen/Qwen2-1.5B-Instruct
- Qwen/Qwen2-7B-Instruct
- Qwen/Qwen2.5-0.5B
- Qwen/Qwen2.5-1.5B
- Qwen/Qwen2.5-3B
- Qwen/Qwen2.5-7B
- Qwen/Qwen2.5-14B
- Qwen/Qwen2.5-0.5B-Instruct
- Qwen/Qwen2.5-1.5B-Instruct
- Qwen/Qwen2.5-3B-Instruct
- Qwen/Qwen2.5-7B-Instruct
- Qwen/Qwen2.5-14B-Instruct
- EleutherAI/pythia-70m-deduped
- EleutherAI/pythia-160m-deduped
- EleutherAI/pythia-410m-deduped
- EleutherAI/pythia-1b-deduped
- EleutherAI/pythia-1.4b-deduped
- EleutherAI/pythia-2.8b-deduped
- EleutherAI/pythia-6.9b-deduped
- EleutherAI/pythia-12b
- google/gemma-2b
- google/gemma-7b
- google/gemma-2-2b
- google/gemma-2-9b
- google/gemma-2b-it
- google/gemma-7b-it
- google/gemma-2-2b-it
- google/gemma-2-9b-it
- 01-ai/Yi-6B
- 01-ai/Yi-6B-Chat
- 01-ai/Yi-9B
- 01-ai/Yi-1.5-6B
- 01-ai/Yi-1.5-6B-Chat
- 01-ai/Yi-1.5-9B
- 01-ai/Yi-1.5-9B-Chat

Add *--dataset_param.apply_chat_template* to enable evaluation with chat templates. Other LLMs available on the HuggingFace Hub can also be run using this code.

You can specify the benchmark by *--task_name*. We use the following tasks in our paper:

- cola
- mnli
- qnli
- rte
- mrpc
- qqp
- sst2
- boolq
- wic
- anli_r1
- commonsense_qa
- social_iqa
- winogrande
- openbookqa
- hellaswag
- arc_easy
- arc_challenge
- headqa_en
- mathqa
- medmcqa
- piqa
- sciq
- nq_open
- gsm8k

For more supported benchmarks, please check [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9/lm_eval/tasks). To determine if a benchmark has a training set, check the corresponding *.yml file for the presence of the keyword *training_split*.
