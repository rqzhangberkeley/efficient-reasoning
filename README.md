# Sample-Efficient Reinforcement Learning for Mathematical Reasoning

This is the codebase sample-efficient reinforcement learning for mathematical reasoning. We aim to explore some ideas about RL or multi-stage RL on a small dataset and see if we ca outperform SFT.


## Baselines: Qwen models.
We first evaluate some baselines on gsm8k, AIME 2024, MATH500 and LIMO-train (817 samples).
The evaluation codebase is from https://github.com/Zanette-Labs/efficient-reasoning/tree/main and is organized by Daman Arora and Andrea Zanette.
You can also refer to https://github.com/QwenLM/Qwen2.5-Math for the parser in the pipeline.
For the performace of Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct on gsm8k, see https://qwenlm.github.io/blog/qwen2.5-llm/.
For the performance of Qwen2.5-Math-1.5B and Qwen2.5-math-7B on gsm8k, and the perfromance of Qwen2.5-Math-1.5/7b-Instruct on gsm8k and AIME2024, see https://qwenlm.github.io/blog/qwen2.5-math/.

We evaluate some baselines on LIMO dataet (See https://arxiv.org/abs/2502.03387).
We also evaluate the LIMO model on several benchmarks.
Running LIMO model takes much longer time since the model generally outputs longer CoTs. It takes 1.5 hours to run gsm8k-eval on LIMO model via 8 GPUs and batched inference.

### GSM8k-eval
| Models| pass@1| pass@8 | pass@1 reported | Average CoT Length |
|:-----------------|:------------------:|:------------------:|:------------------:|:------------------:|
|  Qwen2.5-1.5B-Instruct | 67.5 | 90.1 | 73.2 | 328 |
|  Qwen2.5-Math-1.5B | 76.9 | | 76.8 | 360 |
|  Qwen2.5-Math-1.5B-Instruct | 84.2 | | 84.8 | 320 |
|  Qwen2.5-7B-Instruct |85.0 | | 91.6 | 479 |   
|  Qwen2.5-Math-7B |77.2 | 96.0| 91.6 | 371 |   
|  Qwen2.5-Math-7B-Instruct | 93.3 | | 95.2 | 394 |         
|  Qwen2.5-32B-Instruct |95.5 | 98.2 | 95.9 | 279 |
|  GAIR/LIMO (32B) | 94.5 | 97.3 | / | 1327 |

### AIME2024
| Models| pass@1| pass@10| pass@1 reported | Average CoT Length |
|:------------------------------|:-----------------------:|:-----------------------:|:-----------------------:|:-----------------------:|
|  Qwen2.5-1.5B-Instruct | 0.0 | 10.0 | / | 2145 |
|  Qwen2.5-1.5B-Instruct+Long CoT Example | 3.3 | 10.0 | / | 4094 |
|  Qwen2.5-Math-1.5B | 6.7 | 26.7 | / | 1314 |
|  Qwen2.5-Math-1.5B-Instruct | 6.7 | 23.3 | 10.0 | 979 |
|  Qwen2.5-7B-Instruct | 6.7 | 10.0 | / | 1962 |
|  Qwen2.5-Math-7B | 20.0 | 40.0 | / | 1357|
|  Qwen2.5-Math-7B-Instruct | 6.7 | 16.7 | 16.7 | 1695 |
|  Qwen2.5-32B-Instruct | 13.3 | 30.0 | / | 1128 |
|  GAIR/LIMO (32B) | 66.7 |86.7 |/ | 13620|

### MATH500
| Models| pass@1| pass@8 | pass@1 reported | Average CoT Length |
|:------------------------------|:-----------------------:|:----------------------------:|:----------------------------:|:----------------------------:|
|  Qwen2.5-1.5B-Instruct | 51.8 | 76.0 | |778 |
|  Qwen2.5-Math-1.5B | 62.8 | 87.0 | |730 |
|  Qwen2.5-Math-1.5B-Instruct | 72.4| 88.8 | | 566|
|  Qwen2.5-7B-Instruct | |  | | |
|  Qwen2.5-Math-7B | |  | | |
|  Qwen2.5-Math-7B-Instruct | |  | | |
|  GAIR/LIMO (32B) |  | | | |

### LIMO
| Models| pass@1| pass@8| Average CoT Length |
|:------------------------------|:-----------------------:|:-----------------------:|:-----------------------:|  
|  Qwen2.5-1.5B-Instruct | 5.75| 19.1 | 1757 |
|  Qwen2.5-Math-1.5B | 20.3 | 49.6 | 1240|
|  Qwen2.5-Math-1.5B-Instruct | 29.4 | 51.4 | 951 |
|  Qwen2.5-7B-Instruct | 11.8 | 36.0 | /|
|  Qwen2.5-Math-7B | 39.7 | 66.3 | 1128 |
|  Qwen2.5-Math-7B-Instruct | 27.9 | 52.2 | 1503 |
|  Qwen2.5-32B-Instruct | 34.1 | 54.2 |1070 |
|  GAIR/LIMO (32B) |  | | | |


## RL
We run GRPO on LIMO-train (817 samples) using Qwen2.5-Math-1.5B model. The codebase is from OpenR1 (https://github.com/huggingface/open-r1).



