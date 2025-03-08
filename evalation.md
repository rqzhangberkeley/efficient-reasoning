# Evaluation Results of Models.

## Baselines: Qwen models.
For the performace of Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct on gsm8k, see https://qwenlm.github.io/blog/qwen2.5-llm/.
For the performance of Qwen2.5-Math-1.5B and Qwen2.5-math-7B on gsm8k, and the perfromance of Qwen2.5-Math-1.5/7b-Instruct on gsm8k and AIME2024, see https://qwenlm.github.io/blog/qwen2.5-math/.

### GSM8k-eval
| Models| pass@1| pass@8 | pass@1 reported by Qwen Team |
|:-----------------|:------------------:|:------------------:|:------------------:|
|  Qwen2.5-1.5B-Instruct | 67.5 | 90.1 | 73.2 |
|  Qwen2.5-Math-1.5B | 76.9 | | 76.8 |
|  Qwen2.5-Math-1.5B-Instruct | 84.2 | | 84.8 |
|  Qwen2.5-7B-Instruct |85.0 | | 91.6 |
|  Qwen2.5-Math-7B |77.2 | 96.0| 91.6 |
|  Qwen2.5-Math-7B-Instruct | 93.3 | | 95.2 |

### AIME2024
| Models| pass@1| pass@10| pass@1 reported by Qwen Team |
|:------------------------------|:-----------------------:|:-----------------------:|:-----------------------:|
|  Qwen2.5-1.5B-Instruct | 0.0 | 10.0 | / |
|  Qwen2.5-1.5B-Instruct+Long CoT Example | 3.3 | 16.7 | / |
|  Qwen2.5-Math-1.5B | 6.7 | 26.7 | / |
|  Qwen2.5-Math-1.5B-Instruct | 6.7 | 23.3 | 10.0 |
|  Qwen2.5-7B-Instruct | 6.7 | 10.0 | / |  
|  Qwen2.5-Math-7B | 20.0 | 40.0 | / |
|  Qwen2.5-Math-7B-Instruct | 6.7 | 16.7 | 16.7 |

### MATH500
| Models| pass@1| pass@8 | pass@1 reported by Qwen Team |
|:------------------------------|:-----------------------:|:----------------------------:|:----------------------------:|
|  Qwen2.5-1.5B-Instruct | 51.8 | 76.0 | |
|  Qwen2.5-Math-1.5B | 62.8 | 87.0 | |
|  Qwen2.5-Math-1.5B-Instruct | 72.4| 88.8 | |
|  Qwen2.5-7B-Instruct | |  | |
|  Qwen2.5-Math-7B | |  | |
|  Qwen2.5-Math-7B-Instruct | |  | |

## LIMO
| Models| pass@1| pass@8|
|:------------------------------|:-----------------------:|:-----------------------:|  
|  Qwen2.5-1.5B-Instruct | 5.75| 19.1 |
|  Qwen2.5-Math-1.5B | 20.3 | 49.6 |
|  Qwen2.5-Math-1.5B-Instruct | 29.4 | 51.4 |
|  Qwen2.5-7B-Instruct | 11.8 | 36.0 |
|  Qwen2.5-Math-7B |39.7 | 66.3 |
|  Qwen2.5-Math-7B-Instruct |27.9 | 52.2 |





