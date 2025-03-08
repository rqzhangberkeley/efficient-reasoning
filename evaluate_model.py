import os, time
import json
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
from utils import DATASET_KEYS, RESPONSE_EXTRACTOR, RESPONSE_COMPARATOR
import pandas as pd
import argparse
import numpy as np

def get_scores(ds,
            outputs, 
            save_file_name=None,
            answer_key=None,
            question_key=None,
            response_extractor=None,
            response_comparator=None
            ):

    predictions, golds = [], []
    results = []
    for input, output in zip(ds, outputs): # output is a list of vllm.generate.GenerateResult objects. There can be multiple outputs per input in the output list.
        gold = response_extractor(input[answer_key])
        prediction = [
            response_extractor(resp.text)
            for resp in output.outputs
        ]
        predictions.append(prediction)
        golds.append(gold)
        results.append(
            {
                question_key: input[question_key],
                answer_key: input[answer_key],
                "responses": [resp.text for resp in output.outputs],
                "prediction": prediction,
                "gold": gold,
                "tokens": sum([len(resp.token_ids) for resp in output.outputs]) / len(output.outputs),
                "accuracy": [response_comparator(gold, pred) for pred in prediction],
            }
        )
    if save_file_name is not None: # save.
        with open(save_file_name, 'w') as f:
            json.dump(results, f, indent=4)

    results = pd.DataFrame(results)
    predictions, golds, tokens = results["prediction"], results["gold"], results["tokens"]
    pass_at_1 = sum([any([response_comparator(g, pred) for pred in p[:1]]) for p, g in zip(predictions, golds)]) / len(predictions) # pass@1
    pass_at_k_list = []
    acc_at_k_list = []
    k = TEST_N
    print("Average tokens:", sum(tokens) / len(tokens))

    # pass@k
    for i in range(k):
        pass_at_i = sum([any([response_comparator(g, pred) for pred in p[:i+1]]) for p, g in zip(predictions, golds)]) / len(predictions)
        acc_at_i = sum([response_comparator(g, p[i]) for p, g in zip(predictions, golds)]) / len(predictions)
        acc_at_k_list.append(acc_at_i)
        pass_at_k_list.append(pass_at_i)
        print(
            f"Pass @ {i+1}: {pass_at_i}"
        )

    # determine the most common answer. Compute pass@1(majority)
    def get_most_common(solns):
        soln_counts = {}
        for soln in solns:
            if soln is None:
                continue
            added = False
            for other_solns in solns:
                if response_comparator(soln, other_solns):
                    added = True
                    soln_counts[soln] = soln_counts.get(soln, 0) + 1
            if not added:
                soln_counts[soln] = 1
        if len(soln_counts) == 0:
            return None
        return max(soln_counts, key=soln_counts.get)
    
    predictions_maj = [get_most_common(p) for p in predictions]
    all_preds = sum([[response_comparator(golds[i], p) for p in predictions[i]] for i in range(len(predictions))], [])
    avg_pass_rate = sum(all_preds) / len(all_preds)
    pass_at_n = sum([response_comparator(g, p) for p, g in zip(predictions_maj, golds)]) / len(predictions)
    print(
        f"Pass @ 1(with majority): {pass_at_n}"
    )
    
    return {
        'pass@1': pass_at_1,
        'pass@1(majority)': sum([response_comparator(g, p) for p, g in zip(predictions_maj, golds)]) / len(predictions),
        'average_pass_rate': avg_pass_rate,
        'std_pass_rate': np.std(acc_at_k_list),
        'acc@k': acc_at_k_list,
        'pass@k': pass_at_k_list,
        'avg_tokens': sum(tokens) / len(tokens)
    }

def evaluate_model(model_name,
                    dataset,
                    dataset_name,
                    dataset_split,
                    answer_key,
                    question_key,
                    response_extractor,
                    response_comparator,
                    tok_limit,
                    test_n,
                    max_test_samples,
                    test_temperature,
                    example_prompt=None,
                    example_solution=None,
                    n_gpus=1):
    test_prompts = []
    model = LLM(model_name, 
                tokenizer=f'{model_name}', 
                gpu_memory_utilization=0.9, tensor_parallel_size=n_gpus,
                dtype='bfloat16',
                trust_remote_code=True
                )   
    test_ds = dataset[dataset_split].shuffle(seed=0).select(range(min(max_test_samples, len(dataset[dataset_split]))))
    
    for x in test_ds:
        if example_prompt and example_solution:
            prompt = [{
                "role": "user",
                "content": f"""Please reason step by step, and put your final answer within \\boxed{{}}.

            Here is an example:
            Question: {example_prompt}

            Solution:
            {example_solution}

            Now solve this question:
            {x[question_key]}"""
            }]

        else:
            prompt = [{
                "role": "user",
                "content": f"Please reason step by step, and put your final answer within \\boxed{{}}. Question: {x[question_key]}",
            }]

        prompt_tokens = model.llm_engine.tokenizer.tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        test_prompts.append(prompt_tokens)
    
    sampling_params = SamplingParams(
        temperature=test_temperature,
        max_tokens=tok_limit,
        n=test_n
    )
    if example_prompt is None:
        save_file_name = f"outputs/{dataset_name.replace('/', '_')}_results_{model_name.replace('/', '_')}_{tok_limit}.json"
    elif isinstance(example_prompt,str):
        save_file_name = f"outputs/{dataset_name.replace('/', '_')}_results_{model_name.replace('/', '_')}_{tok_limit}_1shot.json"
    else:
        raise NotImplementedError
    
    sampling_params.stop_token_ids = [model.llm_engine.tokenizer.tokenizer.eos_token_id]
    print("Generating test outputs...")
    print(model.llm_engine.tokenizer.tokenizer.decode(test_prompts[0], skip_special_tokens=False))
    start_time = time.time()
    test_outputs = model.generate(prompt_token_ids=test_prompts, sampling_params=sampling_params, use_tqdm=True) # generate outputs
    end_time = time.time()
    test_scores = get_scores(test_ds, 
                            test_outputs, 
                            save_file_name,
                            answer_key,
                            question_key,
                            response_extractor,
                            response_comparator) # get scores
    print("Test:", test_scores)
    time_taken = end_time - start_time
    print("Time taken:", time_taken)

    return {'test': test_scores, 'time_taken': time_taken}


# This script evaluates a model on a dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_paths', type=str, nargs='+', default=[])
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--scale', type=str, default='auto')
    parser.add_argument('--tok_limit', type=int, default=32768)
    parser.add_argument('--n_gpus',type=int,default=1)
    args = parser.parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    os.environ["VLLM_ATTENTION_ENGINE"] = "flash-v2"

    dataset_name = args.dataset
    model_paths = args.model_paths
    scale = args.scale
    tok_limit = args.tok_limit
    dataset_name = args.dataset
    n_gpus=args.n_gpus
    model_scales = []

    if scale == 'auto':
        for model_path in model_paths:
            if not any(size in model_path for size in ['1.5B', '7B', '14B', '32B']):
                raise ValueError(f"Model path {model_path} must contain a param size, e.g. '1.5B' or '7B' or '14B' or '32B'")
            for size in ['1.5B', '7B', '14B', '32B']:
                if size in model_path:
                    model_scales.append(size)
                    continue
    else:
        model_scales = [scale]

    print("Dataset:", dataset_name)
    QUESTION_KEY = DATASET_KEYS[dataset_name]["question"]
    ANSWER_KEY = DATASET_KEYS[dataset_name]["answer"]
    eq = RESPONSE_COMPARATOR[dataset_name]

    if dataset_name == 'datasets/converted_aime_dataset':
        dataset = load_from_disk(dataset_name)
        TEST_N = 10
        MAX_TOKENS = tok_limit
        TEST_TEMPERATURE = 0.6
        MAX_TEST_SAMPLES = 100
        DATASET_SPLIT = 'test'
    elif dataset_name == 'di-zhang-fdu/MATH500':
        dataset = load_dataset(dataset_name)
        TEST_N = 8
        MAX_TOKENS = tok_limit
        TEST_TEMPERATURE = 0.6
        MAX_TEST_SAMPLES = 500
        DATASET_SPLIT = 'test'
    elif dataset_name == 'openai/gsm8k': # RZ: Let's first try gsm8k
        dataset = load_dataset(dataset_name, 'main')
        TEST_N = 8
        MAX_TOKENS = tok_limit
        TEST_TEMPERATURE = 0.6
        MAX_TEST_SAMPLES = 1319 # RZ: size of gsm8k eval.
        DATASET_SPLIT = 'test'
    elif dataset_name == 'GAIR/LIMO':
        dataset = load_dataset(dataset_name)
        TEST_N = 8
        MAX_TOKENS = tok_limit
        TEST_TEMPERATURE = 0.6
        MAX_TEST_SAMPLES = 817
        DATASET_SPLIT = 'train'

    example_prompt = None
    # "Let $S$ be a set with six elements . Let $\\mathcal{P}$ be the set of all subsets of $S.$ Subsets $A$ and $B$ of $S$ , not necessarily distinct, are chosen independently and at random from $\\mathcal{P}$ . The probability that $B$ is contained in one of $A$ or $S-A$ is $\\frac{m}{n^{r}},$ where $m$ , $n$ , and $r$ are positive integers , $n$ is prime , and $m$ and $n$ are relatively prime . Find $m+n+r.$ (The set $S-A$ is the set of all elements of $S$ which are not in $A.$ )"

    example_solution = None
    # "Okay, let's tackle this problem step by step. So, we have a set S with six elements. The power set P(S) consists of all subsets of S. We're choosing two subsets A and B independently and at random from P(S). We need to find the probability that B is contained in either A or S - A. The answer should be in the form m/n^r where m and n are coprime, n is prime, and then find m + n + r.\n\nFirst, let me make sure I understand the problem. We have two random subsets A and B. Each subset is chosen uniformly at random from all possible subsets of S. The selection is independent, so the choice of A doesn't affect the choice of B and vice versa. We need the probability that B is a subset of A or B is a subset of S - A. That is, B is entirely contained in A or entirely contained in the complement of A.\n\nGiven that S has six elements, each element can be thought of as being in or out of a subset. For each element, there are two possibilities, so the total number of subsets is 2^6 = 64. Therefore, both A and B are chosen from 64 subsets, each with equal probability. Since they are chosen independently, the total number of possible pairs (A, B) is 64 * 64 = 4096. So, the probability we need is the number of favorable pairs (A, B) where B is a subset of A or B is a subset of S - A, divided by 4096.\n\nBut maybe instead of counting pairs, we can model this probabilistically. For each element in S, perhaps we can think about the choices for A and B in terms of their membership for that element.\n\nLet me think. Since A and B are subsets, each element can be in A or not, in B or not. Since they're independent, the choices for A and B are independent for each element. Wait, but subsets are not independent across elements. Wait, actually, when choosing a subset uniformly at random, each element is included independently with probability 1/2. So, perhaps the selection of A and B can be thought of as each element has four possibilities: in A and in B, in A but not in B, not in A but in B, or not in either. Each with probability 1/4, since each choice for A and B is independent.\n\nBut maybe there's a better way. Let me consider each element in S. For B to be a subset of A or a subset of S - A, each element in B must be either entirely in A or entirely not in A. So, for each element in B, if that element is in B, then it must be in A (if we're considering B subset of A) or not in A (if we're considering B subset of S - A). Wait, but B could have some elements in A and some not in A, but for B to be a subset of either A or S - A, all elements of B must be in A or all must be in S - A.\n\nTherefore, for B to be contained in A or in S - A, the entire set B must be a subset of A or a subset of S - A. So, for each element in B, if B is a subset of A, then all elements of B are in A; if B is a subset of S - A, then all elements of B are not in A. Therefore, for each element in B, the corresponding element in A must be either all 1s (if B is in A) or all 0s (if B is in S - A). However, since B is a random subset, the elements in B are also random. Hmm, this might complicate things.\n\nAlternatively, maybe we can fix B and calculate the probability over A, then average over all B. But since A and B are chosen independently and uniformly, maybe it's easier to fix A and B and analyze their relations.\n\nWait, but since they are independent, perhaps the probability can be written as the expectation over all possible B of the probability that B is a subset of A or S - A, where A is random. Or vice versa. Let's try this approach.\n\nLet me fix B and compute the probability over A. For a fixed B, the probability that B is a subset of A or a subset of S - A. Then, since A is chosen uniformly at random, the probability that B is a subset of A is (1/2)^{|B|}, because each element in B must be in A, and the other elements can be arbitrary. Similarly, the probability that B is a subset of S - A is (1/2)^{|B|}, because each element in B must not be in A. However, these two events are not mutually exclusive. If B is the empty set, then B is a subset of both A and S - A. Therefore, we need to subtract the overlap.\n\nSo, for a fixed B, the probability that B is a subset of A or a subset of S - A is (1/2)^{|B|} + (1/2)^{|B|} - (1/2)^{2|B|}. Wait, why 2|B|? Wait, the overlap is the probability that B is a subset of both A and S - A. But if B is a subset of A and a subset of S - A, then B must be a subset of A \u2229 (S - A) = empty set. Therefore, B must be the empty set. So, the overlap is non-zero only if B is empty. Therefore, the overlap is (1/2)^{|B|} if B is empty, otherwise zero. Wait, that's confusing.\n\nWait, let's think again. For a fixed B, the probability that B is a subset of A is (1/2)^{|B|}, as each element of B must be in A. Similarly, the probability that B is a subset of S - A is also (1/2)^{|B|}, as each element of B must not be in A. The intersection of these two events is the event that B is a subset of both A and S - A. But the intersection of A and S - A is empty, so the only way B is a subset of both is if B is empty. So, if B is non-empty, then the intersection is empty. If B is empty, then both events occur with probability 1. So, the probability of the intersection is 1 if B is empty, and 0 otherwise.\n\nTherefore, by inclusion-exclusion, the probability that B is a subset of A or S - A is:\n\nIf B is empty: 1 + 1 - 1 = 1.\n\nIf B is non-empty: (1/2)^{|B|} + (1/2)^{|B|} - 0 = 2*(1/2)^{|B|} = (1/2)^{|B| - 1}.\n\nTherefore, the probability for each fixed B is:\n\n- 1 if B is empty.\n\n- (1/2)^{|B| - 1} if B is non-empty.\n\nTherefore, since we need to compute the average over all B of this probability (since A and B are chosen independently and uniformly), we can compute the expectation over B of this probability.\n\nWait, no. Wait, the problem states that A and B are chosen independently and uniformly at random. Therefore, the total probability is the average over all A and B of the indicator that B is a subset of A or S - A.\n\nAlternatively, since for each B, the probability that B is a subset of A or S - A is as above, then the total probability is the average of this quantity over all B.\n\nTherefore, the total probability is:\n\n(1 / |P|) * [Probability when B is empty + Sum over non-empty B of Probability for each B]\n\nSince there is 1 empty set, and 63 non-empty subsets.\n\nSo, the total probability is (1/64)[1 + Sum_{k=1}^6 (number of subsets of size k) * (1/2)^{k - 1} } ]\n\nBecause for each non-empty B, the probability is (1/2)^{|B| - 1}, and the number of subsets of size k is C(6, k).\n\nTherefore, let's compute this:\n\nTotal probability = (1/64)[1 + Sum_{k=1}^6 C(6, k) * (1/2)^{k - 1} } ]\n\nCompute the sum:\n\nSum_{k=1}^6 C(6, k) * (1/2)^{k - 1}\n\nLet me factor out a 2 to make it Sum_{k=1}^6 C(6, k) * (1/2)^{k - 1} = 2 * Sum_{k=1}^6 C(6, k) * (1/2)^k\n\nBut Sum_{k=0}^6 C(6, k) * (1/2)^k = (1 + 1/2)^6 = (3/2)^6. So, Sum_{k=1}^6 C(6, k)*(1/2)^k = (3/2)^6 - 1.\n\nTherefore, the sum we have is 2*[(3/2)^6 - 1]\n\nTherefore, total probability is (1/64)[1 + 2*((3/2)^6 - 1)] = (1/64)[1 + 2*(3/2)^6 - 2] = (1/64)[2*(3/2)^6 - 1]\n\nCompute 2*(3/2)^6:\n\nFirst, (3/2)^6 = (729)/(64). Therefore, 2*(729/64) = 1458/64 = 729/32.\n\nThen subtract 1: 729/32 - 1 = (729 - 32)/32 = 697/32.\n\nTherefore, total probability is (1/64)*(697/32) = 697/(64*32) = 697/2048.\n\nWait, 64*32 is 2048. Let me check that. 64*32 is indeed 2048. Then 697/2048.\n\nNow, we need to check if 697 and 2048 are coprime. Since 2048 is a power of 2, we need to check if 697 is odd, which it is. So, divide 697 by primes. 697 divided by 17 is 41, since 17*41 = 697. So, 697 = 17*41. 2048 is 2^11. So, since 697 and 2048 share no common factors, the fraction is reduced.\n\nThus, the probability is 697/2048. Therefore, m = 697, n = 2, r = 11. Then m + n + r = 697 + 2 + 11 = 710.\n\nWait, but wait, hold on. Let me double-check my steps because the answer seems a bit large, and I might have messed up somewhere.\n\nFirst, the logic was:\n\nFor each fixed B, the probability over A that B is a subset of A or S - A is:\n\n- 1 if B is empty.\n\n- (1/2)^{|B| - 1} if B is non-empty.\n\nTherefore, the total probability is average over all B of this probability. Since there are 64 subsets B, so we have:\n\n[1 + sum_{B non-empty} (1/2)^{|B| - 1}]/64\n\nThen, we calculated sum_{B non-empty} (1/2)^{|B| - 1} as sum_{k=1}^6 C(6, k)*(1/2)^{k - 1} = 2*[ (3/2)^6 - 1 ]\n\nWait, let's verify that step again.\n\nSum_{k=1}^6 C(6, k)*(1/2)^{k - 1} = Sum_{k=1}^6 C(6, k)*(1/2)^{k - 1}\n\n= Sum_{k=1}^6 C(6, k)*(1/2)^{k} * 2\n\n= 2 * Sum_{k=1}^6 C(6, k)*(1/2)^k\n\n= 2 * [ Sum_{k=0}^6 C(6, k)*(1/2)^k - C(6, 0)*(1/2)^0 ]\n\n= 2 * [ (1 + 1/2)^6 - 1 ]\n\n= 2 * [ (3/2)^6 - 1 ]\n\nYes, that seems right. Then, (3/2)^6 is 729/64, so:\n\n2*(729/64 - 1) = 2*(729/64 - 64/64) = 2*(665/64) = 1330/64 = 665/32\n\nWait, hold on! Wait, I think I miscalculated here. Wait, 729 - 64 is 665, not 729 - 32. Wait, no, original expression is 2*((3/2)^6 - 1) = 2*(729/64 - 1). So 729/64 - 1 is (729 - 64)/64 = 665/64. Then times 2 is 1330/64 = 665/32.\n\nBut in my original calculation, I wrote:\n\nSum_{k=1}^6 C(6, k)*(1/2)^{k - 1} = 2*((3/2)^6 - 1) = 2*(729/64 - 1) = 2*(665/64) = 1330/64 = 665/32.\n\nBut then earlier, I thought that the total probability is (1/64)[1 + 665/32]. Let's compute that:\n\n1 + 665/32 = (32/32) + 665/32 = 697/32. Then, (1/64)*(697/32) = 697/(64*32) = 697/2048. That's correct.\n\nBut 697 is equal to 17*41, and 2048 is 2^11. So, yes, 697 and 2048 are coprime. Therefore, m = 697, n = 2, r = 11. Therefore, m + n + r = 697 + 2 + 11 = 710.\n\nBut wait, the problem statement says the answer is in the form m/n^r where n is prime. Here, n = 2, which is prime, and m and n are coprime. So, the answer is 697/2^11, which gives 710.\n\nBut let me check with another approach to see if this is correct.\n\nAlternative approach:\n\nEach element in S can be categorized based on whether it's in A, B, both, or neither. Since A and B are independent, each element has four possibilities:\n\n1. In A and in B.\n\n2. In A but not in B.\n\n3. Not in A but in B.\n\n4. Neither in A nor in B.\n\nEach with probability 1/4, since each choice for A and B is independent.\n\nNow, for B to be a subset of A or S - A, we need that for each element in B, if it's in B, then it must be in A (if B is a subset of A) or it must not be in A (if B is a subset of S - A). But since B is a subset, all elements of B must satisfy the same condition. So, either all elements in B are in A, or all elements in B are not in A.\n\nTherefore, for each element in B, we have constraints. But since B is a random subset, each element is in B with probability 1/2, independently.\n\nWait, but actually, when we choose B uniformly at random, each element is included in B with probability 1/2, independent of others. Similarly for A. So perhaps we can model this per element.\n\nLet me think in terms of each element. For each element, there are four possibilities as before. The condition that B is a subset of A or S - A translates to, for all elements in B, they are all in A or all not in A. So, for each element, if the element is in B, then either it's in A or not in A, but this choice has to be consistent across all elements in B.\n\nHmm, this seems tricky because the choice for different elements are dependent. If B has multiple elements, then the events that each element in B is in A or not in A are not independent.\n\nAlternatively, perhaps the probability can be computed as follows. For each element, the probability that either the element is not in B, or if it is in B, then it is in A, or if it is in B, then it is not in A. Wait, but the \"or\" here is tricky because it's an exclusive or. Wait, no. The condition is that all elements in B are in A, or all elements in B are not in A. So, for each element, if it is in B, then the same condition must hold for all elements in B.\n\nAlternatively, the probability that B is a subset of A is equal to the probability that all elements in B are in A. Similarly, the probability that B is a subset of S - A is the probability that all elements in B are not in A. But since A and B are independent, the probability that all elements in B are in A is (1/2)^{|B|}, and similarly for the complement.\n\nBut since B is a random variable, we need to take the expectation over B.\n\nWait, this seems similar to my first approach. So, the expected probability would be E_B [ (1/2)^{|B|} + (1/2)^{|B|} - (1/2)^{2|B|} ] when B is empty, otherwise subtract zero. But in expectation, this would be:\n\nSum_{k=0}^6 C(6, k) * [ (1/2)^k + (1/2)^k - (1/2)^{2k} ] * (1/2)^6\n\nWait, no. Wait, the probability is over both A and B. Since A and B are independent, the joint probability is uniform over all pairs (A, B). Therefore, the total number of pairs is 2^6 * 2^6 = 2^12 = 4096. So, perhaps another way to compute this is to consider for each element, the contribution to the probability.\n\nLet me consider each element. For B to be a subset of A or S - A, each element in B must be entirely in A or entirely in S - A. So, for each element, if the element is in B, then it must be in A or not in A, but consistently across all elements in B.\n\nAlternatively, for each element, the possible cases:\n\n1. The element is not in B: Then, no constraint on A.\n\n2. The element is in B: Then, either it is in A (if B is a subset of A) or not in A (if B is a subset of S - A). But since all elements in B must be in A or all not in A, we need to have consistency across all elements in B.\n\nThis seems complicated, but perhaps we can model the probability by considering each element and the constraints they impose.\n\nWait, perhaps using linearity of expectation is not directly applicable here because the events are not independent. However, maybe inclusion-exclusion could be used.\n\nAlternatively, think of it as follows. Let's first choose B, then A. For a given B, the number of A such that B is a subset of A or B is a subset of S - A is equal to 2^{6 - |B|} + 2^{6 - |B|} - 1. Wait, no. If B is a subset of A, then the other 6 - |B| elements can be arbitrary, so there are 2^{6 - |B|} subsets A containing B. Similarly, subsets A such that B is a subset of S - A, which means A is a subset of S - B. The number of such subsets A is 2^{6 - |B|}. However, the overlap between these two sets is the subsets A that contain B and are subsets of S - B. But B and S - B are disjoint, so the only subset A that contains B and is a subset of S - B is if B is empty. So, if B is non-empty, the overlap is zero. If B is empty, both conditions are equivalent to A being any subset, so overlap is 2^6. Therefore, for each B, the number of A such that B is subset of A or S - A is:\n\nIf B is empty: 2^6 + 2^6 - 2^6 = 2^6 = 64.\n\nIf B is non-empty: 2^{6 - |B|} + 2^{6 - |B|} - 0 = 2^{7 - |B|}\n\nTherefore, the total number of favorable pairs (A, B) is:\n\nSum_{B} [ if B is empty: 64; else: 2^{7 - |B|} ]\n\nSo, the total is 64 + Sum_{B non-empty} 2^{7 - |B|}\n\nBut Sum_{B non-empty} 2^{7 - |B|} = Sum_{k=1}^6 C(6, k) * 2^{7 - k}\n\nLet me compute this sum:\n\nSum_{k=1}^6 C(6, k) * 2^{7 - k} = 2^7 * Sum_{k=1}^6 C(6, k) * (1/2)^k\n\n= 128 * [ Sum_{k=0}^6 C(6, k)*(1/2)^k - C(6, 0)*(1/2)^0 ]\n\n= 128 * [ (1 + 1/2)^6 - 1 ]\n\n= 128 * [ (3/2)^6 - 1 ]\n\nCompute (3/2)^6 = 729 / 64. Then,\n\n128 * (729/64 - 1) = 128 * (729/64 - 64/64) = 128 * (665/64) = 2 * 665 = 1330\n\nTherefore, total number of favorable pairs is 64 + 1330 = 1394\n\nTherefore, the probability is 1394 / 4096. Simplify this fraction.\n\nDivide numerator and denominator by 2: 697 / 2048.\n\nWhich matches the previous result. 697 and 2048 are coprime as 697 factors into 17*41 and 2048 is 2^11. Therefore, the probability is 697/2048, so m = 697, n = 2, r = 11. Then, m + n + r = 697 + 2 + 11 = 710.\n\nTherefore, the answer is 710.\n\n**Final Answer**\n\\boxed{710}"

    for model_path, scale in zip(model_paths, model_scales):
        results = {}
        print("Found model_path:", model_path)
        print("This is not a checkpoint, will evaluate directly...")
        scores = evaluate_model(model_path, 
                                dataset, 
                                dataset_name,
                                DATASET_SPLIT,
                                ANSWER_KEY, 
                                QUESTION_KEY, 
                                RESPONSE_EXTRACTOR[dataset_name],
                                eq, 
                                tok_limit, 
                                TEST_N, 
                                MAX_TEST_SAMPLES, 
                                TEST_TEMPERATURE,
                                example_prompt=example_prompt,
                                example_solution=example_solution,
                                n_gpus=n_gpus
                            )
        results[model_path] = scores

        if example_prompt is None:
            results_file = f'results/{dataset_name.replace("/", "_")}_results_{model_path.replace("/", "_")}_{tok_limit}.json'
        elif isinstance(example_prompt,str):
            results_file = f'results/{dataset_name.replace("/", "_")}_results_{model_path.replace("/", "_")}_{tok_limit}_1shot_singleturn.json'
        else:
            raise NotImplementedError

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Finished evaluating {model_path} on {dataset_name} with {scale}.")
