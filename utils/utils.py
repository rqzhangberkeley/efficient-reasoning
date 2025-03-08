from utils.parser import extract_answer
from utils.grader import math_equal

DATASET_KEYS = {
    'openai/gsm8k': {'question': 'question', 'answer': 'answer'},
    'hendrycks/competition_math': {'question': 'problem', 'answer': 'solution'},
    'datasets/converted_aime_dataset': {'question': 'problem', 'answer': 'solution'},
    'di-zhang-fdu/MATH500': {'question': 'problem', 'answer': 'solution'},
    'datasets/compression_dataset': {'question': 'problem', 'answer': 'solution'},
    'GAIR/LIMO': {'question': 'question', 'answer': 'solution'}, # RZ: whether I should use 'solution' or 'answer' here?
}

RESPONSE_EXTRACTOR = {
    'openai/gsm8k': lambda x: extract_answer(x, data_name='gsm8k'),
    'hendrycks/competition_math': lambda x: extract_answer(x, data_name='math'),
    'di-zhang-fdu/MATH500': lambda x: extract_answer(x, data_name='math'),
    'datasets/compression_dataset': lambda x: extract_answer(x, data_name='math'),
    'datasets/converted_aime_dataset': lambda x: extract_answer(x, data_name='math'),
    'GAIR/LIMO': lambda x: extract_answer(x, data_name='math')
}

RESPONSE_COMPARATOR = {
    'openai/gsm8k': lambda x, y: math_equal(x, y, timeout=True),
    'hendrycks/competition_math': lambda x, y: math_equal(x, y, timeout=True),
    'di-zhang-fdu/MATH500': lambda x, y: math_equal(x, y, timeout=True),
    'datasets/compression_dataset': lambda x, y: math_equal(x, y, timeout=True),
    'datasets/converted_aime_dataset': lambda x, y: math_equal(x, y, timeout=True),
    'GAIR/LIMO': lambda x, y: math_equal(x, y, timeout=True)
}
