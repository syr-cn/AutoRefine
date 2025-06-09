# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1_scores(prediction: str, ground_truths: list):
    final_metric = {"f1": 0, "precision": 0, "recall": 0}
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        if (
            normalized_ground_truth in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth
        ):
            continue
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        for k in ["f1", "precision", "recall"]:
            final_metric[k] = max(eval(k), final_metric[k])
    return final_metric

def validate_format(text: str):
    """
    validate the template format
    return: (is valid)
    """
    # extract all assistant responses
    assert '<|im_start|>assistant' in text
    prompt, response = text.split("<|im_start|>assistant", 1)
    if '<refine>' in prompt:
        token_list = ['think', 'search', 'refine', 'answer']
    else:
        token_list = ['think', 'search', 'answer']

    if not response:
        return 0

    for special_tags in token_list:
        start_token = f"<{special_tags}>"
        end_token = f"</{special_tags}>"
        start_count = response.count(start_token)
        end_count = response.count(end_token)
        if start_count != end_count:
            return 0
        if start_count == 0:
            return 0
    return 1

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def cover_em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

def extract_information(solution_str):
    """Extract and concatenate information from <documents> tags, skipping the first."""
    info_pattern = r'<documents>(.*?)</documents>'
    matches = re.findall(info_pattern, solution_str, re.DOTALL)
    
    if len(matches) <= 1:
        return None
    
    # Concatenate from the second match onward
    combined_info = ' '.join(matches[1:]).strip()
    return combined_info

def extract_information_list(solution_str):
    """Extract and concatenate information from <documents> tags, skipping the first."""
    info_pattern = r'<documents>(.*?)</documents>'
    matches = re.findall(info_pattern, solution_str, re.DOTALL)
    
    if len(matches) <= 1:
        return None
    matches = matches[1:]
    return matches

def extract_refine(solution_str):
    assert '<|im_start|>assistant' in solution_str
    solution_str = solution_str.split('<|im_start|>assistant')[1]
    info_pattern = r'<refine>(.*?)</refine>'
    matches = re.findall(info_pattern, solution_str, re.DOTALL)
    
    if len(matches) == 0:
        return None
    
    # Concatenate from the second match onward
    combined_info = ' '.join(matches).strip()
    return combined_info


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def compute_score_format(solution_str, ground_truth, format_score=0.0):
    format_validity = validate_format(solution_str)
    return format_validity

def compute_score_f1(solution_str, ground_truth, format_score=0.0, refine_score=0.0, do_print_frac=-1):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, do_print_frac) == 1 if do_print_frac > 0 else False
    
    if do_print:
        print(f"--------------Begin Case--------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        print(f"--------------End Case--------------")

    if answer is None:
        return 0
    else:
        f1_score = compute_f1_scores(answer, ground_truth['target'])['f1']
        format_validity = validate_format(solution_str)
        refine_subem = compute_refine_score_subem(solution_str, ground_truth, format_score=False, score=True)

        if f1_score > 0:
            return f1_score
        else:
            score = 0.0
            if format_validity:
                score += format_score
            if refine_subem > 0:
                score += refine_score
            return score

def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1., refine_score=0.0, do_print_frac=-1):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, do_print_frac) == 1 if do_print_frac > 0 else False
    
    if do_print:
        print(f"--------------Begin Case--------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        print(f"--------------End Case--------------")

    if answer is None:
        return 0
    else:
        em_score = em_check(answer, ground_truth['target'])
        format_validity = validate_format(solution_str)
        refine_subem = compute_refine_score_subem(solution_str, ground_truth, format_score=False, score=True)

        if em_score > 0:
            return em_score
        else:
            score = 0.0
            if format_validity:
                score += format_score
            if refine_subem > 0:
                score += refine_score
            return score

def compute_score_cem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return 0
    else:
        if cover_em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def compute_information_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    information = extract_information(solution_str=solution_str)
    
    if information is None:
        return 0.0
    elif 'no' in ground_truth['target'] or 'yes' in ground_truth['target']:
        return 0.5
    else:
        if cover_em_check(information, ground_truth['target']):
            return score
        else:
            return format_score


def compute_information_reverse_rank(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    doc_list = extract_information_list(solution_str=solution_str)
    
    if doc_list is None:
        return 0.0
    elif 'no' in ground_truth['target'] or 'yes' in ground_truth['target']:
        return 0.5
    else:
        for idx, doc in enumerate(doc_list):
            if cover_em_check(doc, ground_truth['target']):
                return score / float(idx + 1)
    return format_score

def compute_refine_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    refined_info = extract_refine(solution_str=solution_str)
    if refined_info is None:
        return 0.0
    else:
        if cover_em_check(refined_info, ground_truth['target']):
            return score
        else:
            return format_score
