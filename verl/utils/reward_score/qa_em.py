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
    return final_metric['f1']

def validate_format(prompt, response):
    """
    validate the template format
    return: (is valid)
    """
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

def extract_information(responses_str):
    """Extract and concatenate information from <documents> tags, skipping the first."""
    info_pattern = r'<documents>(.*?)</documents>'
    matches = re.findall(info_pattern, responses_str, re.DOTALL)
    
    if len(matches) <= 1:
        return None
    
    # Concatenate from the second match onward
    combined_info = ' '.join(matches[1:]).strip()
    return combined_info

def extract_information_list(responses_str):
    """Extract and concatenate information from <documents> tags, skipping the first."""
    info_pattern = r'<documents>(.*?)</documents>'
    matches = re.findall(info_pattern, responses_str, re.DOTALL)
    
    if len(matches) <= 1:
        return None
    matches = matches[1:]
    return matches

def extract_refine(responses_str):
    info_pattern = r'<refine>(.*?)</refine>'
    matches = re.findall(info_pattern, responses_str, re.DOTALL)
    
    if len(matches) == 0:
        return None
    
    # Concatenate from the second match onward
    combined_info = ' '.join(matches).strip()
    return combined_info

def extract_solution(responses_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, responses_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 0:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def compute_score_format(responses_str, ground_truth):
    format_validity = validate_format(responses_str, responses_str)
    return format_validity

def compute_reward(solution_str, responses_str, ground_truth, format_score=0., score=1., refine_score=0.0, do_print_frac=-1, score_func=em_check):
    answer = extract_solution(responses_str)
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
        answer_score = score_func(answer, ground_truth['target'])
        format_validity = validate_format(solution_str, responses_str)
        refine_subem = compute_refine_score_subem(responses_str, ground_truth)

        if answer_score > 0:
            return answer_score
        else:
            score = 0.0
            if format_validity:
                score += format_score
            if refine_subem > 0:
                score += refine_score
            return score

def compute_score_em(responses_str, ground_truth):
    answer = extract_solution(responses_str)
    if answer is None:
        return 0
    else:
        return em_check(answer, ground_truth['target'])

def compute_score_f1(responses_str, ground_truth):
    answer = extract_solution(responses_str)
    if answer is None:
        return 0
    else:
        return compute_f1_scores(answer, ground_truth['target'])


def compute_score_cem(responses_str, ground_truth):
    answer = extract_solution(responses_str)
    if answer is None:
        return 0
    else:
        return cover_em_check(answer, ground_truth['target'])


def compute_information_score_subem(responses_str, ground_truth):
    information = extract_information(responses_str)
    
    if information is None:
        return 0.0
    elif 'no' in ground_truth['target'] or 'yes' in ground_truth['target']:
        return 0.5
    else:
        return cover_em_check(information, ground_truth['target'])

def compute_information_reverse_rank(responses_str, ground_truth):
    doc_list = extract_information_list(responses_str)
    info_score = 0.0
    
    if doc_list is None:
        return 0.0
    elif 'no' in ground_truth['target'] or 'yes' in ground_truth['target']:
        return 0.5
    else:
        for idx, doc in enumerate(doc_list):
            if cover_em_check(doc, ground_truth['target']):
                info_score += 1 / float(idx + 1)
    return info_score

def compute_refine_score_subem(responses_str, ground_truth):
    refined_info = extract_refine(responses_str)
    if refined_info is None:
        return 0.0
    else:
        return cover_em_check(refined_info, ground_truth['target'])
