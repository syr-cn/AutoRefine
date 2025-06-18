import json
import concurrent.futures
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import os

from langchain_openai import AzureChatOpenAI
from langchain.schema.messages import HumanMessage
import re

INPUT_PATH = "PATH-TO-RESULT"
OUTPUT_PATH = INPUT_PATH.replace(".jsonl", "-score.jsonl")
MODEL_NAME = "gpt-4o-mini"
API_VERSION = '2024-03-01-preview'
MAX_WORKERS = 5
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_END_POINT = os.getenv("OPENAI_API_ENDPOINT")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

llm = AzureChatOpenAI(
    model=MODEL_NAME,
    azure_endpoint=OPENAI_END_POINT,
    api_version=API_VERSION,
    api_key=OPENAI_API_KEY,
    temperature=0.2,
    max_retries=3,
)

def extract_answer(text):
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else ""

def extract_question(text):
    matches = re.findall(r"Question: (.*?)\n", text, re.DOTALL | re.IGNORECASE)
    return matches[0].strip() if matches else ""

def score_item(item):
    response = item["response"]
    gt_answers = json.dumps(item["ground_truth"])

    # Extract answer from the last <answer> block if present
    extracted_answer = extract_answer(response)
    if not extracted_answer:
        extracted_answer = response
    extracted_question = extract_question(response)

    # Prompt: ask the LLM if the answer is correct
    prompt = (
        f"You are given a question interaction and a ground truth answer.\n"
        f"Determine whether the ground truth can be infered from LLM's final answer.\n\n"
        f"--- Question ---\n{extracted_question}\n\n"
        f"--- LLM's prediction ---\n{extracted_answer}\n\n"
        f"--- GROUND TRUTH ---\n{gt_answers}\n\n"
        f"Only answer 'yes' or 'no'."
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        reply = response.content.strip().lower()
        score = reply.startswith("yes")
    except Exception as e:
        print(f"Error: {e}")
        score = False
    
    item["llm_score"] = score
    item['prompt_tokens'] = response.response_metadata['token_usage']['prompt_tokens']
    item['completion_tokens'] = response.response_metadata['token_usage']['completion_tokens']
    return item

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(score_item, item) for item in data]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(data)):
        results.append(future.result())

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

acc_by_source = defaultdict(lambda: {"correct": 0, "total": 0})
for item in results:
    src = item["data_source"]
    acc_by_source[src]["total"] += 1
    if item.get("llm_score"):
        acc_by_source[src]["correct"] += 1

for src, stats in acc_by_source.items():
    acc = stats["correct"] / stats["total"]
    print(f"{src}: {acc:.2%} ({stats['correct']}/{stats['total']})")

overall_correct = sum(stats["correct"] for stats in acc_by_source.values())
overall_total = sum(stats["total"] for stats in acc_by_source.values())
overall_prompt_tokens = sum(item.get('prompt_tokens', 0) for item in results)
overall_completion_tokens = sum(item.get('completion_tokens', 0) for item in results)
print(f'Overall prompt tokens: {overall_prompt_tokens:,}')
print(f'Overall completion tokens: {overall_completion_tokens:,}')
overall_acc = overall_correct / overall_total if overall_total > 0 else 0
print(f'Overall accuracy: {overall_acc:.2%} ({overall_correct}/{overall_total})')