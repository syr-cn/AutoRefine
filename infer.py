import transformers
import torch
import requests
import re

question_list = [
    "Who was born first out of Cameron Mitchell (Singer) and Léopold De Saussure?", # Ground Truth: "Léopold De Saussure"
    "The Clavivox was invented by an American composer who was born Harry Warnow in what year?", # Ground Truth: "1908"
    "Which movie did Disney produce first, The Many Adventures of Winnie the Pooh or Ride a Wild Pony?", # Ground Truth: "Ride a Wild Pony"
    "Who is the sibling of the author of Kapalkundala?", # Ground Truth: "Sanjib Chandra" or "Sanjib Chandra Chattopadhyay"
]

# Model ID and device setup
model_id = "yrshi/AutoRefine-Qwen2.5-3B-Base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

curr_eos = [151645, 151643] # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<documents>{search_results}</documents>\n\n'

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


# Initialize the stopping criteria
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])


def run_search(question):
    question = question.strip()
    cnt = 0
    trajectory = []
    
    # Prepare the message
    prompt = f"""You are a helpful assistant excel at answering questions with multi-turn search engine calling. \
    To answer questions, you must first reason through the available information using <think> and </think>. \
    If you identify missing knowledge, you may issue a search request using <search> query </search> at any time. The retrieval system will provide you with the three most relevant documents enclosed in <documents> and </documents>. \
    After each search, you need to summarize and refine the existing documents in <refine> and </refine>. \
    You may send multiple search requests if needed. \
    Once you have sufficient information, provide a concise final answer using <answer> and </answer>. For example, <answer> Donald Trump </answer>. Question: {question}\n"""


    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    print(prompt)
    # Encode the chat-formatted prompt and move it to the correct device
    while True:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            trajectory.append(output_text)
            print(output_text)
            break

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        query_text = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if query_text:
            search_results = search(query_text)
        else:
            search_results = ''

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        cnt += 1
        print(search_text)
        trajectory.append(search_text)
    print(f"Total iterations: {cnt}")
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    answer_match = answer_pattern.search(trajectory[-1])
    if answer_match:
        final_answer = answer_match.group(1).strip()
        print(f"Final answer found: {final_answer}")
    else:
        print("No final answer found in the output.")
        final_answer = "No final answer found."
    return ''.join([text.strip() for text in trajectory]), final_answer

if __name__ == "__main__":
    output_text, final_answer = run_search(question_list[0])
    print(f"Output trajectory: {output_text}")
    print(f"Final answer: {final_answer}")