
def make_prefix(dp, template_type):
    question = dp['question']
    if template_type == 'autorefine':
        prefix = f"""You are a helpful assistant excel at answering questions with multi-turn search engine calling. \
To answer questions, you must first reason through the available information using <think> and </think>. \
If you identify missing knowledge, you may issue a search request using <search> query </search> at any time. The retrieval system will provide you with the three most relevant documents enclosed in <documents> and </documents>. \
After each search, you need to summarize and refine the existing documents in <refine> and </refine>. \
You may send multiple search requests if needed. \
Once you have sufficient information, provide a concise final answer using <answer> and </answer>. For example, <answer> Donald Trump </answer>. Question: {question}\n"""
    elif template_type == 'searchr1':
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix
