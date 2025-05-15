
def make_prefix(dp, template_type):
    question = dp['question']
    if template_type == 'autorefine':
        prefix = f"""You are a helpful assistant excel at answering questions with multi-turn search engine calling. \
To answer questions, you must first reason through the available information using <think> and </think>. \
If you identify missing knowledge, you may issue a search request using <search> query </search> at any time. The retrieval system will provide you with the three most relevant documents enclosed in <documents> and </documents>. \
After each search, you need to summarize and refine the existing documents in <refine> and </refine>. \
You may send multiple search requests if needed. \
Once you have sufficient information, provide a concise final answer using <answer> and </answer>. For example, <answer> Donald Trump </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix