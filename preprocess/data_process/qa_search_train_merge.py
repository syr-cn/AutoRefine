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
"""
Preprocess the QA dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse
from utils import make_prefix
import requests


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--data_sources', default='nq')

    args = parser.parse_args()

    # data_source = 'nq'
    data_sources = args.data_sources.split(',')
    all_dataset = []
    cache_file_path = './data/search.json'
    if not os.path.exists(cache_file_path):
        cache_data = {}
    else:
        with open(cache_file_path, 'r') as f:
            cache_data = json.load(f)

    for data_source in data_sources:

        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)

        train_dataset = dataset['train']
        # train_dataset = train_dataset.shuffle(seed=42).select(range(100))

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                example['question'] = example['question'].strip()
                if example['question'][-1] != '?':
                    example['question'] += '?'
                question = make_prefix(example, template_type=args.template_type)
                solution = {
                    "target": example['golden_answers'],
                }

                str_question = example['question']
                if str_question in cache_data:
                    doc_str = cache_data[str_question]
                else:
                    doc_str = search(str_question)
                    cache_data[str_question] = doc_str
                    if random.randint(0, 100) ==0: # save every 100 times
                        with open(cache_file_path, 'w') as f:
                            json.dump(cache_data, f)
                data = {
                    "data_source": data_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                return data

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        all_dataset.append(train_dataset)
        with open(cache_file_path, 'w') as f:
            json.dump(cache_data, f)
    

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_train_dataset = datasets.concatenate_datasets(all_dataset)
    all_train_dataset = all_train_dataset.shuffle(seed=42)

    print(all_train_dataset[0])
    all_train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    assert hdfs_dir is None
