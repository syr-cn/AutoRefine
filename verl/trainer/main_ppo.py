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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_dapo_trainer import RayDAPOTrainer
import re
import numpy as np
import json
from collections import defaultdict


LOG_FUNCS = {
    'information_scores': qa_em.compute_information_score_subem,
    'information_reverse_rank': qa_em.compute_information_reverse_rank,
    'answer_em': qa_em.compute_score_em,
    'answer_f1': qa_em.compute_score_f1,
    'answer_cem': qa_em.compute_score_cem,
    'refine_scores': qa_em.compute_refine_score_subem,
    'format_scores': qa_em.compute_score_format,
}

class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0., refine_score=0., reward_style='EM', log_path=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.refine_score = refine_score
        self.log_path = log_path
        self.reward_style = reward_style

    def get_refine_subem(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            responses_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            compute_score_fn = qa_em.compute_refine_score_subem

            score = compute_score_fn(solution_str=sequences_str, responses_str=responses_str, ground_truth=ground_truth, format_score=self.format_score)

            reward_tensor[i, valid_response_length - 1] = score
        return reward_tensor

    def get_logging_scores(self, data: DataProto, step: int = -1):
        additional_scores = defaultdict(lambda: torch.zeros(len(data), dtype=torch.float32))
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            responses_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            for key, compute_fn in LOG_FUNCS.items():
                score = compute_fn(solution_str=sequences_str, responses_str=responses_str, ground_truth=ground_truth, format_score=0.0)
                additional_scores[key][i] = score
            
            scores_item = {key: additional_scores[key][i].item() for key in additional_scores.keys()}

            data_source = data_item.non_tensor_batch['data_source']
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                if already_print_data_sources[data_source] == 1:
                    print(sequences_str)
                if self.log_path is not None:
                    assert self.log_path.endswith('.jsonl')
                    log_info = {
                        'step': step,
                        'data_source': data_source,
                        'scores': scores_item,
                        'ground_truth': ground_truth['target'].tolist(),
                        'response': sequences_str,
                    }
                    with open(self.log_path, 'a+') as f:
                        f.write(json.dumps(log_info) + '\n')
            
        return additional_scores

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # all_scores = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            responses_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            if self.reward_style.lower() == 'em':
                compute_score_fn = qa_em.compute_score_em
            elif self.reward_style.lower() == 'f1':
                compute_score_fn = qa_em.compute_score_f1
            elif self.reward_style.lower() == 'cem':
                compute_score_fn = qa_em.compute_score_cem
            else:
                raise NotImplementedError
            score = qa_em.compute_reward(solution_str=sequences_str, responses_str=responses_str, ground_truth=ground_truth, score_func=compute_score_fn, format_score=self.format_score, refine_score=self.refine_score, do_print_frac=1024)

            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='grpo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    train_log_jsonl = f'log/train/{config.trainer.experiment_name}.jsonl'
    refine_score = config.actor_rollout_ref.actor.refine_score
    format_score = config.actor_rollout_ref.actor.format_score
    reward_style = config.reward_model.reward_style
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=config.reward_model.train_num_examine, log_path=train_log_jsonl, format_score=format_score, refine_score=refine_score, reward_style=reward_style)

    # Note that we always use function-based RM for validation
    val_log_jsonl = f'log/val/{config.trainer.experiment_name}.jsonl'
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=config.reward_model.val_num_examine, log_path=val_log_jsonl, reward_style=reward_style)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    if config.algorithm.filter_groups.enable:
        Trainer = RayDAPOTrainer
    else:
        Trainer = RayPPOTrainer
    trainer = Trainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
