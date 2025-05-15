from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer import _timer, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage
from verl.trainer.ppo.ray_trainer import compute_data_metrics, compute_timing_metrics, reduce_metrics, compute_throughout_metrics

import uuid
from pprint import pprint
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

from torch.nn.functional import pad
from tensordict import TensorDict

def pad_batches(batch_list, pad_id):
    # Determine max N from variable-length tensors (N=2047 inferred from keys)
    max_N = max(batch['responses'].size(1) for batch in batch_list)
    max_total = max(batch['attention_mask'].size(1) for batch in batch_list)

    padded_batches = []
    for batch in batch_list:
        pad_len = max_N - batch['responses'].size(1)
        total_pad = max_total - batch['attention_mask'].size(1)
        
        def pad_tensor(t, value, target_len):
            return pad(t, (0, target_len - t.size(1)), value=value) if t.size(1) < target_len else t

        padded = {
            'attention_mask': pad_tensor(batch['attention_mask'], 0, max_total),
            'info_mask': pad_tensor(batch['info_mask'], 0, max_total),
            'input_ids': pad_tensor(batch['input_ids'], pad_id, max_total),
            'old_log_probs': pad_tensor(batch['old_log_probs'], 0.0, max_N),
            'position_ids': pad_tensor(batch['position_ids'], 0, max_total),
            'prompts': batch['prompts'],  # already 2048
            'responses': pad_tensor(batch['responses'], pad_id, max_N),
            'responses_with_info_mask': pad_tensor(batch['responses_with_info_mask'], pad_id, max_N),
            'token_level_rewards': pad_tensor(batch['token_level_rewards'], 0.0, max_N),
            'token_level_scores': pad_tensor(batch['token_level_scores'], 0.0, max_N),
            'token_level_information_scores': pad_tensor(batch['token_level_information_scores'], 0.0, max_N),
            'token_level_refine_scores': pad_tensor(batch['token_level_refine_scores'], 0.0, max_N),
            'token_level_answer_em': pad_tensor(batch['token_level_answer_em'], 0.0, max_N),
        }
        padded_batches.append(padded)
        padded_batches = [TensorDict(b, batch_size=b['attention_mask'].size(0)) for b in padded_batches]
    return padded_batches

class RayDAPOTrainer(RayPPOTrainer):
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = 0
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics:')
            for key in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'mean']:
                val_key = f'val/test_score/{key}'
                val = val_metrics.get(val_key, None)
                print(f'{val_key}: {val}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        # start training loop
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        self.best_reward = float('-inf')
        self.best_val = 0.0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation
                gen_batch = new_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                ####################
                # original code here

                with _timer('step', timing_raw):
                    if not self.config.do_search:
                        raise NotImplementedError("The non-search mode is not implemented yet.")
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                    ####################
                    # Below is aLL about agents - the "LLM + forloop"
                    ####################
                    else:
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        with _timer('gen', timing_raw):
                            generation_manager.timing_raw = timing_raw
                            final_gen_batch_output = generation_manager.run_llm_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        for key in final_gen_batch_output.batch.keys():
                            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output)

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        #                                         dtype=object)
                        # new_batch.non_tensor_batch['uid'] = np.array(
                        #     [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                        new_batch.non_tensor_batch['uid'] = new_batch.non_tensor_batch['index'].copy()
                                            
                        # repeat to align with repeated responses in rollout
                        new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        new_batch = new_batch.union(final_gen_batch_output)

                    ####################
                    ####################
                    with _timer('reward', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            assert False
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(new_batch)
                        new_batch.batch['token_level_scores'] = reward_tensor
                        new_batch.batch['token_level_information_scores'] = self.reward_fn.get_subem(new_batch)
                        new_batch.batch['token_level_answer_em'] = self.reward_fn.get_em(batch)

                        refine_reward_tensor = self.reward_fn.get_refine_subem(batch)
                        batch.batch['token_level_refine_scores'] = self.reward_fn.get_refine_subem(batch)
                        if self.config.actor_rollout_ref.actor.refine_lambda > 0:
                            reward_tensor += self.config.actor_rollout_ref.actor.refine_lambda * refine_reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch['token_level_rewards'] = new_batch.batch['token_level_scores']

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size, we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        new_batch.non_tensor_batch[metric_name] = new_batch.batch[metric_name].sum(dim=-1).numpy()
                        # new_batch.non_tensor_batch[metric_name] = new_batch.batch['token_level_scores'].sum(dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch['uid'],
                                                   new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_mean = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_mean[prompt_uid] = np.mean(metric_vals)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        if self.config.algorithm.filter_groups.method == 'dapo':
                            kept_prompt_uids = [
                                uid for uid, std in prompt_uid2metric_std.items()
                                if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                            ]
                        elif self.config.algorithm.filter_groups.method == 'ours':
                            metric_name_info = "token_level_information_scores" # TODO: make this configurable
                            new_batch.non_tensor_batch[metric_name_info] = new_batch.batch[metric_name_info].sum(dim=-1).numpy()
                            prompt_uid2info_vals = defaultdict(list)
                            for uid, reward_val in zip(new_batch.non_tensor_batch['uid'], new_batch.non_tensor_batch[metric_name_info]):
                                prompt_uid2info_vals[uid].append(reward_val)
                            prompt_uid2info_mean = {}
                            for prompt_uid, reward_vals in prompt_uid2info_vals.items():
                                prompt_uid2info_mean[prompt_uid] = np.mean(reward_vals)
                            prompt_uid2info_std = {}
                            for prompt_uid, reward_vals in prompt_uid2info_vals.items():
                                prompt_uid2info_std[prompt_uid] = np.std(reward_vals)

                            bad_sample_ratio = 0
                            min_threshold = 0.1 # maybe too hard sample
                            max_threshold = 0.9 # too easy sample

                            print(f'[min_threshold] {min_threshold=}')
                            print(f'[max_threshold] {max_threshold=}')
                            print(f'[bad_sample_ratio] {bad_sample_ratio=}')
                            good_uids = [uid for uid, mean in prompt_uid2info_mean.items() if (mean >= min_threshold and mean <= max_threshold)]
                            bad_uids = [uid for uid, mean in prompt_uid2info_mean.items() if uid not in good_uids]
                            print(f'[good_uids] {len(good_uids)=}, [bad_uids] {len(bad_uids)=}')

                            if bad_sample_ratio > 0:
                                num_allowed_bad_samples = int(bad_sample_ratio * self.config.data.train_batch_size)
                                selected_bad_uids = random.sample(bad_uids, min(len(bad_uids), num_allowed_bad_samples))
                            else:
                                selected_bad_uids = []
                            selected_uids = good_uids + selected_bad_uids

                            kept_prompt_uids = [
                                uid for uid, std in prompt_uid2metric_std.items()
                                if uid in selected_uids and (std > 0 or len(prompt_uid2metric_vals[uid]) == 1)
                            ]
                        print(f'[kept_prompt_uids] {len(kept_prompt_uids)=}')
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch['uid']):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch.batch, new_batch.batch = pad_batches([batch.batch, new_batch.batch], pad_id=self.tokenizer.pad_token_id)
                            batch = DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f'[DAPO Filtering {self.config.algorithm.filter_groups.method}/{metric_name}] {num_prompt_in_batch=} < {prompt_bsz=}')
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f'[DAPO Filtering {self.config.algorithm.filter_groups.method}/{metric_name}] {num_gen_batches=}. Keep generating...')
                                continue
                            else:
                                raise ValueError(
                                    f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                                )
                        else:
                            # Align the batch
                            print(f'[DAPO Filtering {self.config.algorithm.filter_groups.method}/{metric_name}] {num_prompt_in_batch=} >= {prompt_bsz=}. Stop generating.')
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    assert batch is not None, "Batch should not be None after filtering."
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if not isinstance(batch, DataProto):
                        print(f"Batch is not DataProto, converting...")
                        batch = DataProto(
                            batch.batch,
                            batch.non_tensor_batch,
                            batch.meta_info,
                        )
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    # for key in batch.batch.keys():
                    #     if key != 'old_log_probs':
                    #         batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                n_gpus = self.config.trainer.n_gpus_per_node
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                if metrics['critic/rewards/mean'] > self.best_reward:
                    with _timer('save_best_checkpoint', timing_raw):
                        self._save_checkpoint_best()
                    self.best_reward = max(self.best_reward, metrics['critic/rewards/mean'])
                # TODO: make a canonical logger that supports various backend

                if 'val/test_score/mean' in metrics and metrics['val/test_score/mean'] > self.best_val:
                    with _timer('save_best_checkpoint', timing_raw):
                        self._save_checkpoint_best_val()
                    self.best_val = max(self.best_val, metrics['val/test_score/mean'])

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return