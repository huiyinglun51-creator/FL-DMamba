import time
import numpy as np
import torch
import torch.nn.functional as F
import gym

from models.decision_transformer import DecisionTransformer
from models.decision_mamba import DecisionMamba


class Trainer:
    def __init__(
            self,
            model_type,
            model,
            optimizer,
            batch_size,
            get_batch,
            loss_fn,
            scheduler=None,
            eval_fns=None,
            use_prefix_equiv=False,
            eq_loss_weight=0.1,
            eq_margin=0.25,
            eq_state_neighbors=8,
            eq_sig_spread_min=0.15,
            eq_max_tokens=512,
            eq_max_anchors=256,
    ):
        self.model_type = model_type
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self.batch_size = batch_size
        self.get_batch = get_batch
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.use_prefix_equiv = use_prefix_equiv
        self.eq_loss_weight = eq_loss_weight
        self.eq_margin = eq_margin
        self.eq_state_neighbors = eq_state_neighbors
        self.eq_sig_spread_min = eq_sig_spread_min
        self.eq_max_tokens = eq_max_tokens
        self.eq_max_anchors = eq_max_anchors

        self.start_time = time.time()

    # ** train one iter **
    def train_iteration(self, num_steps, iter_num=0, print_logs=False):
        train_losses = []
        logs = dict()

        train_start = time.time()
        self.model.train()
        for i in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

            if i % 1000 == 0:
                print(f'Step {i}')

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()
        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/num_of_updates'] = iter_num * num_steps
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def _compute_prefix_equiv_loss(self, states, prefix_features, future_sig, mask):
        valid = mask.reshape(-1) > 0
        cur_states = states.reshape(-1, states.shape[-1])[valid]
        reps = prefix_features.reshape(-1, prefix_features.shape[-1])[valid]
        sigs = future_sig.reshape(-1, future_sig.shape[-1])[valid]

        num_tokens = cur_states.shape[0]
        if num_tokens < 4:
            return reps.new_tensor(0.0), 0

        if num_tokens > self.eq_max_tokens:
            keep_idx = torch.randperm(num_tokens, device=cur_states.device)[:self.eq_max_tokens]
            cur_states = cur_states[keep_idx]
            reps = reps[keep_idx]
            sigs = sigs[keep_idx]
            num_tokens = cur_states.shape[0]


        sig_mean = sigs.mean(dim=0, keepdim=True)
        sig_std = sigs.std(dim=0, keepdim=True) + 1e-6
        norm_sigs = (sigs - sig_mean) / sig_std

        with torch.no_grad():
          
            state_dist = torch.cdist(cur_states, cur_states, p=2)
            state_dist.fill_diagonal_(float('inf'))

            k = min(self.eq_state_neighbors, num_tokens - 1)
            if k <= 0:
                return reps.new_tensor(0.0), 0

           
            _, knn_idx = torch.topk(state_dist, k=k, largest=False, dim=1)

           
            sig_dist = torch.cdist(norm_sigs, norm_sigs, p=2)
            local_sig_dist = sig_dist.gather(1, knn_idx)

            
            pos_choice = local_sig_dist.argmin(dim=1)

           
            sorted_local_dist, sorted_local_idx = torch.sort(local_sig_dist, descending=True, dim=1)
            pool_size = max(1, k // 2)
            rand_idx = torch.randint(0, pool_size, (num_tokens,), device=cur_states.device)
            neg_choice = sorted_local_idx[torch.arange(num_tokens), rand_idx]

            row_idx = torch.arange(num_tokens, device=cur_states.device)
            pos_idx = knn_idx[row_idx, pos_choice]
            neg_idx = knn_idx[row_idx, neg_choice]

     
            pos_sig_dist = local_sig_dist[row_idx, pos_choice]
            neg_sig_dist = local_sig_dist[row_idx, neg_choice]
            spread = neg_sig_dist - pos_sig_dist

            
            valid_anchor = spread > self.eq_sig_spread_min
            anchor_idx = torch.nonzero(valid_anchor, as_tuple=False).squeeze(1)

            if anchor_idx.numel() == 0:
                return reps.new_tensor(0.0), 0

            if anchor_idx.numel() > self.eq_max_anchors:
                perm = torch.randperm(anchor_idx.numel(), device=cur_states.device)[:self.eq_max_anchors]
                anchor_idx = anchor_idx[perm]

            
            dynamic_margin_scale = spread[anchor_idx]

        anc_reps = reps[anchor_idx]
        pos_reps = reps[pos_idx[anchor_idx]]
        neg_reps = reps[neg_idx[anchor_idx]]


        dist_pos = F.pairwise_distance(anc_reps, pos_reps, p=2)
        dist_neg = F.pairwise_distance(anc_reps, neg_reps, p=2)


        adaptive_margin = self.eq_margin * torch.clamp(dynamic_margin_scale, min=0.5, max=3.0)

        eq_loss = torch.relu(dist_pos - dist_neg + adaptive_margin).mean()

        return eq_loss, int(anchor_idx.numel())

   
    def train_step(self):
        batch = self.get_batch(self.batch_size)
        if len(batch) == 8:
            states, actions, rewards, dones, rtg, timesteps, mask, future_sig = batch
        else:
            states, actions, rewards, dones, rtg, timesteps, mask = batch
            future_sig = None

        action_target = torch.clone(actions)
        prefix_features = None

        if self.model_type == 'dt':
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rtg[:, :-1], timesteps, attention_mask=mask
            )
        else:
            if self.use_prefix_equiv:
                action_preds, prefix_features = self.model.forward(
                    states, actions, rtg[:, :-1], timesteps, return_features=True
                )
            else:
                action_preds = self.model.forward(states, actions, rtg[:, :-1], timesteps)

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[mask.reshape(-1) > 0]

        action_loss = self.loss_fn(action_preds, action_target)
        total_loss = action_loss

        eq_loss_value = 0.0
        eq_num_pairs = 0
        if self.use_prefix_equiv and (future_sig is not None) and (prefix_features is not None):
            eq_loss, eq_num_pairs = self._compute_prefix_equiv_loss(states, prefix_features, future_sig, mask)
            total_loss = total_loss + self.eq_loss_weight * eq_loss
            eq_loss_value = eq_loss.detach().cpu().item()

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean(
                (action_preds - action_target) ** 2).detach().cpu().item()
            self.diagnostics['training/action_loss'] = action_loss.detach().cpu().item()
            self.diagnostics['training/prefix_equiv_loss'] = eq_loss_value
            self.diagnostics['training/prefix_equiv_pairs'] = eq_num_pairs

        return total_loss.detach().cpu().item()


def get_env_info(env_name, dataset):
    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [1800, 3600, 7200, 36000, 72000]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [6000, 12000, 24000, 120000, 240000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [2500, 5000, 10000, 50000, 100000]
        scale = 1000.
    else:
        raise NotImplementedError

    return env, max_ep_len, env_targets, scale


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
        device='cuda',
):
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)


    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return

    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            actions=actions,
            returns_to_go=target_return.to(dtype=torch.float32),
            timesteps=timesteps,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode == 'normal':
            pred_return = target_return[0, -1] - (reward / scale)
        else:
            pred_return = target_return[0, -1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def get_model_optimizer(variant, state_dim, act_dim, max_ep_len, device):
    if variant['model_type'] == 'dt':
        if variant.get('use_prefix_equiv', False):
            raise NotImplementedError('Prefix-equivalence is currently implemented only for DecisionMamba variants.')
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=variant['embed_dim'],
            max_length=variant['K'],
            max_ep_len=max_ep_len,
            remove_act_embs=variant['remove_act_embs'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * variant['embed_dim'],
            activation_function=variant['activation_function'],
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif variant['model_type'] in ["dmamba-min", "dmamba"]:
        model = DecisionMamba(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=variant['embed_dim'],
            max_length=variant['K'],
            max_ep_len=max_ep_len,
            remove_act_embs=variant['remove_act_embs'],
            model_type=variant['model_type'],
            n_layer=variant['n_layer'],
            n_inner=4 * variant['embed_dim'],
            activation_function=variant['activation_function'],
            resid_pdrop=variant['dropout'],
            drop_p=variant['dropout'],
            window_size=variant['conv_window_size'],
            use_prefix_equiv=variant.get('use_prefix_equiv', False),
            prefix_hidden_size=variant.get('prefix_hidden_size', variant['embed_dim']),
        )
    else:
        raise NotImplementedError
    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    return model, optimizer, scheduler
