# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch OpenAI GPT-2 model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_utils import PreTrainedModel  # , Conv1D
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from models.layers import Block


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config, index) for index in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def forward(self, inputs_embeds=None):
        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class DecisionMamba(nn.Module):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)

    Prefix-equivalence extension:
    - action head stays unchanged
    - optional prefix projection head maps state token representations into a
      continuation-aware embedding space used by the trainer's contrastive loss
    """
    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            remove_act_embs=False,
            **kwargs
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.use_prefix_equiv = kwargs.get('use_prefix_equiv', False)
        self.prefix_hidden_size = kwargs.get('prefix_hidden_size', hidden_size)
        self.supports_prefix_equiv = True

        config = transformers.GPT2Config(
            vocab_size=1, 
            n_embd=hidden_size,
            remove_act_embs=remove_act_embs,
            max_length=max_length,
            **kwargs
        )

      
        self.transformer = GPT2Model(config)
        self.remove_act_embs = remove_act_embs

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

        if self.use_prefix_equiv:
            self.prefix_proj = nn.LayerNorm(hidden_size)
            
        else:
            self.prefix_proj = None

    def forward(self, states, actions, returns_to_go, timesteps, return_features=False):
        batch_size, seq_length = states.shape[0], states.shape[1]

       
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings
        if not self.remove_act_embs:
            action_embeddings = self.embed_action(actions) + time_embeddings

        if self.remove_act_embs:
            num_token_type = 2
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type * seq_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)
        else:
            num_token_type = 3
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type * seq_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)

     
        x = self.transformer(inputs_embeds=stacked_inputs)

     
        x = x.reshape(batch_size, seq_length, num_token_type, self.hidden_size).permute(0, 2, 1, 3)

        state_reps = x[:, 1]
        action_preds = self.predict_action(state_reps)  

        if return_features:
            if self.prefix_proj is not None:
                prefix_features = F.normalize(self.prefix_proj(state_reps), dim=-1)
            else:
                prefix_features = F.normalize(state_reps, dim=-1)
            return action_preds, prefix_features

        return action_preds

    def get_action(self, states, actions, returns_to_go, timesteps):
        
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        states = states[:, -self.max_length:]
        actions = actions[:, -self.max_length:]
        returns_to_go = returns_to_go[:, -self.max_length:]
        timesteps = timesteps[:, -self.max_length:]

       
        states = torch.cat(
            [torch.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim), device=states.device), states],
            dim=1).to(dtype=torch.float32)
        actions = torch.cat(
            [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim), device=actions.device), actions],
            dim=1).to(dtype=torch.float32)
        returns_to_go = torch.cat(
            [torch.zeros((returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
            dim=1).to(dtype=torch.float32)
        timesteps = torch.cat(
            [torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device), timesteps],
            dim=1).to(dtype=torch.long)

        action_preds = self.forward(states, actions, returns_to_go, timesteps)
        return action_preds[0, -1]
