import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch

from base import Solution
from .net import ActorCritic, Actor, Critic
from ..buffer import RolloutBuffer
from ..rl_solver import RLSolver, PPOSolver, A2CSolver, ARPPOSolver, OnlineAgent
from ..utils import RunningMeanStd, get_pyg_data


class HrlAcSolver(OnlineAgent, PPOSolver):

    def __init__(self, controller, recorder, counter, **kwargs):
        OnlineAgent.__init__(self)
        PPOSolver.__init__(self, controller, recorder, counter, **kwargs)
        num_p_net_nodes = kwargs['p_net_setting']['num_nodes']
        self.policy = ActorCritic(p_net_num_nodes=num_p_net_nodes, p_net_feature_dim=1+3, p_net_edge_dim=1, v_net_feature_dim=2+3, v_net_edge_dim=1,
                                    embedding_dim=self.embedding_dim, dropout_prob=self.dropout_prob, batch_norm=self.batch_norm).to(self.device)
        self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters(), 'lr': self.lr_actor / 10},
                {'params': self.policy.critic.parameters(), 'lr': self.lr_critic / 10},
            ],
        )
        self.preprocess_obs = obs_as_tensor
        self.gamma = 1.
        self.gae_lambda = 0.98
        self.norm_reward = True
        self.compute_return_method = 'gae'
            

def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, dict):
        r"""Preprocess the observation to adapte to batch mode."""
        observation = obs
        p_net_data = get_pyg_data(observation['p_net_x'], observation['p_net_edge_index'], observation['p_net_edge_attr'])
        v_net_data = get_pyg_data(observation['v_net_x'], observation['v_net_edge_index'], observation['v_net_edge_attr'])
        obs_p_net = Batch.from_data_list([p_net_data]).to(device)
        obs_v_net = Batch.from_data_list([v_net_data]).to(device)
        obs_v_net_attrs = torch.FloatTensor(np.array([observation['v_net_attrs']])).to(device)
        return {'p_net': obs_p_net, 'v_net': obs_v_net, 'v_net_attrs': obs_v_net_attrs}
    # batch
    elif isinstance(obs, list):
        p_net_data_list, v_net_data_list, v_net_attrs_list = [], [], []
        for observation in obs:
            p_net_data = get_pyg_data(observation['p_net_x'], observation['p_net_edge_index'], observation['p_net_edge_attr'])
            p_net_data_list.append(p_net_data)
            v_net_data = get_pyg_data(observation['v_net_x'], observation['v_net_edge_index'], observation['v_net_edge_attr'])
            v_net_data_list.append(v_net_data)            
            v_net_attrs_list.append(observation['v_net_attrs'])
        obs_p_net = Batch.from_data_list(p_net_data_list).to(device)
        obs_v_net = Batch.from_data_list(v_net_data_list).to(device)
        obs_v_net_attrs = torch.FloatTensor(np.array(v_net_attrs_list)).to(device)
        return {'p_net': obs_p_net, 'v_net': obs_v_net, 'v_net_attrs': obs_v_net_attrs}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")
