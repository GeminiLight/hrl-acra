import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from .sub_env import SubEnv
from .net import ActorCritic
from ..rl_solver import RLSolver, PPOSolver, InstanceAgent
from ..utils import get_pyg_data
from ..obs_handler import POSITIONAL_EMBEDDING_DIM


class HrlRaSolver(InstanceAgent, PPOSolver):
    def __init__(self, controller, recorder, counter, **kwargs):
        InstanceAgent.__init__(self)
        PPOSolver.__init__(self, controller, recorder, counter, **kwargs)
        num_p_net_nodes = kwargs['p_net_setting']['num_nodes']
        self.policy = ActorCritic(p_net_num_nodes=num_p_net_nodes, p_net_feature_dim=4+4+3, p_net_edge_dim=1+1, v_net_feature_dim=3+3+POSITIONAL_EMBEDDING_DIM, v_net_edge_dim=1,
                                    embedding_dim=self.embedding_dim, dropout_prob=self.dropout_prob, batch_norm=self.batch_norm).to(self.device)
        self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                {'params': self.policy.critic.parameters(), 'lr': self.lr_critic},
            ], weight_decay=self.weight_decay
        )
        self.SubEnv = SubEnv
        self.norm_reward = False
        self.preprocess_obs = obs_as_tensor

def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, dict):
        r"""Preprocess the observation to adapte to batch mode."""
        observation = obs
        p_net_data = get_pyg_data(observation['p_net_x'], observation['p_net_edge_index'], observation['p_net_edge_attr'])
        v_net_data = get_pyg_data(observation['v_net_x'], observation['v_net_edge_index'], observation['v_net_edge_attr'])
        obs_p_net = Batch.from_data_list([p_net_data]).to(device)
        obs_v_net = Batch.from_data_list([v_net_data]).to(device)
        obs_curr_v_node_id = torch.LongTensor(np.array([observation['curr_v_node_id']])).to(device)
        return {'p_net': obs_p_net, 'v_net': obs_v_net, 'curr_v_node_id': obs_curr_v_node_id}
    # batch
    elif isinstance(obs, list):
        p_net_data_list, v_net_data_list, curr_v_node_id_list = [], [], []
        for observation in obs:
            p_net_data = get_pyg_data(observation['p_net_x'], observation['p_net_edge_index'], observation['p_net_edge_attr'])
            p_net_data_list.append(p_net_data)
            v_net_data = get_pyg_data(observation['v_net_x'], observation['v_net_edge_index'], observation['v_net_edge_attr'])
            v_net_data_list.append(v_net_data)            
            curr_v_node_id_list.append(observation['curr_v_node_id'])
        obs_p_net = Batch.from_data_list(p_net_data_list).to(device)
        obs_v_net = Batch.from_data_list(v_net_data_list).to(device)
        obs_curr_v_node_id = torch.LongTensor(np.array(curr_v_node_id_list)).to(device)
        return {'p_net': obs_p_net, 'v_net': obs_v_net, 'curr_v_node_id': obs_curr_v_node_id}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")

