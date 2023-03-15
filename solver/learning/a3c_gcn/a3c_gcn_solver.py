import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch

from base import Solution
from .sub_env import SubEnv
from .net import ActorCritic, Actor, Critic
from ..buffer import RolloutBuffer
from ..rl_solver import RLSolver, PPOSolver, InstanceAgent, A2CSolver
from ..utils import get_pyg_data


class A3CGCNSolver(InstanceAgent, A2CSolver):

    def __init__(self, controller, recorder, counter, **kwargs):
        InstanceAgent.__init__(self)
        A2CSolver.__init__(self, controller, recorder, counter, **kwargs)
        num_p_net_nodes = kwargs['p_net_setting']['num_nodes']
        self.policy = ActorCritic(p_net_num_nodes=num_p_net_nodes, output_dim=1, p_net_feature_dim=8, v_node_feature_dim=5, embedding_dim=self.embedding_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                {'params': self.policy.critic.parameters(), 'lr': self.lr_critic},
            ],
        )
        self.SubEnv = SubEnv
        self.preprocess_obs = obs_as_tensor


def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, dict):
        r"""Preprocess the observation to adapte to batch mode."""
        data = get_pyg_data(obs['p_net_x'], obs['p_net_edge_index'])
        obs_p_net = Batch.from_data_list([data]).to(device)
        obs_v_node = torch.FloatTensor(obs['v_node']).unsqueeze(dim=0).to(device)
        return {'p_net': obs_p_net, 'v_node': obs_v_node}
    # batch
    elif isinstance(obs, list):
        p_net_data_list, v_node_list = [], []
        for observation in obs:
            p_net_data = get_pyg_data(observation['p_net_x'], observation['p_net_edge_index'])
            p_net_data_list.append(p_net_data)
            v_node_list.append(observation['v_node'])
        obs_p_net = Batch.from_data_list(p_net_data_list).to(device)
        obs_v_node = torch.FloatTensor(np.array(v_node_list)).to(device)
        return {'p_net': obs_p_net, 'v_node': obs_v_node}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")
