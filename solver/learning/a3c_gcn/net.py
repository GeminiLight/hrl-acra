import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_batch
from ..net import GATConvNet, GCNConvNet, ResNetBlock, MLPNet


class ActorCritic(nn.Module):
    
    def __init__(self, p_net_num_nodes, output_dim, p_net_feature_dim, v_node_feature_dim, embedding_dim=64):
        super(ActorCritic, self).__init__()
        self.actor = Actor(p_net_num_nodes, output_dim, p_net_feature_dim, v_node_feature_dim, embedding_dim)
        self.critic = Critic(p_net_num_nodes, output_dim, p_net_feature_dim, v_node_feature_dim, embedding_dim)

    def act(self, obs):
        return self.actor.act(obs)

    def evaluate(self, obs):
        return self.critic.evaluate(obs)


class Actor(nn.Module):

    def __init__(self, p_net_num_nodes, output_dim, p_net_feature_dim, v_node_feature_dim, embedding_dim=64):
        super(Actor, self).__init__()
        self.gnn = nn.Sequential(
            GCNConvNet(p_net_feature_dim, output_dim, embedding_dim=embedding_dim, dropout_prob=0., return_batch=True),
            nn.Flatten(),
            nn.Linear(p_net_num_nodes, p_net_num_nodes),
            nn.ReLU(),
            nn.Linear(p_net_num_nodes, p_net_num_nodes),
        )

    def act(self, obs):
        """Return logits of actions"""
        p_net_embedding = self.gnn(obs['p_net'])
        action_logits = p_net_embedding
        return action_logits


class Critic(nn.Module):

    def __init__(self, p_net_num_nodes, output_dim, p_net_feature_dim, v_node_feature_dim, embedding_dim=64):
        super(Critic, self).__init__()
        self.gnn = nn.Sequential(
            GCNConvNet(p_net_feature_dim, output_dim, embedding_dim=embedding_dim, dropout_prob=0., return_batch=True),
            nn.Flatten(),
            nn.Linear(p_net_num_nodes, p_net_num_nodes),
            nn.ReLU(),
            nn.Linear(p_net_num_nodes, 1),
        )
        # )

    def evaluate(self, obs):
        """Return logits of actions"""
        p_net_embedding = self.gnn(obs['p_net'])
        # value = torch.mean(p_net_embedding, dim=-1, keepdim=True)
        value = p_net_embedding
        return value
