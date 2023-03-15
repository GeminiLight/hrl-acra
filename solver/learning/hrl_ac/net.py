import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_batch
from ..net import GATConvNet, GraphAttentionPooling, GraphPooling, ResNetBlock, MLPNet, DeepEdgeFeatureGAT, MLPNet


class Encoder(nn.Module):
    """The released version cancels the GRU modules to accelerate convergence for readers' fast validation, while the loss of performance is negligible. 
    """
    def __init__(self, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(Encoder, self).__init__()
        self.p_net_gnn = DeepEdgeFeatureGAT(p_net_feature_dim, embedding_dim, num_layers=5, embedding_dim=embedding_dim, edge_dim=p_net_edge_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.v_net_gnn = DeepEdgeFeatureGAT(v_net_feature_dim, embedding_dim, num_layers=3, embedding_dim=embedding_dim, edge_dim=v_net_edge_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.v_net_gap = GraphAttentionPooling(embedding_dim)
        self.p_net_gap = GraphAttentionPooling(embedding_dim)
        self.p_net_mean_pool = GraphPooling(aggr='mean')
        self.v_net_mean_pool = GraphPooling(aggr='mean')
        self.p_net_sum_pool = GraphPooling(aggr='sum')
        self.v_net_sum_pool = GraphPooling(aggr='sum')


    def forward(self, p_net_batch, v_net_batch):
        # import pdb; pdb.set_trace()
        v_net_node_embeddings = self.v_net_gnn(v_net_batch)
        v_net_gap_global_embedding = self.v_net_gap(v_net_node_embeddings, v_net_batch.batch)
        p_net_node_embeddings = self.p_net_gnn(p_net_batch)
        p_net_gap_global_embedding = self.p_net_gap(p_net_node_embeddings, p_net_batch.batch)
        p_net_mean_global_embedding = self.p_net_mean_pool(p_net_node_embeddings, p_net_batch.batch)
        v_net_mean_global_embedding = self.v_net_mean_pool(v_net_node_embeddings, v_net_batch.batch)
        p_net_sum_global_embedding = self.p_net_sum_pool(p_net_node_embeddings, p_net_batch.batch)
        v_net_sum_global_embedding = self.v_net_sum_pool(v_net_node_embeddings, v_net_batch.batch)
        p_net_global_embedding = p_net_gap_global_embedding + p_net_mean_global_embedding + p_net_sum_global_embedding
        v_net_global_embedding = v_net_gap_global_embedding + v_net_mean_global_embedding + v_net_sum_global_embedding
        fusion_embedding = torch.concat([p_net_global_embedding, v_net_global_embedding], dim=-1) 
        return fusion_embedding


class ActorCritic(nn.Module):
    
    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(ActorCritic, self).__init__()
        self.actor = Actor(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.critic = Critic(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)

    def act(self, obs):
        return self.actor.act(obs)

    def evaluate(self, obs):
        return self.critic.evaluate(obs)

    def predict(self, obs):
        output = self.predictor.predict(obs)
        return output


class Actor(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(Actor, self).__init__()
        self.encoder = Encoder(p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob, batch_norm)
        embedding_dims = [embedding_dim*2, embedding_dim]
        self.net = MLPNet(embedding_dim*2, 2, num_layers=3, embedding_dims=embedding_dims, batch_norm=False)

    def act(self, obs):
        """Return logits of actions"""
        action_logits = self.net(self.encoder(obs['p_net'], obs['v_net']))
        return action_logits


class Critic(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(Critic, self).__init__()
        self.encoder = Encoder(p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob, batch_norm)
        embedding_dims = [embedding_dim*2, embedding_dim]
        self.net = MLPNet(embedding_dim*2, 1, num_layers=3, embedding_dims=embedding_dims, batch_norm=False)

    def evaluate(self, obs):
        """Return logits of actions"""
        value = self.net(self.encoder(obs['p_net'], obs['v_net']))
        # value = torch.mean(p_net_embeddings, dim=-1, keepdim=True)
        # value = p_net_embeddings
        return value