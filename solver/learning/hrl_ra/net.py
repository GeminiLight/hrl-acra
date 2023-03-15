import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_batch

from ..net import *


class VNetEncoder(nn.Module):

    GNNConvNet = DeepEdgeFeatureGAT

    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(VNetEncoder, self).__init__()
        self.v_net_gnn = self.GNNConvNet(v_net_feature_dim, embedding_dim, num_layers=3, embedding_dim=embedding_dim, edge_dim=v_net_edge_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.v_net_mean_pooling = GraphPooling('mean')
        self.v_net_att_pooling = GraphPooling('att', output_dim=embedding_dim)
        self.pe_mlp_net = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU())
        self.gap_mlp_net = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU())

    def forward(self, v_net_batch):
        # The old implementation of positional encoder
        #   In the lastest implementation, we regard positional embedding as input connnecting with other features
        # v_node_x, v_net_mask = to_dense_batch(v_net_batch.x, v_net_batch.batch)
        # v_node_x = self.pe_encoder(v_node_x)
        # v_net_feat_dim = v_node_x.shape[-1]
        # v_net_dense_mask = v_net_mask.unsqueeze(-1).repeat(1, 1, v_node_x.shape[-1])
        # v_node_x = torch.masked_select(v_node_x, v_net_dense_mask)
        # v_node_x = v_node_x.reshape(-1, v_net_feat_dim)
        # new_v_net_batch = v_net_batch.clone()
        # new_v_net_batch.x = v_node_x

        v_node_embeddings = F.leaky_relu(self.v_net_gnn(v_net_batch))
        v_node_dense_embeddings, v_net_mask = to_dense_batch(v_node_embeddings, v_net_batch.batch)
        v_mean_graph_embedding = self.v_net_mean_pooling(v_node_embeddings, v_net_batch.batch)
        v_att_graph_embedding = self.v_net_att_pooling(v_node_embeddings, v_net_batch.batch)
        v_graph_embedding = v_mean_graph_embedding + v_att_graph_embedding
        return v_graph_embedding, v_node_dense_embeddings


class PNetDecoder(nn.Module):

    GNNConvNet = DeepEdgeFeatureGAT

    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(PNetDecoder, self).__init__()
        self.p_net_lin = nn.Linear(p_net_feature_dim, embedding_dim)
        self.p_net_gnn = self.GNNConvNet(embedding_dim, 1, num_layers=5, embedding_dim=embedding_dim, edge_dim=p_net_edge_dim, dropout_prob=dropout_prob, batch_norm=batch_norm, return_batch=True)


    def forward(self, p_net_batch, v_net_batch, curr_v_node_id, v_graph_embedding, v_node_dense_embeddings):
        curr_v_node_id = curr_v_node_id.unsqueeze(1).unsqueeze(1).long()
        curr_v_node_embeding = v_node_dense_embeddings.gather(1, curr_v_node_id.expand(v_node_dense_embeddings.size()[0], -1, v_node_dense_embeddings.size()[-1])).squeeze(1)
        p_node_embeddings = F.leaky_relu(self.p_net_lin(p_net_batch['x']))
        p_node_embeddings = p_node_embeddings.reshape(p_net_batch.num_graphs, -1, p_node_embeddings.shape[-1])
        p_node_embeddings = p_node_embeddings + v_graph_embedding.unsqueeze(1) + curr_v_node_embeding.unsqueeze(1)
        p_node_embeddings = p_node_embeddings.reshape(-1, p_node_embeddings.shape[-1])
        new_p_net_batch = p_net_batch.clone()
        new_p_net_batch.x = p_node_embeddings
        p_node_embeddings = self.p_net_gnn(new_p_net_batch).squeeze(-1)
        return p_node_embeddings


class ActorCritic(nn.Module):
    
    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(ActorCritic, self).__init__()
        self.actor = Actor(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.critic = Critic(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)

    def act(self, obs):
        return self.actor.act(obs)

    def evaluate(self, obs):
        return self.critic.evaluate(obs)


class Actor(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(Actor, self).__init__()
        self.v_net_encoder = VNetEncoder(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.p_net_decoder = PNetDecoder(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob, batch_norm)
        
    def encode(self, v_net_batch):
        v_graph_embedding, v_node_dense_embeddings = self.v_net_encoder(v_net_batch)
        return v_graph_embedding, v_node_dense_embeddings

    def act(self, obs, v_graph_embedding=None, v_node_dense_embeddings=None):
        """Return logits of actions"""
        # import pdb; pdb.set_trace()
        if v_graph_embedding is None or v_node_dense_embeddings is None:
            v_graph_embedding, v_node_dense_embeddings = self.encode(obs['v_net'])
        action_logits = self.p_net_decoder(obs['p_net'], obs['v_net'], obs['curr_v_node_id'], v_graph_embedding, v_node_dense_embeddings)
        return action_logits


class Critic(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(Critic, self).__init__()
        self.v_net_encoder = VNetEncoder(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.p_net_decoder = PNetDecoder(p_net_num_nodes, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim, dropout_prob, batch_norm)

    def encode(self, v_net_batch):
        v_graph_embedding, v_node_embeddings = self.v_net_encoder(v_net_batch)
        return v_graph_embedding, v_node_embeddings

    def evaluate(self, obs, v_graph_embedding=None, v_node_dense_embeddings=None):
        """Return logits of actions"""
        if v_graph_embedding is None or v_node_dense_embeddings is None:
            v_graph_embedding, v_node_dense_embeddings = self.encode(obs['v_net'])
        v_node_dense_embeddings
        fusion = self.p_net_decoder(obs['p_net'], obs['v_net'], obs['curr_v_node_id'], v_graph_embedding, v_node_dense_embeddings)
        value = torch.mean(fusion, dim=-1, keepdim=True)
        return value
    