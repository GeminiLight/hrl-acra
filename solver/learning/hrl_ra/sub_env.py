import numpy as np
import networkx as nx
from gym import spaces
from ..sub_rl_environment import JointPRStepSubRLEnv, PlaceStepSubRLEnv


class SubEnv(JointPRStepSubRLEnv):

    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        kwargs['node_ranking_method'] = 'nps'
        super(SubEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)
    
    def get_observation(self):
        p_net_obs = self._get_p_net_obs()
        v_net_obs = self._get_v_net_obs()
        return {
            'p_net_x': p_net_obs['x'],
            'p_net_edge_index': p_net_obs['edge_index'],
            'p_net_edge_attr': p_net_obs['edge_attr'],
            'v_net_x': v_net_obs['x'],
            'v_net_edge_index': v_net_obs['edge_index'],
            'v_net_edge_attr': v_net_obs['edge_attr'],
            'curr_v_node_id': self.curr_v_node_id
        }

    def _get_p_net_obs(self, ):
        attr_type_list = ['resource', 'extrema']
        v_node_min_link_demend = self.obs_handler.get_v_node_min_link_demend(self.v_net, self.curr_v_node_id)
        p_subnet = self.obs_handler.get_subgraph_view(self.p_net, v_node_min_link_demend)
        # node data
        node_data = self.obs_handler.get_node_attrs_obs(self.p_net, node_attr_types=attr_type_list, node_attr_benchmarks=self.node_attr_benchmarks)
        p_node_degree = self.obs_handler.get_node_degree_obs(p_subnet, self.degree_benchmark)
        p_node_link_max_resource = self.obs_handler.get_link_aggr_attrs_obs(p_subnet, link_attr_types=attr_type_list, aggr='max', link_attr_benchmarks=self.link_attr_benchmarks)
        p_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(p_subnet, link_attr_types=attr_type_list, aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        p_node_average_distance = self.obs_handler.get_average_distance_for_v_node(p_subnet, self.v_net, self.solution['node_slots'], self.curr_v_node_id, normalization=True, )
        p_nodes_status = self.obs_handler.get_p_nodes_status(self.p_net, self.v_net, self.solution['node_slots'], self.curr_v_node_id)
        v2p_node_link_demand = self.obs_handler.get_v2p_node_link_demand(self.p_net, self.v_net, self.solution['node_slots'], self.curr_v_node_id, self.link_attr_benchmarks)
        # p_node_positions = self.obs_handler.get_p_node_positions(self.p_net, self.solution['node_slots']) 
        node_data = np.concatenate((node_data, p_nodes_status, v2p_node_link_demand, p_node_degree, p_node_link_max_resource, p_node_link_sum_resource, p_node_average_distance), axis=-1)
        edge_index = self.obs_handler.get_link_index_obs(p_subnet)
        link_data = self.obs_handler.get_link_attrs_obs(p_subnet, link_attr_types=attr_type_list, link_attr_benchmarks=self.link_attr_benchmarks)
        # data
        p_net_obs = {
            'x': node_data,
            'edge_index': edge_index,
            'edge_attr': link_data
        }
        return p_net_obs

    def _get_v_net_obs(self):
        # node data
        node_data = self.obs_handler.get_node_attrs_obs(self.v_net, node_attr_types=['resource'], node_attr_benchmarks=self.node_attr_benchmarks)
        consist_decision = self.solution['place_result'] & self.solution['route_result']
        v_node_degree = self.obs_handler.get_node_degree_obs(self.v_net, self.degree_benchmark)
        v_node_link_max_resource = self.obs_handler.get_link_aggr_attrs_obs(self.v_net, link_attr_types=['resource'], aggr='max', link_attr_benchmarks=self.link_attr_benchmarks)
        v_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(self.v_net, link_attr_types=['resource'], aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        v_node_status = self.obs_handler.get_v_nodes_status(self.v_net, self.solution['node_slots'], self.curr_v_node_id, consist_decision=consist_decision)
        # import pdb; pdb.set_trace()
        v_node_positions = self.obs_handler.get_v_node_positions(list(self.ranked_v_net_nodes.keys()))
        node_data = np.concatenate((node_data, v_node_status, v_node_degree, v_node_link_max_resource, v_node_link_sum_resource, v_node_positions), axis=-1)
        # edge_index
        edge_index = self.obs_handler.get_link_index_obs(self.v_net)
        # edge_attr
        link_data = self.obs_handler.get_link_attrs_obs(self.v_net, link_attr_types=['resource'], link_attr_benchmarks=self.link_attr_benchmarks)
        v_net_obs = {
            'x': node_data,
            'edge_index': edge_index,
            'edge_attr': link_data,
        }
        return v_net_obs

    def compute_reward(self, solution):
        r"""Calculate deserved reward according to the result of taking action."""
        weight = (1 / self.v_net.num_nodes)
        if solution['result']:
            node_load_balance = self.get_node_load_balance(self.selected_p_net_nodes[-1])
            reward = weight * (solution['v_net_r2c_ratio'] + 0.01 * node_load_balance) 
            reward += solution['v_net_r2c_ratio']
        elif solution['place_result'] and solution['route_result']:
            node_load_balance = self.get_node_load_balance(self.selected_p_net_nodes[-1])
            reward = weight * ((solution['v_net_r2c_ratio']) + 0.01 * node_load_balance)
        else:
            reward = - weight
        # reward = reward * self.v_net.total_resource_demand / 500
        self.solution['v_net_reward'] += reward
        return reward
