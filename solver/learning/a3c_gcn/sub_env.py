import numpy as np
import networkx as nx
from gym import spaces
from ..sub_rl_environment import JointPRStepSubRLEnv, PlaceStepSubRLEnv


class SubEnv(JointPRStepSubRLEnv):

    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        super(SubEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)
        num_p_net_node_attrs = len(self.p_net.get_node_attrs(['resource', 'extrema']))
        num_p_net_link_attrs = len(self.p_net.get_link_attrs(['resource', 'extrema']))
        num_p_net_features = num_p_net_node_attrs + 1
        self.observation_space = spaces.Dict({
            'p_net_x': spaces.Box(low=0, high=1, shape=(self.p_net.num_nodes, num_p_net_features), dtype=np.float32),
            'p_net_edge_index': spaces.Box(low=0, high=self.p_net.num_nodes, shape=(2, self.p_net.num_links), dtype=np.int32),
            'p_net_edge_attr': spaces.Box(low=0, high=self.p_net.num_nodes, shape=(self.p_net.num_links, 2), dtype=np.int32),
            'v_node': spaces.Box(low=0, high=100, shape=(int(num_p_net_node_attrs/2) + int(num_p_net_link_attrs/2) + 2, ), dtype=np.float32)
        })
        
    def compute_reward(self, solution):
        r"""Calculate deserved reward according to the result of taking action."""
        if solution['result'] :
            reward = solution['v_net_r2c_ratio']
        elif solution['place_result'] and solution['route_result']:
            curr_place_progress = self.get_curr_place_progress()
            node_load_balance = self.get_node_load_balance(self.selected_p_net_nodes[-1])
            reward = curr_place_progress * (solution['v_net_r2c_ratio'])
        else:
            reward = - self.get_curr_place_progress()
        self.solution['v_net_reward'] += reward
        return reward

    def get_observation(self):
        p_net_obs = self._get_p_net_obs()
        v_node_obs = self._get_v_node_obs()
        v_node_obs = np.expand_dims(v_node_obs, axis=0).repeat(self.p_net.num_nodes, axis=0)
        p_net_obs['x'] = np.concatenate((p_net_obs['x'], v_node_obs), axis=-1).astype(np.float32)
        # sub - attr
        return {'p_net_x': p_net_obs['x'],
                'p_net_edge_index': p_net_obs['edge_index'],
                'v_node': v_node_obs}

    def _get_p_net_obs(self):
        """
        node_resource, average_distance, p_net_degreees, p_net_nodes_states, v_node_features
        """
        # node data
        node_data = self.obs_handler.get_node_attrs_obs(self.p_net, node_attr_types=['resource', 'extrema'], node_attr_benchmarks=self.node_attr_benchmarks)
        edge_aggr_data = self.obs_handler.get_link_aggr_attrs_obs(self.p_net, link_attr_types=['resource', 'extrema'], aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        selected_nodes = np.zeros(self.p_net.num_nodes, dtype=np.float32)
        selected_nodes[self.selected_p_net_nodes] = 1.
        node_data = np.concatenate((node_data, edge_aggr_data, np.expand_dims(selected_nodes, axis=-1)), axis=-1)
        # edge_index
        edge_index = self.obs_handler.get_link_index_obs(self.p_net)
        # data
        p_net_obs = {
            'x': node_data,
            'edge_index': edge_index,
        }
        return p_net_obs

    def _get_v_node_obs(self):
        if self.curr_v_node_id  >= self.v_net.num_nodes:
            return []
        norm_unplaced = (self.v_net.num_nodes - (self.curr_v_node_id + 1)) / self.v_net.num_nodes
        node_demand = []
        for n_attr in self.v_net.get_node_attrs('resource'):
            node_demand.append(self.v_net.nodes[self.curr_v_node_id][n_attr.name] / self.node_attr_benchmarks[n_attr.name])
        norm_node_demand = np.array(node_demand, dtype=np.float32)

        max_link_demand = []
        mean_link_demand = []
        sum_link_demand = []
        for l_attr in self.v_net.get_link_attrs('resource'):
            link_demand = [self.v_net.links[(n, self.curr_v_node_id)][l_attr.name] for n in self.v_net.adj[self.curr_v_node_id]]
            max_link_demand.append(max(link_demand) / self.link_attr_benchmarks[l_attr.name])
            mean_link_demand.append((sum(link_demand) / len(link_demand)) / self.link_attr_benchmarks[l_attr.name])
            sum_link_demand.append(sum(link_demand) / self.link_attr_benchmarks[l_attr.name])
        # v_net_obs = np.concatenate([norm_node_demand, max_link_demand, mean_link_demand, [num_neighbors, norm_unplaced, norm_all_nodes, norm_curr_vid]], axis=0)
        v_net_obs = np.concatenate([norm_node_demand, sum_link_demand, [norm_unplaced]], axis=0)
        return v_net_obs