import gym
import copy
import numpy as np
from gym import spaces

from base import Solution
from solver.heuristic.node_rank import GRCRankSolver, NRMRankSolver
from solver.learning.hrl_ra.hrl_ra_solver import HrlRaSolver

from ..rl_environment import SolutionStepRLEnv


class OnlineEnv(SolutionStepRLEnv):
    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, verbose=False, allow_rejection=False, **kwargs):
        kwargs_for_solver = copy.deepcopy(kwargs)
        kwargs_for_solver['k_searching'] = 1
        super(OnlineEnv, self).__init__(p_net, v_net_simulator, controller, recorder, counter, verbose=verbose, allow_rejection=allow_rejection, **kwargs_for_solver)
        max_num_v_net_nodes = self.v_net_simulator.v_sim_setting['v_net_size']['high']
        max_num_v_net_links = max_num_v_net_nodes * max_num_v_net_nodes
        self.max_num_v_net_nodes = max_num_v_net_nodes
        self.max_num_v_net_links = max_num_v_net_links
        self.action_space = spaces.Discrete(2)
        sub_solver_name = kwargs.get('sub_solver_name', 'nrm_rank')
        kwargs_for_sub_solver = copy.deepcopy(kwargs)
        kwargs_for_sub_solver['verbose'] = 0
        print(f'Employ {sub_solver_name} as sub solver')
        if sub_solver_name == 'nrm_rank':
            self.sub_solver = NRMRankSolver(self.controller, self.recorder, self.counter, **kwargs_for_sub_solver)
        elif sub_solver_name == 'grc_rank':
            self.sub_solver = GRCRankSolver(self.controller, self.recorder, self.counter, **kwargs_for_sub_solver)
        elif sub_solver_name == 'hrl_ra':
            pretrained_subsolver_model_path = kwargs.get('pretrained_subsolver_model_path', None)
            self.sub_solver = HrlRaSolver(self.controller, self.recorder, self.counter, **kwargs_for_sub_solver)
            if pretrained_subsolver_model_path not in [None, '']:
                print('Loading pretrained lower-level agent...')
                self.sub_solver.load_model(pretrained_subsolver_model_path)
            else:
                print('Randomly initialize the parameters of lower-level agent!')
            self.sub_solver.eval()
        else:
            raise NotImplementedError(f'Please specify a available sub solver: not {sub_solver_name}!')
        self.global_timestep_count = 0
        self.global_moving_average_reward = 0
        self.global_cumulative_reward = 0
        self.actual_cumulative_reward = 0
        # self.sub_solver = RandomWalkRankBFSSolver()
            
    def compute_reward(self, solution):
        r"""Calculate deserved reward according to the result of taking action."""
        w_a = 1
        w_b = solution['v_net_lifetime'] / self.v_net_simulator.v_sim_setting['lifetime']['scale']
        revenue_benchmark = 100
        if solution['result']:
            basic_reward = solution['v_net_revenue'] / revenue_benchmark
            weight = w_a + w_b
            reward = weight * basic_reward * solution['v_net_r2c_ratio']
        elif (not solution['result']) and (not solution['early_rejection']):
            basic_reward = self.v_net.total_resource_demand / revenue_benchmark
            reward = - 0.01 * (self.v_net.num_nodes)
        else:
            reward = 0
        self.actual_cumulative_reward += reward
        self.v_net_reward += reward
        self.global_timestep_count += 1
        self.global_cumulative_reward += reward
        average_reward = reward - self.global_cumulative_reward / self.global_timestep_count
        self.extra_record_info.update({
            'actual_cumulative_reward': self.actual_cumulative_reward,
            'global_cumulative_reward': self.global_cumulative_reward,
            'average_reward_benchmark': self.global_cumulative_reward / self.global_timestep_count,
            'cumulative_reward': self.cumulative_reward,
            'average_reward': average_reward,
            'actual_reward': reward,
        })
        self.cumulative_reward += average_reward
        # print(f'v_net_id: {self.v_net.id:4d}, actual_reward: {reward:+2.2f}, average_reward: {average_reward:+2.2f}, average_benchmark: {self.global_cumulative_reward / self.global_timestep_count:+2.2f}, ')
        return average_reward

    def step(self, action):
        if action:
            instance = {'v_net': self.v_net, 'p_net': self.p_net}
            solution = self.sub_solver.solve(instance)
        else:
            solution = Solution(self.v_net)
            solution['early_rejection'] = True
        return super().step(solution)

    def get_info(self, record=...):
        return super().get_info(record)

    def get_observation(self):
        p_net_obs = self._get_p_net_obs()
        v_net_obs = self._get_v_net_obs()
        v_net_attrs = self._get_v_net_attrs_obs()
        padding_v_net_attrs = np.expand_dims(v_net_attrs, axis=0).repeat(v_net_obs['x'].shape[0], axis=0)
        v_net_obs['x'] = np.concatenate((v_net_obs['x'], padding_v_net_attrs), axis=-1).astype(np.float32)
        return {
            'p_net_x': p_net_obs['x'],
            'p_net_edge_index': p_net_obs['edge_index'],
            'p_net_edge_attr': p_net_obs['edge_attr'],
            'v_net_attrs': v_net_attrs,
            'v_net_x': v_net_obs['x'],
            'v_net_edge_index': v_net_obs['edge_index'],
            'v_net_edge_attr': v_net_obs['edge_attr'],
        }
    

    def _get_p_net_obs(self):
        node_data = self.obs_handler.get_node_attrs_obs(self.p_net, node_attr_types=['resource'], node_attr_benchmarks=self.node_attr_benchmarks)
        p_node_degree = self.obs_handler.get_node_degree_obs(self.p_net, self.degree_benchmark)
        p_node_link_max_resource = self.obs_handler.get_link_aggr_attrs_obs(self.p_net, link_attr_types=['resource'], aggr='max', link_attr_benchmarks=self.link_attr_benchmarks)
        p_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(self.p_net, link_attr_types=['resource'], aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        node_data = np.concatenate((node_data, p_node_degree, p_node_link_max_resource, p_node_link_sum_resource), axis=-1)
        edge_index = self.obs_handler.get_link_index_obs(self.p_net)
        link_data = self.obs_handler.get_link_attrs_obs(self.p_net, link_attr_types=['resource'], link_attr_benchmarks=self.link_attr_benchmarks)
        # data
        p_net_obs = {
            'x': node_data,
            'edge_index': edge_index,
            'edge_attr': link_data
        }
        return p_net_obs

    def _get_v_net_obs(self):
        node_data = self.obs_handler.get_node_attrs_obs(self.v_net, node_attr_types=['resource'], node_attr_benchmarks=self.node_attr_benchmarks)
        v_node_degree = self.obs_handler.get_node_degree_obs(self.v_net, self.degree_benchmark)
        v_node_link_max_resource = self.obs_handler.get_link_aggr_attrs_obs(self.v_net, link_attr_types=['resource'], aggr='max', link_attr_benchmarks=self.link_attr_benchmarks)
        v_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(self.v_net, link_attr_types=['resource'], aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        node_data = np.concatenate((node_data, v_node_degree, v_node_link_max_resource, v_node_link_sum_resource), axis=-1)
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

    def _pad_v_net_obs(self, v_net_obs):
        num_v_net_nodes = self.v_net.num_nodes
        num_v_net_links = self.v_net.num_links
        v_net_obs['x'] = np.pad(v_net_obs['x'], ((0, self.max_num_v_net_nodes-num_v_net_nodes), (0, 0)), 'constant', constant_values=0)
        v_net_obs['edge_index'] = np.pad(v_net_obs['edge_index'], ((0, 0), (0, self.max_num_v_net_links-num_v_net_links)), 'constant', constant_values=0)
        v_net_obs['edge_attr'] = np.pad(v_net_obs['edge_attr'], ((0, self.max_num_v_net_links-num_v_net_links), (0, 0)), 'constant', constant_values=0)
        return v_net_obs

    def _get_v_net_attrs_obs(self):
        norm_lifetime = self.v_net.lifetime / self.v_net_simulator.v_sim_setting['lifetime']['scale']
        v_net_attrs = np.array([norm_lifetime], dtype=np.float32)
        return v_net_attrs
