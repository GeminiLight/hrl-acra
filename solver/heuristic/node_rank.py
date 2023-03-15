import networkx as nx

from base import Solution
from ..solver import Solver
from ..rank.node_rank import OrderNodeRank, GRCNodeRank, FFDNodeRank, NRMNodeRank, RWNodeRank, RandomNodeRank
from ..rank.link_rank import OrderLinkRank, FFDLinkRank


class NodeRankSolver(Solver):
    
    def __init__(self, controller, recorder, counter, **kwargs):
        super(NodeRankSolver, self).__init__(controller, recorder, counter, **kwargs)

    def solve(self, instance):
        v_net, p_net  = instance['v_net'], instance['p_net']

        solution = Solution(v_net)
        node_mapping_result = self.node_mapping(v_net, p_net, solution)
        if node_mapping_result:
            link_mapping_result = self.link_mapping(v_net, p_net, solution)
            if link_mapping_result:
                # SUCCESS
                solution['result'] = True
                return solution
            else:
                # FAILURE
                solution['route_result'] = False
        else:
            # FAILURE
            solution['place_result'] = False
        solution['result'] = False
        return solution

    def node_mapping(self, v_net, p_net, solution):
        r"""Attempt to accommodate VNF in appropriate physical node."""
        v_net_rank = self.node_rank(v_net)
        p_net_rank = self.node_rank(p_net)
        sorted_v_nodes = list(v_net_rank)
        sorted_p_nodes = list(p_net_rank)
        
        node_mapping_result = self.controller.node_mapping(v_net, p_net, 
                                                        sorted_v_nodes=sorted_v_nodes, 
                                                        sorted_p_nodes=sorted_p_nodes, 
                                                        solution=solution, 
                                                        reusable=False, 
                                                        inplace=True, 
                                                        matching_mathod=self.matching_mathod)
        return node_mapping_result

    def link_mapping(self, v_net, p_net, solution):
        r"""Seek a path connecting """
        if self.link_rank is None:
            sorted_v_links = v_net.links
        else:
            v_net_edges_rank_dict = self.link_rank(v_net)
            v_net_edges_sort = sorted(v_net_edges_rank_dict.items(), reverse=True, key=lambda x: x[1])
            sorted_v_links = [edge_value[0] for edge_value in v_net_edges_sort]

        link_mapping_result = self.controller.link_mapping(v_net, p_net, solution=solution, 
                                                        sorted_v_links=sorted_v_links, 
                                                        shortest_method=self.shortest_method,
                                                        k=self.k_shortest, inplace=True)
        return link_mapping_result


class OrderRankSolver(NodeRankSolver):
    
    name = 'order_rank'

    def __init__(self, controller, recorder, counter, **kwargs):
        super(OrderRankSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.node_rank = OrderNodeRank()
        self.link_rank = None


class GRCRankSolver(NodeRankSolver):

    name = 'grc_rank'

    def __init__(self, controller, recorder, counter, sigma=0.00001, d=0.85, **kwargs):
        super(GRCRankSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.node_rank = GRCNodeRank(sigma=sigma, d=d)
        self.link_rank = None
        # self.shortest_method = 'available_shortest'

class NRMRankSolver(NodeRankSolver):

    name = 'nrm_rank'

    def __init__(self, controller, recorder, counter, **kwargs):
        super(NRMRankSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.node_rank = NRMNodeRank()
        self.link_rank = None


class PLRankSolver(NodeRankSolver):
    r"""
    An implementation of PL solver proposed in
    Fan et al. "Efficient Virtual Network Embedding of Cloud-Based Data Center Networks into Optical Networks". TPDS, 2021. 
    """
    def __init__(self, controller, recorder, counter, **kwargs):
        super(PLRankSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.node_rank = NRMNodeRank()
        self.link_rank = None

    def rank_path(self, v_net, p_net, v_link, p_pair, p_paths):
        link_resource_attrs = v_net.get_link_attrs(['resource'])
        node_resource_attrs = v_net.get_node_attrs(['resource'])
        p_path_rank_values_dict = {}
        for p_path in p_paths:
            p_links = self.controller.path_to_links(p_path)
            p_links_bw_list = []
            for p_link in p_links:
                p_links_bw_list.append(sum(p_net.links[p_link][l_attr.name] for l_attr in link_resource_attrs))

            p_nodes_resource_list = []
            p_nodes = p_paths[1:-2]
            for p_nodes in p_nodes:
                p_nodes_resource_list.append(sum(p_net.node[p_link][n_attr.name] for n_attr in node_resource_attrs))

            hop = len(p_path)
            min_bw = min(p_links_bw_list)
            max_nr = max(p_nodes_resource_list)
            p_path_rank = min_bw / (max_nr * hop + 1e-6)
            p_path_rank_values_dict[p_path] = p_path_rank
        p_path_ranks = sorted(p_path_rank_values_dict.items(), reverse=True, key=lambda x: x[1])
        sorted_p_paths = [i for i, v in p_path_ranks]
        return sorted_p_paths

    def node_mapping(self, v_net, p_net, solution):
        r"""Attempt to accommodate VNF in appropriate physical node."""
        v_net_rank = self.node_rank(v_net)
        num_neighbors_list = [len(v_net.adj[i]) for i in range(v_net.num_nodes)]
        v_bfs_root = num_neighbors_list.index(max(num_neighbors_list))
        hop_far_v_bfs_root = nx.single_source_dijkstra_path_length(v_net, v_bfs_root)
        v_ranked_value_list = []
        for v_node_id, hop_count in hop_far_v_bfs_root.items():
            v_ranked_value_list.append([v_node_id, hop_count, v_net_rank[v_node_id]])
        v_ranked_value_list.sort(key=lambda x: (x[1], -x[2]))

        sorted_v_nodes = [v_rank_values[0] for v_rank_values in v_ranked_value_list]
        sorted_v_nodes = list(v_net_rank)

        for v_node_id in sorted_v_nodes:
            selected_p_node_list = list(solution.node_slots.values())
            p_candidate_nodes = self.controller.find_candidate_nodes(v_net, p_net, v_node_id, filter=selected_p_node_list, check_node_constraint=True, check_link_constraint=False)
            if len(p_candidate_nodes) == 0:
                solution['place_result'] = False
                return False

            p_net_s_rank = self.node_rank(p_net)
            p_candidate_node_rank_values = {}
            for p_node_id in p_candidate_nodes:
                p_node_s_value = p_net_s_rank[p_node_id]
                if len(selected_p_node_list) == 0:
                    p_node_t_value = 0
                else:
                    p_net_hop_far_p_node_id = nx.single_source_dijkstra_path_length(p_net, p_node_id)
                    p_node_t_value = sum(p_net_hop_far_p_node_id[select_p_node_id] for select_p_node_id in selected_p_node_list)
                p_node_rank_value = p_node_s_value / (p_node_t_value + 1e-6)
                p_candidate_node_rank_values[p_node_id] = p_node_rank_value
            p_candidate_nodes_rank = sorted(p_candidate_node_rank_values.items(), reverse=True, key=lambda x: x[1])
            sorted_v_nodes = [i for i, v in p_candidate_nodes_rank]
            p_node_id = sorted_v_nodes[0]
            place_result, place_info= self.controller.place(v_net, p_net, v_node_id, p_node_id, solution)
            if not place_result:
                return False
        return True
