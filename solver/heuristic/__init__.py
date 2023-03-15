from .node_rank import NodeRankSolver, GRCRankSolver, OrderRankSolver, NRMRankSolver

from base.environment import *
from base.register import Register


__all__ = [
    NodeRankSolver, 
    GRCRankSolver, 
    OrderRankSolver, 
    NRMRankSolver,
]

Register.register('grc_rank', {'solver': GRCRankSolver, 'env': SolutionStepEnvironment})
Register.register('order_rank', {'solver': OrderRankSolver, 'env': SolutionStepEnvironment})
Register.register('nrm_rank', {'solver': NRMRankSolver, 'env': SolutionStepEnvironment})
