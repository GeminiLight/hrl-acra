from .environment import SolutionStepEnvironment
from solver.heuristic.node_rank import *


def load_simulator(solver_name):
    # rank
    if solver_name == 'grc_rank':
        Env, Solver = SolutionStepEnvironment, GRCRankSolver
    elif solver_name == 'nrm_rank':
        Env, Solver = SolutionStepEnvironment, NRMRankSolver
    elif solver_name == 'pl_rank':
        Env, Solver = SolutionStepEnvironment, PLRankSolver
    elif solver_name == 'gae_vne':
        from solver.learning.gae_vne import GAESolver
        Env, Solver = SolutionStepEnvironment, GAESolver
    elif solver_name == 'mcts_vne':
        from solver.learning.mcts_vne import MCTSSolver
        Env, Solver = SolutionStepEnvironment, MCTSSolver
    elif solver_name == 'pg_cnn2':
        from solver.learning.pg_cnn2 import PgCnn2Solver
        Env, Solver = SolutionStepEnvironment, PgCnn2Solver
    elif solver_name == 'a3c_gcn':
        from solver.learning.a3c_gcn import A3CGCNSolver
        Env, Solver = SolutionStepEnvironment, A3CGCNSolver
    elif solver_name == 'hrl_ra':
        from solver.learning.hrl_ra import HrlRaSolver
        Env, Solver = SolutionStepEnvironment, HrlRaSolver
    elif solver_name == 'hrl_ac':
        from solver.learning.hrl_ac import OnlineEnv, HrlAcSolver
        Env, Solver = OnlineEnv, HrlAcSolver
    else:
        raise ValueError('The solver is not yet supported; \n Please attempt to select another one.', solver_name)
    return Env, Solver
