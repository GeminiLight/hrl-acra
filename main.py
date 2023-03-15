# os.chdir(os.path.join(os.getcwd(), 'code/virne-dev'))
import os
from config import get_config, show_config, save_config, load_config
from base.loader import load_simulator
from data.generator import Generator
from base import BasicScenario, SolverLibrary


def run(config):
    print(f"\n{'-' * 20}    Start     {'-' * 20}\n")

    print(f'Use {config.solver_name} Solver...\n')
    # Load environment and algorithm
    Env, Solver = load_simulator(config.solver_name)
    scenario = BasicScenario.from_config(Env, Solver, config)
    scenario.run()

    print(f"\n{'-' * 20}   Complete   {'-' * 20}\n")


if __name__ == '__main__':
    ## -- available solver -- ##
    # heuritics_solver_name_list = ['nrm_rank', 'grc_rank', 'pl_rank']
    # learning_based_solver_name_list = ['mcts_vne', 'gae_vne', 'a3c_gcn', '']
    # our_proposed_solver_name_list = ['hrl_ra', 'hrl_ac']

    # Simulation
    config = get_config()
    run(config)
