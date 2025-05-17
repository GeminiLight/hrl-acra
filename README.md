# Code for HRL-ACRA

> [!IMPORTANT]
> :sparkles: The foundation model of HRL-ACRA has been integrated into our benchmark [Virne](https://github.com/GeminiLight/virne)
> 
> :sparkles: You can easily run it with a specific solver name `ppo_gat_seq2seq+` in Virne!

This is the implementation of our paper, "[Joint Admission Control and Resource Allocation of Virtual Network Embedding via Hierarchical Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/10291038)", accepted by IEEE Transactions on Services Computing (TSC).

## Installation

It is suggested to enable the GPU to accelerate the training and inference process.

### With GPU

```bash
sh install.sh -c=11.3
```

Here, `c` donates the version of CUDA, and `10.2` or `11.3 ` is recommended.

### With CPU

```bash
sh install.sh
```

## Quick Start

### Run HRL-ACRA

#### 1. Pretrain Lower-level Agent

```bash
python main.py \
--solver_name="hrl_ra" \
--eval_interval=10 \
--num_train_epochs=100 \
--summary_file_name="exp-wx_100-hrl_ra-training.csv" \
--seed=0
```

#### 2. Train Upper-level Agent

```bash
python main.py \
--solver_name="hrl_ac" \
--sub_solver_name="hrl_ra" \
--eval_interval=10 \
--num_train_epochs=500 \
--summary_file_name="exp-wx_100-hrl_ac-training.csv" \
--pretrained_subsolver_model_path=$PretrainedHrlRaModelPath \
--seed=0
```

Here, you should specify `pretrained_sub_model_path` as the file path of pretrain lower-level agent model.

#### 3. Test HRL-ACRA

```bash
python main.py \
--solver_name="hrl_ac" \
--sub_solver_name="hrl_ra"
--decode_strategy="beam" \
--k_searching=1 \
--num_train_epochs=0 \
--pretrained_model_path=$PretrainedHrlAcModelPath \
--pretrained_subsolver_model_path=$PretrainedHrlraModelPath \
--summary_file_name="exp-wx_100-hrl_acra-testing.csv" \
--seed=0
```

Here, you should specify the `pretrained_model_path` as  the file path of pretrain upper-level agent model  and  `pretrained_sub_model_path` as the file path of pretrain lower-level agent model. `k_searching` donates the searching width of beam search. When it is set to 1, it degenerates into a greedy search. When its value is larger, the quality of the obtained solution is often higher. You can determine its value according to the configuration of your computer.

### Run Baselines

There are baseline solvers divided into two categories, learning-based and heuristic. 

|  Baseline Name   | Solver NAME  | Category | Need to Pretrain |
|  ----    | ----  | ---- | ---- |
|  GRC     | grc_rank | heuristic | False |
|  NRM     | nrm_rank | heuristic | False |
|  PL      | pl_rank | heuristic | False |
|  MCTS    | mct_vne | learning-based | False |
|  GAE_BFS | gae_bfs | learning-based | False |
|  A3C-GCN | a3c_gcn | learning-based | True |
|  PG-CNN  | pg_cnn | learning-based | True |


#### Run Baselines That Not Need to Pretrain

For heuristic solvers, they are learning-free for using fixed strategies. You can specify the `SOLVER_NAME` with the `solver_name` of heuristic baselines.

```bash
python main.py \
--solver_name=$SOLVER_NAME \
--summary_file_name="exp_wx_100-baselines.csv" \
--seed=0 
```

Here, you can replace `$SOLVER_NAME` with `grc_rank`, `nrm_rank`, `pl_rank`, `mcts_vne`, and `gae_vne`.

#### Run Baselines That Need to Pretrain

##### Train Baselines

```bash
python main.py \
--solver_name=$SOLVER_NAME \
--eval_interval=10 \
--num_train_epochs=100 \
--summary_file_name="exp_wx_100-baselines-training.csv" \
--seed=0
```

Here, you can replace `$SOLVER_NAME` with `pg_cnn2` and `a3c_gcn`.


##### Test Baseline

```bash
python main.py \
--solver_name=$SOLVER_NAME \
--sub_solver_name="hrl_ra" \
--num_train_epochs=0 \
--summary_file_name="exp-wx_100-baselines-testing.csv" \
--pretrained_model_path=$PretrainedBaselineModelPath
--seed=0 \
```

Here, you should specify `pretrained_sub_model_path` as the file path of pretrain baseline agent model.


## File Structure

```
.
|____settings ------------------------------- simulation settings 
|____config.py ------------------------------ config varibles
|____dataset -------------------------------- pre-generate dataset
|____install.sh ----------------------------- installation shell
|____solver --------------------------------- various VNR solvers
| |____solver.py ---------------------------- basic class of solver 
| |____learning
| | |____a3c_gcn ---------------------------- baseline solver: a3c-gcn
| | |____pg_cnn2 ---------------------------- baseline solver: pg-cnn
| | |____gae_vne ---------------------------- baseline solver: gae-bfs
| | |____hrl_ac ----------------------------- our upper-level agent for adimission control
| | |____hrl_ra ----------------------------- our lower-level agent for resource allocation
| | |____mcts_vne --------------------------- baseline solver: mcts
| | |____utils.py
| | |____obs_handler.py --------------------- obtain observation from environment
| | |____buffer.py -------------------------- reinforcement learning buffer
| | |____sub_rl_environment.py -------------- lower-level environment for resource allocation
| | |____rl_environment.py ------------------ upper-level environment for adimission control
| |____heuristic
|   |____node_rank.py ----------------------- baseline solvers: grc, nrm, pl
|____README.md
|____.gitignore
|____data
| |____attribute.py
| |______init__.py
| |____physical_network.py ------------------ physical network
| |____virtual_network.py ------------------- virtual network
| |____generator.py ------------------------- randomly generation
| |____utils.py
| |____network.py --------------------------- basic class of network
| |____virtual_network_request_simulator.py - vnr simulator
|____base
| |____controller.py ------------------------ basic class of network
| |____register.py -------------------------- solver register
| |____solution.py -------------------------- solution class
| |____recorder.py -------------------------- running logger
| |____utils.py
| |____loader.py ---------------------------- solver loader
| |____scenario.py -------------------------- simulation scenario
| |____environment.py ----------------------- basic class of Internet provider
| |____counter.py --------------------------- recoder statistic
```


## Various Simulation Settings

### Physical Network

The simulation settings file of physical network is placed at `./settings/p_net_setting.yaml`.

#### Default Settings

```yaml
num_nodes: 100
save_dir: dataset/p_net
topology:
  type: waxman
  wm_alpha: 0.5
  wm_beta: 0.2
link_attrs_setting:
  - distribution: uniform
    dtype: int
    generative: true
    high: 100
    low: 50
    name: bw
    owner: link
    type: resource
  - name: max_bw
    originator: bw
    owner: link
    type: extrema
node_attrs_setting:
  - name: cpu
    distribution: uniform
    dtype: int
    generative: true
    high: 100
    low: 50
    owner: node
    type: resource
  - name: max_cpu
    originator: cpu
    owner: node
    type: extrema
file_name: p_net.gml
```

#### Real Topologies Validation 

To conduct validation on real topologies, please replace the `topology` with following parameters.

- Brain Topology

```yaml
topology:
  file_path: './dataset/topology/Brain.gml'
```

- Geant Topology

```yaml
topology:
  file_path: './dataset/topology/Geant.gml'
```

### Virtual Network Requests

The simulation settings file of Virtual Network Requests is placed at `./settings/v_sim_setting.yaml`.

#### Default Settings

```yaml
num_v_nets: 1000
topology:
  random_prob: 0.5
  type: random
v_net_size:
  distribution: uniform
  dtype: int
  low: 2
  high: 10
arrival_rate:
  distribution: possion
  dtype: float
  lam: 0.04
  reciprocal: true
lifetime:
  distribution: exponential
  dtype: float
  scale: 1000
node_attrs_setting:
  - name: cpu
    distribution: uniform
    dtype: int
    generative: true
    low: 0
    high: 50
    owner: node
    type: resource
link_attrs_setting:
  - name: bw
    distribution: uniform
    dtype: int
    generative: true
    low: 0
    high: 50
    owner: link
    type: resource
save_dir: dataset/v_nets
v_nets_file_name: v_net.gml
v_nets_save_dir: v_nets
events_file_name: events.yaml
setting_file_name: v_sim_setting.yaml

```

#### Arrival Rate Test

To simulate various arrival rates of VNRs, please update the field `arrival_rate -> lam`.

```yaml
arrival_rate:
  distribution: possion
  dtype: float
  lam: 0.04  # replace with 0.06 or 0.08
  reciprocal: true
```

#### Node Size Test

To simulate various node sizes of VNRs, please update the field `node_size -> high`.

```yaml
v_net_size:
  distribution: uniform
  dtype: int
  low: 2
  high: 10   # replace with 15 or 20
```

#### Resource Request Test

To simulate various arrival rate of VNRs, please update both the field `node_attrs_setting -> [0] -> high` and the field `link_attrs_setting -> [0] -> high`.

```yaml
node_attrs_setting:
  - name: cpu
    distribution: uniform
    dtype: int
    generative: true
    low: 0
    high: 50  # replace with 70 or 90
    owner: node
    type: resource
link_attrs_setting:
  - name: bw
    distribution: uniform
    dtype: int
    generative: true
    low: 0
    high: 50  # replace with 70 or 90
    owner: link
    type: resource
```

## Citation

If you find this code useful, please cite our paper:

```
@ARTICLE{tfwang-tsc-2024-hrl-acra,
  author={Wang, Tianfu and Shen, Li and Fan, Qilin and Xu, Tong and Liu, Tongliang and Xiong, Hui},
  journal={IEEE Transactions on Services Computing}, 
  title={Joint Admission Control and Resource Allocation of Virtual Network Embedding via Hierarchical Deep Reinforcement Learning}, 
  year={2024},
  volume={17},
  number={3},
  pages={1001-1015},
}
```

## Related Resources

- [Virne](https://github.com/GeminiLight/virne) is a simulator for resource allocation problems in network virtualization.
- [SDN-NFV-Papers](https://github.com/GeminiLight/sdn-nfv-papers) is a paper list about Resource Allocation in Network Functions Virtualization (NFV) and Software-Defined Networking (SDN).
