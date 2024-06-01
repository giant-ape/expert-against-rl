import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
from tf2rl.algos.sac import SAC
from tf2rl.algos.expert_prior import ExpertPrior
from tf2rl.algos.expert_against import ExpertAgainst
from tf2rl.algos.expert_against2 import ExpertAgainst2
from tf2rl.algos.expert_against3 import ExpertAgainst3
from tf2rl.algos.expert_against4 import ExpertAgainst4
from tf2rl.algos.ppo import PPO
from tf2rl.algos.gail import GAIL

from tf2rl.experiments.trainer import Trainer

from tf2rl.experiments.exit_trainer import EXIT_Trainer

from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.experiments.irl_trainer import IRLTrainer

import tensorflow as tf
import gym
import numpy as np
import glob
import random
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
import shutil
from smarts.env.hiway_env import HiWayEnv
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB
from smarts.core.controllers import ActionSpaceType

#### Load expert trajectories ####
def load_expert_trajectories(filepath):
    filenames = glob.glob(filepath)

    trajectories = []
    for filename in filenames:
        trajectories.append(np.load(filename))

    obses = []
    next_obses = []
    actions = []
    
    for trajectory in trajectories:
        obs = trajectory['obs']
        action = trajectory['act']

        for i in range(obs.shape[0]-1):
            obses.append(obs[i])
            next_obses.append(obs[i+1])
            act = action[i]
            act[0] += random.normalvariate(0, 0.1) # speed
            act[0] = np.clip(act[0], 0, 10)
            act[0] = 2.0 * ((act[0] - 0) / (10 - 0)) - 1.0 # normalize speed
            act[1] += random.normalvariate(0, 0.1) # lane change
            act[1] = np.clip(act[1], -1, 1)
            actions.append(act)
    
    expert_trajs = {'obses': np.array(obses, dtype=np.float32),
                    'next_obses': np.array(next_obses, dtype=np.float32),
                    'actions': np.array(actions, dtype=np.float32)}

    return expert_trajs

#### Environment specs ####
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(80, 80, 9))
AGENT_ID = 'Agent-007'
states = np.zeros(shape=(80, 80, 9))


#### RL training ####r
parser = Trainer.get_argument()
parser.add_argument("--algo", help="algorithm to run", default='expert_against2')
parser.add_argument("--scenario", help="scenario to run", default='left_lc')
parser.add_argument("--prior", help="path to the EXIT_xpert prior models", default='/home/whut613/zjy/Expert-Prior-RL/left_lc')
args = parser.parse_args()
args.max_steps = 10e4
args.save_summary_interval = 128
args.use_prioritized_rb = False
args.n_experiments = 10
args.logdir = f'./train_results/{args.scenario}/{args.algo}'

# define scenario
if args.scenario == 'left':
    scenario_path = ['scenarios/left']
    max_episode_steps = 400
elif args.scenario == 'straight':
    scenario_path = ["scenarios/straight"]
    max_episode_steps = 600
elif args.scenario == 'exit_left':
    scenario_path = ['scenarios/exit_left']
    max_episode_steps = 600
elif args.scenario == 'left_lc':
    scenario_path = ['scenarios/left_lc']
    max_episode_steps = 400
elif args.scenario == 'roundabout':
    scenario_path = ['scenarios/roundabout']
    max_episode_steps = 600
else:
    raise NotImplementedError

# define agent interface
agent_interface = AgentInterface(
    max_episode_steps=max_episode_steps,
    waypoint_paths=True,
    neighborhood_vehicle_states=NeighborhoodVehicles(radius=60),
    top_down_rgb=RGB(80, 80, 32/80),
    action=ActionSpaceType.LaneWithContinuousSpeed,
)

# define agent specs
agent_spec = AgentSpec(
    interface=agent_interface
)



if args.algo == 'gail':
    expert_trajs = load_expert_trajectories(args.prior+'/*.npz')

for i in range(args.n_experiments):
    print(f'Progress: {i+1}/{args.n_experiments}')

    # create env
    env = HiWayEnv(scenarios=scenario_path, agent_specs={AGENT_ID: agent_spec}, headless=True, seed=i)
    env.observation_space = OBSERVATION_SPACE
    env.action_space = ACTION_SPACE
    env.agent_id = AGENT_ID

    expert_data_folder_path = 'train_expert_data/{}'.format(args.scenario)
    expert_model_folder_path = '{}'.format(args.scenario)
    expert_model = args.prior
    counter = 0

    for filename in os.listdir(expert_data_folder_path):
        file_path = os.path.join(expert_data_folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 递归删除子文件夹
        except Exception as e:
            print(f"删除 {file_path} 时出错: {e}")

    for filename in os.listdir(expert_model_folder_path):
        if filename not in ['ensemble_5.h5', 'ensemble_4.h5', 'ensemble_3.h5', 'ensemble_2.h5', 'ensemble_1.h5']:
            file_path = os.path.join(expert_model_folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 递归删除子文件夹
            except Exception as e:
                print(f"删除 {file_path} 时出错: {e}")


    if args.algo == 'sac':
        if args.scenario == 'exit_left':
            policy = SAC(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0], 
                        auto_alpha=True, memory_capacity=int(2e4), batch_size=32, n_warmup=5000)
            trainer =  EXIT_Trainer(policy, env, args)
        else:
            policy = SAC(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0], 
                        auto_alpha=True, memory_capacity=int(2e4), batch_size=32, n_warmup=5000)
            trainer = Trainer(policy, env, args)

    elif args.algo == 'value_penalty':
        policy = ExpertPrior(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0], 
                             prior=args.prior, auto_alpha=False, alpha=0.2, memory_capacity=int(2e4), batch_size=32, n_warmup=5000)
        trainer = Trainer(policy, env, args)

    elif args.algo == 'policy_constraint':
        policy = ExpertPrior(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0], 
                             prior=args.prior, auto_alpha=True, epsilon=0.2, memory_capacity=int(2e4), batch_size=32, n_warmup=5000)
        trainer = Trainer(policy, env, args)

    elif args.algo == 'expert_against':
        policy = ExpertAgainst(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0], 
                               prior=args.prior, auto_alpha=True, epsilon=0.2,memory_capacity=int(2e4), batch_size=32, n_warmup=5000)
        trainer = Trainer(policy, env, args)
    elif args.algo == 'expert_against2':
        policy = ExpertAgainst2(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0], 
                               prior=args.prior, auto_alpha=True, epsilon=0.2, memory_capacity=int(2e4), batch_size=32, n_warmup=5000)
        trainer = Trainer(policy, env, args)
    elif args.algo == 'expert_against4':
        policy = ExpertAgainst4(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0], 
                               prior=args.prior, auto_alpha=True, epsilon=0.2, memory_capacity=int(2e4), batch_size=32, n_warmup=5000)
        trainer = Trainer(policy, env, args)    
    elif args.algo == 'ppo':
        policy = PPO(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0],
                     batch_size=32, clip_ratio=0.2, n_epoch=10, entropy_coef=0.01, horizon=512)
        trainer = OnPolicyTrainer(policy, env, args)

    elif args.algo == 'gail':
        policy = PPO(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0],
                     batch_size=32, clip_ratio=0.2, n_epoch=10, entropy_coef=0.01, horizon=512)
        irl = GAIL(state_shape=env.observation_space.shape, action_dim=env.action_space.high.size, batch_size=32, n_training=1)
        trainer = IRLTrainer(policy, env, args, irl, expert_trajs["obses"], expert_trajs["next_obses"], expert_trajs["actions"])

    else:
        raise NotImplementedError

    # begin training
    trainer()

    # close env
    env.close()
