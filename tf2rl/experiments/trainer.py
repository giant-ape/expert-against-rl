import os
import time
import logging
import argparse
import csv

import numpy as np
import tensorflow as tf
from gym.spaces import Box
import glob
import random
from matplotlib import pyplot as plt
from time import sleep
from scipy.stats import norm
import sys
from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.initialize_logger import initialize_logger
from tf2rl.envs.normalizer import EmpiricalNormalizer
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Input


import shutil

# if tf.config.experimental.list_physical_devices('GPU'):
#     for cur_device in tf.config.experimental.list_physical_devices("GPU"):
#         print(cur_device)
#         tf.config.experimental.set_memory_growth(cur_device, enable=True)
# define imitation learning actor
def Actor(state_shape, action_dim, name='actor'):
    tf.random.set_seed(random.randint(1, 1000)) # reset tensorflow random seed

    obs = Input(shape=state_shape)
    conv_1 = Conv2D(16, 3, strides=3, activation='relu')(obs)
    conv_2 = Conv2D(64, 3, strides=2, activation='relu')(conv_1)
    conv_3 = Conv2D(128, 3, strides=2, activation='relu')(conv_2)
    conv_4 = Conv2D(256, 3, strides=2, activation='relu')(conv_3)
    info = GlobalAveragePooling2D()(conv_4)
    dense_1 = Dense(128, activation='relu')(info) 
    dense_2 = Dense(32, activation='relu')(dense_1)
    mean = Dense(action_dim, activation='linear')(dense_2)
    std = Dense(action_dim, activation='softplus')(dense_2)

    model = tf.keras.Model(obs, [mean, std], name=name)

    return model

# set up ensemble
state_shape = (80, 80, 9)
action_dim = 2
states = np.zeros(shape=(80, 80, 9))

# observation space
def observation_adapter(env_obs):
    global states

    new_obs = env_obs.top_down_rgb[1] / 255.0
    states[:, :, 0:3] = states[:, :, 3:6]
    states[:, :, 3:6] = states[:, :, 6:9]
    states[:, :, 6:9] = new_obs

    if env_obs.events.collisions or env_obs.events.reached_goal:
        states = np.zeros(shape=(80, 80, 9))

    return np.array(states, dtype=np.float32)

# reward function
def reward_adapter(env_obs):
    progress = env_obs.ego_vehicle_state.speed * 0.1
    goal = 1 if env_obs.events.reached_goal else 0
    crash = -1 if env_obs.events.collisions else 0

    print("success")if env_obs.events.reached_goal else 0
    return 0.01 * progress + goal + crash

# action space
def action_adapter(model_action): 
    speed = model_action[0] # output (-1, 1)
    speed = (speed - (-1)) * (13.4 - 0) / (1 - (-1)) # scale to (0, 10)
    
    speed = np.clip(speed, 0, 13.4)
    model_action[1] = np.clip(model_action[1], -1, 1)

    # discretization
    if model_action[1] < -1/3:
        lane = -1
    elif model_action[1] > 1/3:
        lane = 1
    else:
        lane = 0

    return (speed, lane)

class Trainer:
    def __init__(
            self,
            policy,
            env,
            args,
            test_env=None):
        if isinstance(args, dict):
            _args = args
            args = policy.__class__.get_argument(Trainer.get_argument())
            args = args.parse_args([])
            for k, v in _args.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                else:
                    raise ValueError(f"{k} is invalid parameter.")

        self._set_from_args(args)
        self._policy = policy
        self._env = env
        self._test_env = self._env if test_env is None else test_env

        self.agent_obs = []
        self.agent_act = []
        self.agent_rew_expert = []
        self.agent_rew_rl = []

        self.scenario = args.scenario
        self.sample = 10
        self.expert_data_folder_path = 'train_expert_data/{}'.format(self.scenario)
        self.expert_model_folder_path = '{}'.format(self.scenario)
        self.expert_model = args.prior
        self.counter = 0

        # for filename in os.listdir(self.expert_data_folder_path):
        #     file_path = os.path.join(self.expert_data_folder_path, filename)
        #     try:
        #         if os.path.isfile(file_path) or os.path.islink(file_path):
        #             os.unlink(file_path)  # 删除文件或符号链接
        #         elif os.path.isdir(file_path):
        #             shutil.rmtree(file_path)  # 递归删除子文件夹
        #     except Exception as e:
        #         print(f"删除 {file_path} 时出错: {e}")

        # for filename in os.listdir(self.expert_model_folder_path):
        #     if filename not in ['ensemble_5.h5', 'ensemble_4.h5', 'ensemble_3.h5', 'ensemble_2.h5', 'ensemble_1.h5']:
        #         file_path = os.path.join(self.expert_model_folder_path, filename)
        #         try:
        #             if os.path.isfile(file_path) or os.path.islink(file_path):
        #                 os.unlink(file_path)  # 删除文件或符号链接
        #             elif os.path.isdir(file_path):
        #                 shutil.rmtree(file_path)  # 递归删除子文件夹
        #         except Exception as e:
        #             print(f"删除 {file_path} 时出错: {e}")

        if self._normalize_obs:
            assert isinstance(env.observation_space, Box)
            self._obs_normalizer = EmpiricalNormalizer(shape=env.observation_space.shape)

        # prepare log directory
        self._output_dir = prepare_output_dir(
            args=args, user_specified_dir=self._logdir,
            suffix="{}_{}".format(self._policy.policy_name, args.dir_suffix))
        self.logger = initialize_logger(
            logging_level=logging.getLevelName(args.logging_level),
            output_dir=self._output_dir)
        if not os.path.exists('{}/Model'.format(self._logdir)):
            os.makedirs('{}/Model'.format(self._logdir))

        # create training log
        with open(self._output_dir + '/training_log.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['episode', 'step', 'episodic reward', 'success rate', 'episodic length'])

        # if evaluate the model
        if args.evaluate:
            assert args.model_dir is not None
        self._set_check_point(args.model_dir)

        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self._output_dir)
        self.writer.set_as_default()


    def _set_check_point(self, model_dir):
        # Save and restore model
        self._checkpoint = tf.train.Checkpoint(policy=self._policy)
        self.checkpoint_manager = tf.train.CheckpointManager(self._checkpoint, directory=self._output_dir, max_to_keep=5)

        if model_dir is not None:
            assert os.path.isdir(model_dir)
            self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            self._checkpoint.restore(self._latest_path_ckpt)
            self.logger.info("Restored {}".format(self._latest_path_ckpt))

    def __call__(self):
        total_steps = 0
        tf.summary.experimental.set_step(total_steps)
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()
        n_episode = 0
        episode_returns = []
        success_log = [0]
        best_train = -np.inf

        replay_buffer = get_replay_buffer(
            self._policy, self._env, self._use_prioritized_rb,
            self._use_nstep_rb, self._n_step)

        obs = self._env.reset()
        env_obs = obs[self._env.agent_id]
        obs = observation_adapter(env_obs)
        
        while total_steps < self._max_steps:
            if total_steps < self._policy.n_warmup:
                action = self._env.action_space.sample()
            else:
                action = self._policy.get_action(obs)
            if np.isnan(action[0]) or np.isnan(action[1]):
                total_steps = 100000
                break
            env_action = action_adapter(action)


            rl_av, expert_av = self._policy.get_action_value(obs, action)
            # print(env_action)
            next_obs, reward, done, info = self._env.step({self._env.agent_id: env_action})
            next_env_obs = next_obs[self._env.agent_id]
            next_obs = observation_adapter(next_env_obs)
            # reward = reward[self._env.agent_id]
            reward = reward_adapter(next_env_obs)
            done = done[self._env.agent_id]
            info = info[self._env.agent_id]
            
            if self._show_progress:
                obs_tensor = tf.expand_dims(obs, axis=0)
                # agent distribution
                agent_dist = self._policy.actor._compute_dist(obs_tensor)

            episode_steps += 1
            episode_return += reward
            total_steps += 1
            tf.summary.experimental.set_step(total_steps)

            # if the episode is finished
            done_flag = done
            if (hasattr(self._env, "_max_episode_steps") and
                episode_steps == self._env._max_episode_steps):
                done_flag = False
            
            replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done_flag)

            self.agent_obs.append(states.astype(np.float32))
            self.agent_act.append(action.astype(np.float32))
            self.agent_rew_expert.append(expert_av.astype(np.float32))
            self.agent_rew_rl.append(rl_av.astype(np.float32))

            obs = next_obs

            # add to training log
            if total_steps % 5 == 0:
                success = np.sum(success_log[-20:]) / 20 
                with open(self._output_dir + '/training_log.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([n_episode, total_steps, episode_returns[n_episode-1] if episode_returns else -1, success, episode_steps])

            # end of a episode
            if done or episode_steps == self._episode_max_steps:
                Trainer.training_expert(self, total_steps, info)
                # if task is successful
                success_log.append(1 if info['env_obs'].events.reached_goal else 0)

                # reset env
                replay_buffer.on_episode_end()
                obs = self._env.reset()
                env_obs = obs[self._env.agent_id]
                obs = observation_adapter(env_obs)
                
                # display info
                n_episode += 1
                fps = episode_steps / (time.perf_counter() - episode_start_time)
                success = np.sum(success_log[-20:]) / 20

                self.logger.info("Total Episode: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                    n_episode, total_steps, episode_steps, episode_return, fps))

                tf.summary.scalar(name="Common/training_return", data=episode_return)
                tf.summary.scalar(name='Common/training_success', data=success)
                tf.summary.scalar(name="Common/training_episode_length", data=episode_steps)

                # reset variables
                episode_returns.append(episode_return)
                episode_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter()

                # save policy model
                if n_episode > 20 and np.mean(episode_returns[-20:]) >= best_train:
                    best_train = np.mean(episode_returns[-20:])
                    self._policy.actor.network.save('{}/Model/Model_{}_{:.4f}.h5'.format(self._logdir, n_episode, best_train))

            if total_steps < self._policy.n_warmup:
                continue

            if total_steps % self._policy.update_interval == 0:
                samples = replay_buffer.sample(self._policy.batch_size)
                
                with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                    self._policy.step = total_steps

                    self._policy.train(
                        samples["obs"], samples["act"], samples["next_obs"],
                        samples["rew"], np.array(samples["done"], dtype=np.float32),
                        None if not self._use_prioritized_rb else samples["weights"])

                if self._use_prioritized_rb:
                    td_error = self._policy.compute_td_error(
                        samples["obs"], samples["act"], samples["next_obs"],
                        samples["rew"], np.array(samples["done"], dtype=np.float32))
                    replay_buffer.update_priorities(samples["indexes"], np.abs(td_error) + 1e-6)
                    tf.summary.scalar(name=self._policy.policy_name + "/td_error", data=tf.reduce_mean(td_error))

            if total_steps % self._test_interval == 0:
                avg_test_return, avg_test_steps = self.evaluate_policy(total_steps)
                self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                    total_steps, avg_test_return, self._test_episodes))

                tf.summary.scalar(name="Common/average_test_return", data=avg_test_return)
                tf.summary.scalar(name="Common/average_test_episode_length", data=avg_test_steps)
                tf.summary.scalar(name="Common/fps", data=fps)
                self.writer.flush()
                
                # reset env
                obs = self._env.reset()
                env_obs = obs[self._env.agent_id]
                obs = observation_adapter(env_obs)
                episode_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter()

            # save checkpoint
            if total_steps % self._save_model_interval == 0:
                self.checkpoint_manager.save()

        tf.summary.flush()

    def evaluate_policy_continuously(self):
        """
        Periodically search the latest checkpoint, and keep evaluating with the latest model until user kills process.
        """
        if self._model_dir is None:
            self.logger.error("Please specify model directory by passing command line argument `--model-dir`")
            exit(-1)

        self.evaluate_policy(total_steps=0)
        while True:
            latest_path_ckpt = tf.train.latest_checkpoint(self._model_dir)
            if self._latest_path_ckpt != latest_path_ckpt:
                self._latest_path_ckpt = latest_path_ckpt
                self._checkpoint.restore(self._latest_path_ckpt)
                self.logger.info("Restored {}".format(self._latest_path_ckpt))
            self.evaluate_policy(total_steps=0)

    def evaluate_policy(self, total_steps):
        tf.summary.experimental.set_step(total_steps)
        if self._normalize_obs:
            self._test_env.normalizer.set_params(*self._env.normalizer.get_params())

        avg_test_return = 0.
        avg_test_steps = 0

        if self._save_test_path:
            replay_buffer = get_replay_buffer(self._policy, self._test_env, size=self._episode_max_steps)

        for i in range(self._test_episodes):
            episode_return = 0.
            obs = self._test_env.reset()
            env_obs = obs[self._env.agent_id]
            obs = observation_adapter(env_obs)
            avg_test_steps += 1

            for _ in range(self._episode_max_steps):
                action = self._policy.get_action(obs, test=True)
                env_action = action_adapter(action)
                next_obs, reward, done, _ = self._test_env.step({self._test_env.agent_id: env_action})
                next_env_obs = next_obs[self._test_env.agent_id]
                next_obs = observation_adapter(next_env_obs)
                # reward = reward[self._test_env.agent_id]
                reward = reward_adapter(next_env_obs)
                done = done[self._test_env.agent_id]

                avg_test_steps += 1
                if self._save_test_path:
                    replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done)

                episode_return += reward
                obs = next_obs
                
                if done:
                    break

            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(total_steps, i, episode_return)
            avg_test_return += episode_return

        return avg_test_return / self._test_episodes, avg_test_steps / self._test_episodes

    def _set_from_args(self, args):
        # experiment settings
        self._max_steps = args.max_steps
        self._episode_max_steps = (args.episode_max_steps
                                   if args.episode_max_steps is not None else args.max_steps)
        self._n_experiments = args.n_experiments
        self._show_progress = args.show_progress
        self._save_model_interval = args.save_model_interval
        self._save_summary_interval = args.save_summary_interval
        self._normalize_obs = args.normalize_obs
        self._logdir = args.logdir
        self._model_dir = args.model_dir
        # replay buffer
        self._use_prioritized_rb = args.use_prioritized_rb
        self._use_nstep_rb = args.use_nstep_rb
        self._n_step = args.n_step
        # test settings
        self._test_interval = args.test_interval
        self._show_test_progress = args.show_test_progress
        self._test_episodes = args.test_episodes
        self._save_test_path = args.save_test_path
        self._save_test_movie = args.save_test_movie
        self._show_test_images = args.show_test_images

    @staticmethod
    def get_argument(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        # experiment settings
        parser.add_argument('--max-steps', type=int, default=int(1e6),
                            help='Maximum number steps to interact with env.')
        parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
                            help='Maximum steps in an episode')
        parser.add_argument('--n-experiments', type=int, default=1,
                            help='Number of experiments')
        parser.add_argument('--show-progress', action='store_true',
                            help='Call `render` in training process')
        parser.add_argument('--save-model-interval', type=int, default=int(5e4),
                            help='Interval to save model')
        parser.add_argument('--save-summary-interval', type=int, default=int(1e3),
                            help='Interval to save summary')
        parser.add_argument('--model-dir', type=str, default=None,
                            help='Directory to restore model')
        parser.add_argument('--dir-suffix', type=str, default='',
                            help='Suffix for directory that contains results')
        parser.add_argument('--normalize-obs', action='store_true', default=False,
                            help='Normalize observation')
        parser.add_argument('--logdir', type=str, default='results',
                            help='Output directory')
        # test settings
        parser.add_argument('--evaluate', action='store_true',
                            help='Evaluate trained model')
        parser.add_argument('--test-interval', type=int, default=int(20e4),
                            help='Interval to evaluate trained model')
        parser.add_argument('--show-test-progress', action='store_true',
                            help='Call `render` in evaluation process')
        parser.add_argument('--test-episodes', type=int, default=20,
                            help='Number of episodes to evaluate at once')
        parser.add_argument('--save-test-path', action='store_true',
                            help='Save trajectories of evaluation')
        parser.add_argument('--show-test-images', action='store_true',
                            help='Show input images to neural networks when an episode finishes')
        parser.add_argument('--save-test-movie', action='store_true',
                            help='Save rendering results')
        # replay buffer
        parser.add_argument('--use-prioritized-rb', action='store_true',
                            help='Flag to use prioritized experience replay')
        parser.add_argument('--use-nstep-rb', action='store_true',
                            help='Flag to use nstep experience replay')
        parser.add_argument('--n-step', type=int, default=4,
                            help='Number of steps to look over')
        # others
        parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'WARNING'],
                            default='INFO', help='Logging level')
        return parser


    def training_expert(self, total_steps, info):


        return_expert = np.mean(self.agent_rew_expert)
        return_rl = np.mean(self.agent_rew_rl)
        print("agent_rew_expert",self.agent_rew_expert)
        print("agent_rew_rl",self.agent_rew_rl)

        print("return_expert",return_expert)
        print("return_rl",return_rl)
        print("success")if info['env_obs'].events.reached_goal else 0
        if info['env_obs'].events.reached_goal and return_rl > return_expert:
            np.savez('train_expert_data/{}/demo_{}.npz'.format(self.scenario, self.counter), 
                    obs=np.array(self.agent_obs, dtype=np.float32), act=np.array(self.agent_act, dtype=np.float32))
            self.counter += 1
        self.agent_obs = []
        self.agent_act = []
        self.agent_rew_expert = []
        self.agent_rew_rl = []

        files = glob.glob(os.path.join(self.expert_data_folder_path, '*'))
        file_count = len(files)

        if file_count > self.sample:
            OBS = []
            ACT = []
            files = glob.glob(self.expert_data_folder_path + '/*.npz')
            for file in files:
                obs = np.load(file)['obs']
                act = np.load(file)['act']
                for i in range(obs.shape[0]):
                    OBS.append(obs[i])
                    act[i, 0] += random.normalvariate(0, 0.1) # add a small noise to speed
                    act[i, 0] = np.clip(act[i, 0], 0, 13.4) 
                    act[i, 0] = 2.0 * ((act[i, 0] - 0) / (13.4 - 0)) - 1.0 # normalize speed 
                    act[i, 1] += random.normalvariate(0, 0.1) # add a small noise to lane change, which does not affect the decision
                    ACT.append(act[i])
            OBS = np.array(OBS, dtype=np.float32)
            ACT = np.array(ACT, dtype=np.float32)

            # model training
            epochs = 100
            EPS = 1e-6
            
            idx = len(glob.glob(os.path.join(self.expert_model_folder_path, '*')))

            model = Actor(state_shape, action_dim, name='prior_{}'.format(idx))

            print('===== Training Ensemble Model {} ====='.format(idx+1))
            tf.random.set_seed(random.randint(1, 1000)) # reset tensorflow random seed
            np.random.seed(random.randint(1, 1000)) # reset numpy random seed
            optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4) # set up optimizer

            # create dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((OBS, ACT))
            train_dataset = train_dataset.shuffle(OBS.shape[0]).batch(32)
            train_loss_results = []
            
            # start training
            for epoch in range(epochs):
                epoch_loss = []
                for x, y in train_dataset:
                    with tf.GradientTape() as tape:
                        mean, std = model(x)
                        var = tf.square(std)
                        loss_value = 0.5 * tf.reduce_mean(tf.math.log(var + EPS) + tf.math.square(y - mean)/(var + EPS)) 

                    epoch_loss.append(loss_value.numpy())
                    gradients = tape.gradient(loss_value, model.trainable_weights)
                    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                
                train_loss_results.append(np.mean(epoch_loss))
                sys.stdout.write("Progress: {}/{}, Loss: {:.6f}\n".format(epoch+1, epochs, np.mean(epoch_loss)))
                sys.stdout.flush()
            
            # save trained model
            model.save('{}/ensemble_{}.h5'.format(self.expert_model_folder_path, idx+1))

            for filename in os.listdir(self.expert_data_folder_path):
                file_path = os.path.join(self.expert_data_folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # 删除文件或符号链接
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # 递归删除子文件夹
                except Exception as e:
                    print(f"删除 {file_path} 时出错: {e}")
            self._policy.actor._expert_ensemble = [load_model(model) for model in glob.glob(self.expert_model+'/ensemble*.h5')]