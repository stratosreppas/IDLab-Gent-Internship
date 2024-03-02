#Q: how to run it
#A: python run.py --alg ppo --use_case redis --goal cost --training
#Q: how to run it for testing
#A: python run.py --alg ppo --use_case redis --goal cost --testing --load_path logs/ppo_env_redis_goal_cost_k8s_False_totalSteps_100000_pred_1_ts_1.zip
#Q: how did you know about the pred and ts in the name it is in another file
#A: it is in the run.py file
#Q: in what line
#A: line 50
#Q: show me the contents of the line
#A: prediction = args.prediction
#     timeseries = args.timeseries  # 1: 1 step, 2: 2 steps, 3: 3 steps
#Q: show me the path of the file that contains that line


import logging
import argparse
import time

import torch
import pandas as pd

import numpy as np
import tqdm
from matplotlib import pyplot as plt


from setuptools.command.alias import alias
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor, SubprocVecEnv

from gym_hpa.envs import Redis, OnlineBoutique
from deepsets.ppo.ppo import DSPPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Logging
from policies.util.util import test_model

logging.basicConfig(filename='run.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description='Run ILP!')
parser.add_argument('--alg', default='ds', help='The algorithm: ["ppo", "recurrent_ppo", "a2c", "ds"]')
parser.add_argument('--k8s', default=False, action="store_true", help='K8s mode')
parser.add_argument('--use_case', default='online_boutique', help='Apps: ["redis", "online_boutique"]')
parser.add_argument('--goal', default='cost', help='Reward Goal: ["cost", "latency"]')

parser.add_argument('--training', default=False, action="store_true", help='Training mode')
parser.add_argument('--testing', default=True, action="store_true", help='Testing mode')
parser.add_argument('--loading', default=False, action="store_true", help='Loading mode')
parser.add_argument('--load_path', default='', help='Loading path, ex: logs/model/test.zip')
parser.add_argument('--test_path', default='models/ds_tr_redis_200k_cost_4_obs_new_sim/training_1/steps/150400', help='Testing path, ex: logs/model/test.zip')

parser.add_argument('--steps', default=500, help='The steps for saving.')
parser.add_argument('--total_steps', default=200000, help='The total number of steps.')

args = parser.parse_args()

SEED = 2
MONITOR_PATH = f"./results/ppo_deepset_{SEED}_plot.monitor.csv"

def get_model(alg, env, tensorboard_log, args):
    model = 0
    if alg == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
    elif alg == 'a2c':
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)  # , n_steps=steps
    elif alg == 'ds':
        ###This section creates the ds environments. We are using SubprocVecEnv for parallel computing to
        ### accelerate the learning process. Use only one environment in cluster mode. Use as many environment as your
        ### threads (or a little less) for fastest training.
        env.reset()
        _, _, _, info = env.step([0, 0])
        info_keywords = tuple(info.keys())
        envs = SubprocVecEnv(
            [
                lambda: get_env(args.use_case, args.k8s, args.goal)
                for i in range(8)
            ]
        )
        envs = VecMonitor(envs, MONITOR_PATH, info_keywords=info_keywords)
        model = DSPPO(envs, num_steps=100, n_minibatches=8, tensorboard_log=None)
    else:
        logging.info('Invalid algorithm!')

    return model

def get_load_model(alg, tensorboard_log, load_path):
    if alg == 'ppo':
        return PPO.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        return RecurrentPPO.load(load_path, reset_num_timesteps=False, verbose=1,
                                 tensorboard_log=tensorboard_log)  # n_steps=steps
    elif alg == 'a2c':
        return A2C.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log)
    elif alg == 'ds':
        return
    else:
        logging.info('Invalid algorithm!')


def get_env(use_case, k8s, goal):
    env = 0
    if use_case == 'redis':
        env = Redis(k8s=k8s, goal_reward=goal)
    elif use_case == 'online_boutique':
        env = OnlineBoutique(k8s=k8s, goal_reward=goal)
    else:
        logging.info('Invalid use_case!')

    return env


def main():
    # Import and initialize Environment
    logging.info(args)

    alg = args.alg
    k8s = args.k8s
    use_case = args.use_case
    goal = args.goal
    loading = args.loading
    load_path = args.load_path
    training = args.training
    testing = args.testing
    test_path = args.test_path

    steps = int(args.steps)
    total_steps = int(args.total_steps)

    env = get_env(use_case, k8s, goal)

    scenario = ''
    if k8s:
        scenario = 'real'
    else:
        scenario = 'simulated'

    tensorboard_log = "results/" + use_case + "/" + scenario + "/" + goal + "/"

    name = alg + "_env_" + env.name + "_goal_" + goal + "_k8s_" + str(k8s) + "_totalSteps_" + str(total_steps) + '_none_pen'

    # callback
    checkpoint_callback = CheckpointCallback(save_freq=steps, save_path="logs/" + name, name_prefix=name)

    if training:
        if loading:  # resume training
            model = get_model(alg, env, tensorboard_log, args)
            model.load(load_path)
            model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)
        else:
            model = get_model(alg, env, tensorboard_log, args)
            model.learn(total_timesteps=total_steps, tb_log_name=name + "_run_m", callback=checkpoint_callback)

        model.save(name)

    if testing:
        ### This is the only case where we alter the main functioning of the stable version of the algorithm.
        ### It is because we need to process the observation before making a step. So the following code works only for DS.
        # test_model(model, env, n_episodes=2000, n_steps=110, smoothing_window=100, fig_name=name + "_test_reward_best.png")
        env=env
        env.reset()
        _, _, _, info = env.step([0,0])
        info_keywords = tuple(info.keys())
        n_episodes = 100
        n_steps = 100
        smoothing_window = 5
        fig_name = "_test_reward_final.png"
        episode_rewards = []
        reward_sum = 0

        envs = DummyVecEnv([lambda: env])
        envs = VecMonitor(envs, MONITOR_PATH, info_keywords=info_keywords)
        # envs = VecNormalize(envs, norm_obs=True, norm_reward=False)
        model = DSPPO(envs, num_steps=n_steps, n_minibatches=1, ent_coef=0.001, tensorboard_log=None, seed=SEED)
        model.load(test_path)
        print('Testing:', test_path)
        for e in range(n_episodes):
            obs = envs.reset()
            obs = model.reshape_obs(obs)
            done = False
            while not done:
                print(obs)
                action = model.predict(obs)
                valid_action = [[action[0] // env.num_actions, action[0] % env.num_actions]]
                print(valid_action)
                obs, reward, dones, info = model.env.step(valid_action)
                print('reward:', reward)
                obs = model.reshape_obs(obs)
                reward_sum += reward
                done = dones[0]
            episode_rewards.append(reward_sum)
            print("Episode {} | Total reward: {} |".format(e, str(reward_sum)))
            reward_sum = 0

        plt.figure()
        rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig('test_results_max_1' + fig_name, dpi=250, bbox_inches='tight')

if __name__ == "__main__":
    main()
