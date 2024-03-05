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

from setuptools.command.alias import alias
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from sb3_contrib import RecurrentPPO, MaskablePPO

from gym_hpa.envs import Redis, OnlineBoutique
from stable_baselines3.common.callbacks import CheckpointCallback

# Logging
from policies.util.util import test_model

logging.basicConfig(filename='run.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description='Run ILP!')
parser.add_argument('--alg', default='a2c', help='The algorithm: ["ppo", "recurrent_ppo", "a2c"]')
parser.add_argument('--k8s', default=False, action="store_true", help='K8s mode')
parser.add_argument('--use_case', default='redis', help='Apps: ["redis", "online_boutique"]')
parser.add_argument('--goal', default='latency', help='Reward Goal: ["cost", "latency"]')

parser.add_argument('--training', default=False, action="store_true", help='Training mode')
parser.add_argument('--testing', default=True, action="store_true", help='Testing mode')
parser.add_argument('--loading', default=False, action="store_true", help='Loading mode')
parser.add_argument('--load_path', default='logs/lets_try_normal_env_now/lets_try_this_env_now_10000_steps.zip', help='Loading path, ex: logs/model/test.zip')
parser.add_argument('--test_path', default='models/a2c_100k_redis_cost_3_obs_new_sim_none_penalty_norm/training_1/steps/finished_100000_steps.zip', help='Testing path, ex: logs/model/test.zip')

parser.add_argument('--steps', default=500, help='The steps for saving.')
parser.add_argument('--total_steps', default=200000, help='The total number of steps.')

args = parser.parse_args()


def get_model(alg, env, tensorboard_log):
    model = 0
    if alg == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
    elif alg == 'a2c':
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)  # , n_steps=steps
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

    name = 'stable'

    # callback
    checkpoint_callback = CheckpointCallback(save_freq=steps, save_path="logs/" + name, name_prefix=name)

    if training:
        if loading:  # resume training
            model = get_load_model(alg, tensorboard_log, load_path)
            model.set_env(env)
            model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)
        else:
            model = get_model(alg, env, tensorboard_log)
            model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)

        model.save(name)

    if testing:
        model = get_load_model(alg, tensorboard_log, test_path)
        test_model(model, env, n_episodes=100, n_steps=110, smoothing_window=5, fig_name="results.png")


if __name__ == "__main__":
    main()
