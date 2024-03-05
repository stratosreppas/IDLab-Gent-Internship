#Q: how to run it
#A: python run.py --alg ppo --use_case redis --goal cost --training

import logging
import argparse

from setuptools.command.alias import alias
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from sb3_contrib import RecurrentPPO
from gym_hpa.predictions import Prediction

from gym_hpa.envs import Redis, OnlineBoutique
from stable_baselines3.common.callbacks import CheckpointCallback

# Logging
from policies.util.util import test_model

logging.basicConfig(filename='run.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description='Run ILP!')
parser.add_argument('--alg', default='a2c', help='The algorithm: ["ppo", "recurrent_ppo", "a2c"]')
parser.add_argument('--k8s', default=False, action="store_true", help='K8s mode')
parser.add_argument('--use_case', default='redis', help='Apps: ["redis", "onlineboutique"]')
parser.add_argument('--goal', default='cost', help='Reward Goal: ["cost", "latency"]')

parser.add_argument('--prediction', default='ses', help='Prediction method: ["lstm", "arima", "naive", "prophet", "ses", "average"]')
parser.add_argument('--timeseries', default=False, help='Timeseries: if True, use corrected data, else use initial data')

parser.add_argument('--training', default=False, action="store_true", help='Training mode')
parser.add_argument('--testing', default=True, action="store_true", help='Testing mode')
parser.add_argument('--loading', default=False, action="store_true", help='Loading mode')
parser.add_argument('--load_path', default='logs/model/test.zip', help='Loading path, ex: logs/model/test.zip')
parser.add_argument('--test_path', default='logs/fin_a2c_env_redis_gym_goal_cost_k8s_False_totalSteps_100000_pred_ses_dynamic/fin_a2c_env_redis_gym_goal_cost_k8s_False_totalSteps_100000_pred_ses_dynamic_35500_steps.zip', help='Testing path, ex: logs/model/test.zip')

parser.add_argument('--steps', default=500, help='The steps for saving.')
parser.add_argument('--total_steps', default=100000, help='The total number of steps.')

args = parser.parse_args()


def get_model(alg, env, tensorboard_log):
    """
    Sets up the model using the specified algorithm and returns it. After that, it is ready to be trained.
    """
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


def get_prediction(method, timeseries, deployments):
    """
    Returns the predicted timeseries for each of the methods.
    """
    for d in deployments:
        print('Predicting for', d.name)
        pred = Prediction(trainset='datasets/real/redis/v1/redis_gym_observation.csv', val_col=d.name+'_cpu_usage', date_col='date')
        if timeseries:
            print('Loading dataset...')
            pred.trainset, pred.testset = pred.convert_to_timeseries()
            print('Dataset loaded!')
        print('Predicting...')
        if method == 'ses':
            predictions = pred.ses(0.6)
        elif method == 'arima':
            pred.model = pred.arima_fit()
           # save(pred.model) # run fitting once and save to file (takes a lot of time and the dataset is the same)
            predictions = pred.arima(pred.model)
        elif method == 'lstm':
            pred.trainset = pred.trainset[:len(pred.testset)//2]
            predictions = pred.lstm(200, 1)
        elif method == 'naive':
            predictions = pred.naive()
        elif method == 'average':
            predictions = pred.average()
        else:
            logging.info('Invalid prediction method!')
        print('Prediction done!')
        pred.testset[pred.val_col+'_predictions'] = predictions
        print('RMSE: ', pred.rmse())
        print('MAE: ', pred.mae())
        print('MAPE: ', pred.mape())
        print('Saving dataset...')
        pred.testset.to_csv('datasets/real/redis/v1/redis_gym_observation.csv', index=False)
        print('Dataset saved!')

    return predictions


def get_load_model(alg, tensorboard_log, load_path):
    """
    Loads a pretrained model and continues its training using the specified algorithm.
    """

    if alg == 'ppo':
        return PPO.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        return RecurrentPPO.load(load_path, reset_num_timesteps=False, verbose=1,
                                 tensorboard_log=tensorboard_log)  # n_steps=steps
    elif alg == 'a2c':
        return A2C.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log)
    else:
        logging.info('Invalid algorithm!')


def get_env(use_case, k8s, goal, prediction):
    """
    Returns the environment for each of the use cases. For more info check the Redis and OnlineBoutique classes.
    """

    env = 0
    if use_case == 'redis':
        env = Redis(k8s=k8s, goal_reward=goal, prediction=prediction)
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
    prediction = args.prediction
    timeseries = args.timeseries
    loading = args.loading
    load_path = args.load_path
    training = args.training
    testing = args.testing
    test_path = args.test_path

    steps = int(args.steps)
    total_steps = int(args.total_steps)

    env = get_env(use_case, k8s, goal, prediction)
    # get_prediction(prediction, timeseries, env.deploymentList)

    scenario = ''
    if k8s:
        scenario = 'real'
    else:
        scenario = 'simulated'

    tensorboard_log = "results/" + use_case + "/" + scenario + "/" + goal + "/"

    name = 'fin_' + alg + "_env_" + env.name + "_goal_" + goal + "_k8s_" + str(k8s) + "_totalSteps_" + str(total_steps) + \
           "_pred_" + prediction + '_dynamic'

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
        test_model(model, env, n_episodes=100, n_steps=110, smoothing_window=5, fig_name=name + "_lets.png")


if __name__ == "__main__":
    main()
