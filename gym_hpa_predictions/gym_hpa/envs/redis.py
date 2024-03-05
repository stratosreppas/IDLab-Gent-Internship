import csv
import datetime
import logging
import time
import pickle
from statistics import mean

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from datetime import datetime

from gym_hpa_predictions.gym_hpa.envs.deployment import get_max_cpu, get_max_mem, get_max_traffic, get_redis_deployment_list
from gym_hpa_predictions.gym_hpa.envs.util import save_to_csv, get_cost_reward, get_latency_reward_redis, get_num_pods
from gym_hpa_predictions.gym_hpa.predictions import Prediction ###

# MIN and MAX Replication
MIN_REPLICATION = 1
MAX_REPLICATION = 8

MAX_STEPS = 25  # MAX Number of steps per episode

# Possible Actions (Discrete)
ACTION_DO_NOTHING = 0
ACTION_ADD_1_REPLICA = 1
ACTION_ADD_2_REPLICA = 2
ACTION_ADD_3_REPLICA = 3
ACTION_ADD_4_REPLICA = 4
ACTION_ADD_5_REPLICA = 5
ACTION_ADD_6_REPLICA = 6
ACTION_ADD_7_REPLICA = 7
ACTION_TERMINATE_1_REPLICA = 8
ACTION_TERMINATE_2_REPLICA = 9
ACTION_TERMINATE_3_REPLICA = 10
ACTION_TERMINATE_4_REPLICA = 11
ACTION_TERMINATE_5_REPLICA = 12
ACTION_TERMINATE_6_REPLICA = 13
ACTION_TERMINATE_7_REPLICA = 14

# Deployments
DEPLOYMENTS = ["redis-leader", "redis-follower"]

# Action Moves
MOVES = ["None", "Add-1", "Add-2", "Add-3", "Add-4", "Add-5", "Add-6", "Add-7",
         "Stop-1", "Stop-2", "Stop-3", "Stop-4", "Stop-5", "Stop-6", "Stop-7"]

# IDs
ID_DEPLOYMENTS = 0
ID_MOVES = 1

ID_MASTER = 0
ID_SLAVE = 1

# Reward objectives
LATENCY = 'latency'
COST = 'cost'


class Redis(gym.Env):
    """Horizontal Scaling for Redis in Kubernetes - an OpenAI gym environment"""

    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, prediction, k8s=False, goal_reward=COST, waiting_period=1):
        """
        Our custom OpenAI environment for the Redis Application

        :param: prediction: the prediction algorithm to be used
        :param: k8s: If True, we run in cluster mode, else on simulation mode
        :param: goal_reward: The reward function to be used. It is either COST or LATENCY
        :param: waiting_period: The time to wait for the number of pods to be updated in cluster mode
        """

        super(Redis, self).__init__()

        self.k8s = k8s
        self.name = "redis_gym"
        self.__version__ = "0.0.1"
        self.seed()
        self.goal_reward = goal_reward
        self.waiting_period = waiting_period  # seconds to wait after action

        logging.info("[Init] Env: {} | K8s: {} | Version {} |".format(self.name, self.k8s, self.__version__))

        # Current Step
        self.current_step = 0

        # Actions identified by integers 0-n -> 15 actions!
        self.num_actions = 15

        # Multi-Discrete version
        # Deployment: Discrete 2 - Master[0], Slave[1]
        # Action: Discrete 9 - None[0], Add-1[1], Add-2[2], Add-3[3], Add-4[4],
        #                      Stop-1[5], Stop-2[6], Stop-3[7], Stop-4[8]

        self.action_space = spaces.MultiDiscrete([2, self.num_actions])

        # Observations: 22 Metrics! -> 2 * 11 = 22
        # "number_pods"                     -> Number of deployed Pods
        # "cpu_usage_aggregated"            -> via metrics-server
        # "mem_usage_aggregated"            -> via metrics-server
        # "cpu_requests"                    -> via metrics-server/pod
        # "mem_requests"                    -> via metrics-server/pod
        # "cpu_limits"                      -> via metrics-server
        # "mem_limits"                      -> via metrics-server
        # "lstm_cpu_prediction_1_step"      -> via pod annotation
        # "lstm_cpu_prediction_5_step"      -> via pod annotation
        # "average_number of requests"      -> Prometheus metric: sum(rate(http_server_requests_seconds_count[5m]))

        self.min_pods = MIN_REPLICATION
        self.max_pods = MAX_REPLICATION
        self.num_apps = 2

        # Deployment Data
        self.deploymentList = get_redis_deployment_list(self.k8s, self.min_pods, self.max_pods)

        self.observation_space = self.get_observation_space()

        # Action and Observation Space
        logging.info("[Init] Action Spaces: " + str(self.action_space))
        logging.info("[Init] Observation Spaces: " + str(self.observation_space))

        # Info
        self.total_reward = None
        self.avg_pods = []
        self.avg_latency = []

        # episode over
        self.episode_over = False
        self.info = {}

        # Keywords for Reward calculation
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False
        self.cost_weight = 0  # add here a value to consider cost in the reward function

        self.time_start = 0
        self.execution_time = 0
        self.episode_count = 0
        self.file_results = "results.csv"
        self.obs_csv = self.name + "_actual_observation"
        self.df = pd.read_csv("datasets/real/" + self.deploymentList[0].namespace + "/v1/"
                              + self.name + '_' + 'observation.csv')

        self.none_counter = 0 ### initialization of the none counter used in the reward function
        self.prediction = prediction ### initialization of the prediction variable, that will store the prediction for the next observation

    def step(self, action):
        """
        Performs one step within the environment. It uses the action provided.

        :param action: The action to perform
        """

        if self.current_step == 1:
            if not self.k8s:
                self.simulation_update(action)
                # ob = self.get_state()
                # date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # self.save_obs_to_csv(self.obs_csv, np.array(ob), date, self.deploymentList[0].latency)

            self.time_start = time.time()

        # Get first action: deployment
        if action[ID_DEPLOYMENTS] == 0:  # master
            n = ID_MASTER  # master
        else:
            n = ID_SLAVE  # slave

        # Execute one time step within the environment
        self.take_action(action[ID_MOVES], n)

        # Wait a few seconds if on real k8s cluster
        if self.k8s:
            if action[ID_MOVES] != ACTION_DO_NOTHING \
                    and self.constraint_min_pod_replicas is False \
                    and self.constraint_max_pod_replicas is False:
                # logging.info('[Step {}] | Waiting {} seconds for enabling action ...'
                # .format(self.current_step, self.waiting_period))
                time.sleep(self.waiting_period)  # Wait a few seconds...

        # Update observation before reward calculation:
        if self.k8s:  # k8s cluster
            for d in self.deploymentList:
                d.update_obs_k8s()
        else:  # simulation
            self.simulation_update(action)

        # Get reward
        reward = self.get_reward

        # Update Infos
        self.total_reward += reward
        self.avg_pods.append(get_num_pods(self.deploymentList))
        self.avg_latency.append(self.deploymentList[0].latency)

        # Print Step and Total Reward
        # if self.current_step == MAX_STEPS:
        logging.info('[Step {}] | Action (Deployment): {} | Action (Move): {} | Reward: {} | Total Reward: {}'.format(
            self.current_step, DEPLOYMENTS[action[0]], MOVES[action[1]], reward, self.total_reward))

        # Save the step to the observation csv file
        ob = self.get_state()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_obs_to_csv(self.obs_csv, np.array(ob), date, self.deploymentList[0].latency)

        self.info = dict(
            total_reward=self.total_reward,
        )

        # Update Reward Keywords
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        if self.current_step == MAX_STEPS:
            self.episode_count += 1
            self.execution_time = time.time() - self.time_start
            save_to_csv(self.file_results, self.episode_count, mean(self.avg_pods), mean(self.avg_latency),
                        self.total_reward, self.execution_time)

        # return ob, reward, self.episode_over, self.info
        return np.array(ob), reward, self.episode_over, self.info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        :return: The numpy array of an initial observation, using the get_state() function
        """

        self.current_step = 0
        self.episode_over = False
        self.total_reward = 0
        self.avg_pods = []
        self.avg_latency = []

        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        # Deployment Data
        self.deploymentList = get_redis_deployment_list(self.k8s, self.min_pods, self.max_pods)

        return np.array(self.get_state())

    def render(self, mode='human', close=False):
        """
        Render the environment to the screen. Not implemented.
        """
        return

    def take_action(self, action, id):
        """
        Performs the action provided (move or deploy) and updates the environment.

        :param action: The action to perform. It is only the number of pods to deplou/terminate, not the MultiDiscrete action itself
        :param id: The id of the deployment the action will be done in
        """

        self.current_step += 1

        # Stop if MAX_STEPS
        if self.current_step == MAX_STEPS:
            # logging.info('[Take Action] MAX STEPS achieved, ending ...')
            self.episode_over = True

        # ACTIONS
        if action == ACTION_DO_NOTHING:
            self.none_counter += 1 ### increments the none counter if the corresponding action is made
            # logging.info("[Take Action] SELECTED ACTION: DO NOTHING ...")
            pass

        elif action == ACTION_ADD_1_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 1 Replica ...")
            self.deploymentList[id].deploy_pod_replicas(1, self)

        elif action == ACTION_ADD_2_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 2 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(2, self)

        elif action == ACTION_ADD_3_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 3 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(3, self)

        elif action == ACTION_ADD_4_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 4 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(4, self)

        elif action == ACTION_ADD_5_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 5 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(5, self)

        elif action == ACTION_ADD_6_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 6 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(6, self)

        elif action == ACTION_ADD_7_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 7 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(7, self)

        elif action == ACTION_TERMINATE_1_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 1 Replica ...")
            self.deploymentList[id].terminate_pod_replicas(1, self)

        elif action == ACTION_TERMINATE_2_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 2 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(2, self)

        elif action == ACTION_TERMINATE_3_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 3 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(3, self)

        elif action == ACTION_TERMINATE_4_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 4 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(4, self)

        elif action == ACTION_TERMINATE_5_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 5 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(5, self)

        elif action == ACTION_TERMINATE_6_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 6 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(6, self)

        elif action == ACTION_TERMINATE_7_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 7 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(7, self)

        else:
            logging.info('[Take Action] Unrecognized Action: ' + str(action))

    @property
    def get_reward(self):
        """
        Returns the reward based on the current state. We use the current state because it is updated right after the
        action, so it represents the state, after the action that the RL agent made (see step() function). It sets the
        penalty for exceeding the limitations and uses the calculate_reward function to calculate the reward,
        if the action was allowed by these limitations.
        Perhaps a masking function could result in a better agent, since now all the actions have only positive values
        in the cost instance, so none action shouldn't be preferred.

        :return: The reward for the action made
        """

        '''
        ob = self.get_state()
        logging.info('[Reward] | Master Pods: {} | CPU Usage: {} | MEM Usage: {} | Requests: {} | Response Time: {} | '
                     'Slave Pods: {} | CPU Usage: {} | MEM Usage: {} | Requests: {} | Response Time: {} |'.format(
            ob.__getitem__(0), ob.__getitem__(1), ob.__getitem__(2), ob.__getitem__(9), ob.__getitem__(10),
            ob.__getitem__(11), ob.__getitem__(12), ob.__getitem__(13), ob.__getitem__(20), ob.__getitem__(21), ))
        '''
        # Reward based on Keyword!
        if self.constraint_max_pod_replicas:
            if self.goal_reward == COST:
                return -1  # penalty
            elif self.goal_reward == LATENCY:
                return -250  # penalty

        if self.constraint_min_pod_replicas:
            if self.goal_reward == COST:
                return -1  # penalty
            elif self.goal_reward == LATENCY:
                return -250  # penalty

        # Reward Calculation
        reward = self.calculate_reward()
        # logging.info('[Get Reward] Reward: {} | Ob: {} |'.format(reward, ob))
        # logging.info('[Get Reward] Acc. Reward: {} |'.format(self.total_reward))

        return reward

    def get_state(self):
        """
        Returns the current state of the environment
        """
        # Observations: metrics - 3 Metrics!!
        # "number_pods"
        # "cpu"
        # "mem"
        # "requests"

        ### DYNAMIC FORECAST ###
        # TODO: comment this section if you want to apply static forecast

        for deployment in self.deploymentList:
            val_col = deployment.name + '_cpu_usage' ### MODIFY HERE TO CHANGE THE PREDICTION TARGET
            pred = Prediction(date_col='date', val_col=val_col, current_value=int(deployment.cpu_usage),
                              trainset=self.obs_csv + '_recent.csv')
            if self.prediction == 'naive':
                deployment.predictions = int(pred.naive_dynamic())
            elif self.prediction == 'ses':
                # if this is the first step of the episode, then we need to reset the predictions,
                # as ses is highly dependent on the initial value. In that case, we use the static prediction
                if self.current_step != 0:
                    deployment.predictions = int(pred.ses_dynamic())
            elif self.prediction == 'lstm':
                deployment.predictions = int(pred.lstm_dynamic())
            elif self.prediction == 'arima':
                if self.current_step % (5 * self.current_step+1) == 0:
                    arima_model = pred.arima_fit()
                    with open('gym_hpa/predictions/pred_models/dynamic_arima_model.pkl', 'wb') as directory:
                        pickle.dump(arima_model, directory)
                with open('gym_hpa/predictions/pred_models/dynamic_arima_model.pkl', 'rb') as directory:
                    model = pickle.load(directory)
                deployment.predictions = int(pred.arima_dynamic(model))
            elif self.prediction == 'average':
                deployment.predictions = int(pred.average_dynamic())

        ob = (
            self.deploymentList[0].num_pods, self.deploymentList[0].desired_replicas,
            self.deploymentList[0].cpu_usage, self.deploymentList[0].mem_usage,
            self.deploymentList[0].received_traffic, self.deploymentList[0].transmit_traffic,
            self.deploymentList[1].num_pods, self.deploymentList[1].desired_replicas,
            self.deploymentList[1].cpu_usage, self.deploymentList[1].mem_usage,
            self.deploymentList[1].received_traffic, self.deploymentList[1].transmit_traffic,
            self.deploymentList[0].predictions, self.deploymentList[1].predictions ### add then predictions in the observation space
        )

        return ob

    def get_observation_space(self):
        """
        Returns the shape and the restrictions for the observation space of the environment
        """

        return spaces.Box(
                low=np.array([
                    self.min_pods,  # Number of Pods  -- master metrics
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  ### Predicted CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- slave metrics
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  ### Predicted CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                ]), high=np.array([
                    self.max_pods,  # Number of Pods -- master metrics
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_cpu(),  ### Predicted CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- slave metrics
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_cpu(),  ### Predicted CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                ]),
                dtype=np.float32
            )

    # calculates the reward based on the objective
    def calculate_reward(self):
        """
        Calculates the reward based on the objective
        """
        reward = 0
        if self.goal_reward == COST:
            reward = get_cost_reward(self.deploymentList)
            ### none penalty: if the agent uses more than twice the none action without achieving the highest reward, penaltize it increasingly.
            if reward != 2 and self.none_counter>2:
                reward = -self.none_counter

        elif self.goal_reward == LATENCY:
            reward = get_latency_reward_redis(ID_MASTER, self.deploymentList)
            ### none penalty: if the agent uses more than twice the none action, penaltize it increasingly.
            ### No second condition was added in this case like above, because spamming the none action sometimes bore good latency values.
            if self.none_counter>2:
                reward = -self.none_counter * 100

        return reward

    def simulation_update(self, action):
        """
        Updates the simulation based on the current step.

        Note that if it chooses the in-sample predictions for the new observations space. So, if we disable the
        out-of-sample predictions in the get_state() function, this is the observation that the agent gets in
        simulation mode. In cluster mode, it was not implemented as it was considered not to be useful.

        NEW: The environment corresponds better to the agents actions,
             applying as most important criteria for the update to have
             the corresponding number of pods in the deployment the action was made

        :param: action: The action is used in the new environment, as we take it into consideration.
        """

        # For the first step, get a random row from the dataframe
        if self.current_step == 1:
            # Get a random sample!
            sample = self.df.sample()
            # print(sample)


            self.deploymentList[0].num_pods = int(sample['redis-leader_num_pods'].values[0])
            self.deploymentList[0].num_previous_pods = int(sample['redis-leader_num_pods'].values[0])
            self.deploymentList[0].predictions = int(sample['redis-leader_cpu_usage_predictions'].values[0]) ### MODIFY HERE TO CHANGE THE PREDICTION TARGET

            self.deploymentList[1].num_pods = int(sample['redis-follower_num_pods'].values[0])
            self.deploymentList[1].num_previous_pods = int(sample['redis-follower_num_pods'].values[0])
            self.deploymentList[1].predictions = int(sample['redis-follower_cpu_usage_predictions'].values[0]) ### MODIFY HERE TO CHANGE THE PREDICTION TARGET

            # For the next steps, try to find a row in the dataset that matches the current state as well as
            # the previous state which is influenced by the agent's actions. This way, there can be a dynamic
            # relationship between the agent and the environment, while using real data.
        else:

            ### New Simulation ###

            action_num = action[ID_DEPLOYMENTS]
            other_num = 1 - action[ID_DEPLOYMENTS]

            action_pods = self.deploymentList[action_num].num_pods
            action_previous_pods = self.deploymentList[action_num].num_previous_pods
            other_pods = self.deploymentList[other_num].num_pods
            other_previous_pods = self.deploymentList[other_num].num_previous_pods

            diff_action = action_pods - action_previous_pods
            diff_other = other_pods - other_previous_pods

            action_deployment = DEPLOYMENTS[action_num]
            other_deployment = DEPLOYMENTS[other_num]

            self.df['diff-' + action_deployment] = self.df[action_deployment + '_num_pods'].diff()
            self.df['diff-' + other_deployment] = self.df[other_deployment + '_num_pods'].diff()

            data = self.df.loc[self.df[action_deployment + '_num_pods'] == action_pods]
            next_data = data.loc[data[other_deployment + '_num_pods'] == other_pods]

            # action_pods -> y, other_pods -> n
            if next_data.size == 0:
                next_data = data.loc[data['diff-' + action_deployment] == diff_action]

                if next_data.size == 0:
                    # print('# action_pods -> y, other_pods -> n, diff-action -> n, diff-other -> n')
                    sample = data.sample()
                else:
                    # print('# action_pods -> y, other_pods -> n, diff-action -> y, diff-other -> n')
                    sample = next_data.sample()

            else:
                # action_pods->y, other_pods->y
                new_data = next_data.loc[next_data['diff-' + action_deployment] == diff_action]

                if new_data.size == 0:
                    # action_pods -> y, other_pods -> y, diff-action -> n
                    next_data = next_data.loc[next_data['diff-' + other_deployment] == diff_other]

                    if next_data.size == 0:
                        # print('# action_pods -> y, other_pods -> y, diff-action -> n, diff-other -> n')
                        sample = data.sample()
                    else:
                        # print('# action_pods -> y, other_pods -> y, diff-action -> n, diff-other -> y')
                        sample = next_data.sample()
                else:
                    # action_pods -> y, other_pods -> y, diff-action -> y
                    next_data = new_data.loc[new_data['diff-' + other_deployment] == diff_other]
                    if next_data.size == 0:
                        # print('# action_pods -> y, other_pods -> y, diff-action -> y, diff-other -> n')
                        sample = new_data.sample()
                    else:
                        # print('# action_pods -> y, other_pods -> y, diff-action -> y, diff-other -> y')
                        sample = next_data.sample()

        # print(sample)

        # Update the state of the environment

        self.deploymentList[0].cpu_usage = int(sample['redis-leader_cpu_usage'].values[0])
        self.deploymentList[0].mem_usage = int(sample['redis-leader_mem_usage'].values[0])
        self.deploymentList[0].received_traffic = int(sample['redis-leader_traffic_in'].values[0])
        self.deploymentList[0].transmit_traffic = int(sample['redis-leader_traffic_out'].values[0])
        self.deploymentList[0].latency = float(sample['redis-leader_latency'].values[0])
        self.deploymentList[0].predictions = float(sample['redis-leader_cpu_usage_predictions'].values[0]) ### MODIFY HERE TO CHANGE THE PREDICTION TARGET

        self.deploymentList[1].cpu_usage = int(sample['redis-follower_cpu_usage'].values[0])
        self.deploymentList[1].mem_usage = int(sample['redis-follower_mem_usage'].values[0])
        self.deploymentList[1].received_traffic = int(sample['redis-follower_traffic_in'].values[0])
        self.deploymentList[1].transmit_traffic = int(sample['redis-follower_traffic_out'].values[0])
        self.deploymentList[1].latency = float(sample['redis-follower_latency'].values[0])
        self.deploymentList[1].predictions = float(sample['redis-follower_cpu_usage_predictions'].values[0]) ### MODIFY HERE TO CHANGE THE PREDICTION TARGET




        for d in self.deploymentList:
            # Update Desired replicas
            d.update_replicas()
        return


    def save_obs_to_csv(self, obs_file, obs, date, latency, past_horizon=100):
        """
        Updates the observation file, that contains the past observations, with the current observation

        :param: obs_file: The name of the fiel to store the observations
        :param: obs: The current observation to be saved
        :param: date: The date
        :param: latency: The calculated latency of the current observation
        :param: past_horizon: The number of the latest observations to be saved to a smaller file, used in predictions
        """
        file = open(obs_file+'.csv', 'a+', newline='')  # append

        # new
        # file = open(file_name, 'w', newline='') # new
        fields = []
        with file:
            fields.append('date')
            for d in self.deploymentList:
                fields.append(d.name + '_num_pods')
                fields.append(d.name + '_desired_replicas')
                fields.append(d.name + '_cpu_usage')
                fields.append(d.name + '_cpu_usage_predictions') ###
                fields.append(d.name + '_mem_usage')
                fields.append(d.name + '_traffic_in')
                fields.append(d.name + '_traffic_out')
                fields.append(d.name + '_latency')

            new_entry = {'date': date,
                 'redis-leader_num_pods': float("{}".format(obs[0])),
                 'redis-leader_desired_replicas': float("{}".format(obs[1])),
                 'redis-leader_cpu_usage': float("{}".format(obs[2])),
                 'redis-leader_cpu_usage_predictions': float("{}".format(obs[12])), ###
                 'redis-leader_mem_usage': float("{}".format(obs[3])),
                 'redis-leader_traffic_in': float("{}".format(obs[4])),
                 'redis-leader_traffic_out': float("{}".format(obs[5])),
                 'redis-leader_latency': float("{:.3f}".format(latency)),
                 'redis-follower_num_pods': float("{}".format(obs[6])),
                 'redis-follower_desired_replicas': float("{}".format(obs[7])),
                 'redis-follower_cpu_usage': float("{}".format(obs[8])),
                 'redis-follower_cpu_usage_predictions': float("{}".format(obs[13])), ###
                 'redis-follower_mem_usage': float("{}".format(obs[9])),
                 'redis-follower_traffic_in': float("{}".format(obs[10])),
                 'redis-follower_traffic_out': float("{}".format(obs[11])),
                 'redis-follower_latency': float("{:.3f}".format(latency)),
                 }
            '''
            fields = ['date', 'redis-leader_num_pods', 'redis-leader_desired_replicas', 'redis-leader_cpu_usage', 'redis-leader_mem_usage',
                      'redis-leader_cpu_request', 'redis-leader_mem_request', 'redis-leader_cpu_limit', 'redis-leader_mem_limit',
                      'redis-leader_traffic_in', 'redis-leader_traffic_out', 'redis-leader_cpu_usage_predictions', redis-follower_cpu_usage_predictions 
                      'redis-follower_num_pods', 'redis-follower_desired_replicas', 'redis-follower_cpu_usage',
                      'redis-follower_mem_usage', 'redis-follower_cpu_request', 'redis-follower_mem_request', 'redis-follower_cpu_limit',
                      'redis-follower_mem_limit', 'redis-follower_traffic_in', 'redis-follower_traffic_out']
            '''
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writerow(new_entry)

        ### Implementation of the recent observation space. It keeps the last (past_horizon) values.
        recent_file = open(obs_file + '_recent.csv', 'a+', newline='')

        with recent_file:
            recent_writer = csv.DictWriter(recent_file, fieldnames=fields)
            recent_writer.writerow(new_entry)

            # Read the current lines from the small file
            recent_file.seek(0)
            lines = recent_file.readlines()

            # If the number of lines exceeds past_horizon, remove the first line
            if len(lines) > past_horizon-1:
                lines.pop(1)

                recent_file.seek(0)
                recent_file.truncate()
                recent_file.writelines(lines)
        return
