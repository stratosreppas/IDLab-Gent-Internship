import random
import time
from typing import Optional, Union

import gym
import gym.spaces
import numpy as np
import numpy.typing as npt
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from gym_hpa_deepsets.deepsets.common.algorithm import Algorithm
from gym_hpa_deepsets.deepsets.ppo.agents import DeepSetAgent


# borrowed from CleanRL single-file PPO implementation
class DSPPO(Algorithm):
    """
    Deep Set Proximal Policy Optimization (PPO) algorithm. Realizes the functioning of the Deep Set Agent.
    """
    def __init__(
        self,
        env: Union[SubprocVecEnv, DummyVecEnv],
        learning_rate: float = 0.01,
        anneal_lr: bool = False,
        num_steps: int = 128,
        gae: bool = False,
        gae_lambda: float = 0.1,
        gamma: float = 0.1,
        n_minibatches: int = 4,
        update_epochs: int = 1,
        norm_adv: bool = False,
        clip_coef: float = 0.2,
        clip_vloss: bool = True,
        ent_coef: float = 0.1,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        seed: int = 1,
        device: str = "cpu",
        tensorboard_log: str = "./run",
    ):
        super().__init__(env, learning_rate, seed, device, tensorboard_log)
        self.num_envs = self.env.num_envs
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.num_steps = num_steps
        self.gae = gae
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_minibatches = n_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.hyperparams = {
            "num_envs": self.num_envs,
            "learning_rate": self.learning_rate,
            "anneal_lr": anneal_lr,
            "num_steps": num_steps,
            "use_gae": self.gae,
            "gae_lambda": self.gae_lambda,
            "gamma": self.gamma,
            "n_minibatches": self.n_minibatches,
            "update_epochs": self.update_epochs,
            "normalize_advantages": self.norm_adv,
            "clip_coefficient": self.clip_coef,
            "clip_value_loss": self.clip_vloss,
            "entropy_coefficient": self.ent_coef,
            "value_function_coefficient": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
        }

        # Sets the size of the batch and the minibatch.
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.n_minibatches)

        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in self.hyperparams.items()])),
        )

        self.num_apps = self.env.get_attr("num_apps")[0]  # the number of microservices the environment features
        self.num_actions = self.env.get_attr("num_actions")[0]  # the number of possible actions the agent has

        # Here we set the random seeds to ensure reproducibility. The seeds are numbers from 0 to 2^32-1, that indicate
        # a specific state of the random number generator, so if we set the same seed, we will get the same sequence
        # of random numbers in the same order in different runs.
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # TODO: modify it to multibinary
        # assert isinstance(self.env.action_space, gymnasium.spaces.Discrete), "only discrete action space is supported"

        # The agent is a Deep Set Agent, which is a neural network that takes as input
        # the state of the environment and outputs the probability of taking each action.
        # The optimizer is the Adam optimizer, which is a gradient descent optimizer, used to update the parameters of
        # the agent.

        self.agent = DeepSetAgent(self.env).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        # Here we initialize the tensors that will be used to store the data collected from the environment.
        # The tensors are initialized with zeros and then moved to the device (CPU or GPU) that we are using.
        # The tensors are initialized with the following shapes:

        # - obs: (n_steps, num_apps, obs_shape / num_apps + 1) - the observations
        self.obs = torch.zeros((self.num_steps, self.num_envs, self.num_apps * self.num_actions, ((self.env.observation_space.shape[0]) // self.num_apps)+1)).to(self.device)
        # - actions: (n_steps, num_envs, *action_shape) - the actions taken

        self.actions = torch.zeros(self.num_steps, self.num_envs).to(self.device)
        # print(self.actions.shape)
        # self.masks = torch.zeros((self.num_steps, self.num_envs, self.env.action_space.n), dtype=torch.bool).to(self.device) ##
        # - logprobs: (n_steps, num_envs) - the log probabilities of the actions taken by the agent
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        # - rewards: (n_steps, num_envs) - the rewards received by the agent
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        # - dones: (n_steps, num_envs) - the done values, received by the agent
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        # - values: (n_steps, num_envs) - the values predicted by the agent
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

    def reshape_obs(self, next_obs):
        """
        This function reshapes the observation as obtained by the step() function of the environment to convert it
        to an observation that can be used as input in the deep set network.
        :param next_obs: The observation to be reshaped
        :return: A normalized tensor that has num_apps*num_actions lines and columns as many as the metrics of each microservice,
                 plus one column that signifies the action number.
        """
        # Normalize the environment
        next_obs = np.array(self.env.env_method('normalize', next_obs))
        # Reshape the observation
        next_obs = next_obs[0].reshape(1, -1)
        original_tensor = next_obs.reshape(self.num_envs, self.num_apps,
                                           (self.env.observation_space.shape[0] // self.num_apps))
        # Now we add an extra column that represents the action to be made.
        # It is an integer, from 0 to num_actions, normalized.

        # Duplicate the tuples along the second axis
        duplicated_tensor = np.repeat(original_tensor, self.num_actions, axis=1)

        # Create an array of alternating 0 and 1 values for each tuple
        extra_column = (np.arange(duplicated_tensor.shape[1]) % self.num_actions) / self.num_actions

        # Reshape the extra column to match the shape of duplicated_tensor
        extra_column_reshaped = extra_column.reshape(1, -1, 1)

        # Tile the extra column along the first axis to match the shape of duplicated_tensor
        extra_column_tiled = np.tile(extra_column_reshaped, (duplicated_tensor.shape[0], 1, 1))

        # Concatenate the duplicated_tensor and extra_column_tiled along the third axis
        tensor_with_extra_column = np.concatenate((duplicated_tensor, extra_column_tiled), axis=2)

        # Convert to tensor
        final_obs = torch.Tensor(tensor_with_extra_column).to(self.device)
        return final_obs

    def learn(self, tb_log_name=None, callback=None, total_timesteps: int = 500000):
        """
        Here is where the algorithm is executed. The algorithm runs for a specified number of steps, which is the
        total_timesteps parameter. At each step, the agent takes an action in the environment, receives the reward
        and the next observation, and stores the data in the tensors initialized in the constructor. When the number
        of steps reaches the batch size, the agent updates its parameters using the data collected.
        """
        total_reward = 0
        global_step = 0  # Initialize the global step counter
        start_time = time.time() # Initialize the timer

        # ALGO Logic: Once the environment is initialized the first observation is obtained.
        # It is a tensor of shape (num_envs, *obs_shape), where num_envs is the number of parallel environments
        # that are being used.
        # The obs_shape is the shape of the
        # observation space of the environment, which in our case is (12,).

        # The first observation, as obtained by resetting the environment.
        next_obs = self.env.reset()
        next_obs = self.reshape_obs(next_obs)
        # print(next_obs)

        # Initialize the tensor to store the state of the environment after the actions of the agent to zeros
        next_done = torch.zeros(self.num_envs).to(self.device)

        # Initialize the tensor to store the masks of the environment after the actions of the agent to zeros
        # next_masks = torch.tensor(np.array(self.env.env_method("action_masks")), dtype=torch.bool).to(self.device) ##

        # Calculate the updates that will be performed to the agent
        num_updates = total_timesteps // self.batch_size
        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so. Just skip for now.
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            # The code of each step. The step is performing the functioning of the DeepSet.
            for step in range(0, self.num_steps):
                # Increment the global step counter
                global_step += 1 * self.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done
                # self.masks[step] = next_masks ##

                # ALGO LOGIC: action logic

                # Here we use the agent to select an action (remember it already has a model loaded)
                # The action is a tensor of shape (num_envs, *action_shape), where num_envs is the number of parallel
                # environments that are being used.`

                with torch.no_grad():
                    # Get the action, the log probability of the action and the value of the state by the agent
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)  # ,masks=next_masks) ##

                    # Store the value of the state in the tensor
                    self.values[step] = value.flatten()

                # Store the action and the log probability of the action in the tensors
                # print('On step:', step)
                self.actions[step] = action
                # print('action:', action)
                self.logprobs[step] = logprob

                # ALGO LOGIC: action decoding
                # We use a MultiDiscrete action space, but the agent chooses one action. In the action chosen by the
                # agent, there is the information of both the number of pods to deploy/terminated
                # and in which microservice to do that. It is obtained as shown below.

                # Modify the action to be able t be used as a multibinary vector instead of a single one.
                valid_action = [[int(action[i]) // self.num_actions, int(action[i]) % self.num_actions] for i in range(self.num_envs)]
                # print('so action:', valid_action) ##

                # TRY NOT TO MODIFY: execute the environment and log data.

                # Makes the step in the environment, taking into account the action of the agent
                next_obs, reward, done, info = self.env.step(valid_action)
                total_reward += reward

                # Store the reward in the tensor
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)

                # Prepare the masks for the next iteration
                # next_masks = torch.tensor(np.array(self.env.env_method("action_masks")), dtype=torch.bool).to(self.device) ##

                # Prepare the obs and done for the next iteration
                next_obs = self.reshape_obs(next_obs)
                next_done = torch.Tensor(done).to(self.device) ##

                # Graph staff

            print(f"global_step: {global_step}, episodic_return={total_reward}")
            self.writer.add_scalar("rollout/reward", total_reward[0], global_step)
            total_reward = 0

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values            # flatten the batch
            b_obs = self.obs.reshape((-1, self.num_actions * self.num_apps, ((self.env.observation_space.shape[0]) // self.num_apps)+1)) ###
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1)) ###

            #  b_masks = self.masks.reshape((-1, +self.env.action_space.n))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []

            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]  # ,b_masks[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # Entropy loss
                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    # Agent optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

                # Explained Variance
                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
                self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
              #  print("FPS:", int(global_step / (time.time() - start_time)))
                self.writer.add_scalar("charts/FPS", int(global_step / (time.time() - start_time)), global_step)

                # Save the model on every end of an optimization cycle.
                self.save('run_1/'+'{}'.format(global_step))
    def predict(self, obs: npt.NDArray, masks: Optional[npt.NDArray] = None) -> npt.NDArray:
        """
        Get the model's action(s) from an observation
        """
        with torch.no_grad():
            action = self.agent.get_action(
                torch.as_tensor(obs, dtype=torch.float32), deterministic=True #, masks=torch.as_tensor(masks, dtype=torch.bool)
            ).numpy()
        return action

    def save(self, path: str) -> None:
        """
        Save the current parameters to file
        """
        torch.save(self.agent.state_dict(), path)
        return

    def load(self, path: str, reset_num_timesteps=False, verbose=1, tensorboard_log=None) -> None:
        """
        Load model parameters from file. The other parameters that are on the function are simply in order to use the
        same function for all the algorithms, since the stable version's algorithms require these parameters.
        """
        self.agent.load_state_dict(torch.load(path))
        return
