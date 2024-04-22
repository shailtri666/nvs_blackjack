import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

random.seed(666)
np.random.seed(666)


class QLearningAgent:
    def __init__(self, env, gamma, epsilon, epsilon_decay_factor, learning_rate):
        self.env = env

        self.__reinitialize_variables()

        self.gamma = gamma
        self.original_epsilon = epsilon
        self.epsilon_decay_factor = epsilon_decay_factor
        self.learning_rate = learning_rate

    def train_agent(self, num_episodes):
        self.__reinitialize_variables()
        optimal_policy = defaultdict(lambda: np.zeros(self.env.action_space.n))

        state, info = self.env.reset()
        current_epsilon = self.original_epsilon

        for _ in tqdm(range(1, num_episodes + 1)):
            episode_complete_flag = False

            while not episode_complete_flag:
                action = self.__select_appropriate_action(current_epsilon, state)
                new_state, reward, episode_complete_flag, truncated, info = self.env.step(action)

                self.__update_q_values(state, action, new_state, reward, truncated)

                current_epsilon, state = self.__update_parameters(state, action, current_epsilon, new_state)

            state, info = self.env.reset()

        for state in self.q_values.keys():
            optimal_policy[state] = np.argmax(self.q_values[state])

        return self.q_values, optimal_policy

    def __select_appropriate_action(self, current_epsilon, state):
        if random.uniform(0, 1) < current_epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_values[state])

    def __reinitialize_variables(self):
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.num_value_updates = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.training_error = []

    def get_training_error(self):
        return self.training_error

    def __update_q_values(self, state, action, new_state, reward, truncated):
        # eta = 1 / (1 + self.num_value_updates[state][action])
        v_opt_new_state = (not truncated) * max(self.q_values[new_state])
        # v_opt_new_state = max(self.q_values[new_state])
        temporal_difference = reward + self.gamma * v_opt_new_state - self.q_values[state][action]
        self.q_values[state][action] = self.q_values[state][action] + self.learning_rate * temporal_difference
        # self.q_values[state][action] = self.q_values[state][action] + eta * temporal_difference
        self.training_error.append(temporal_difference)

    def __update_parameters(self, state, action, current_epsilon, new_state):
        self.num_value_updates[state][action] += 1
        current_epsilon -= self.epsilon_decay_factor
        return current_epsilon, new_state


# Configuration D - i_epsilon 1, gamma 1, epsilon_decay i_epsilon/n_episodes, 0.0000001, 41.625
# Configuration A - i_epsilon 0.99, gamma 0.9, epsilon_decay *0.9, eta = 1/1+num_updates(s, a), 38.557
# Configuration B - i_epsilon 0.99, gamma 1, epsilon_decay *0.9, eta = 1/1+num_updates(s, a), 38.025
# Configuration C - i_epsilon 0.99, gamma 1, epsilon_decay -0.0009, 0.0001, 41.513
# Configuration F - i_epsilon 0.99, gamma 1, epsilon_decay i_epsilon/n_episodes, 0.0001, 42.099
# Configuration E - i_epsilon 1, gamma 1, epsilon_decay i_epsilon/n_episodes, 0.0001, 41.634
initial_epsilon = 1
n_episodes = 1000000
blackjack_env = gym.make('Blackjack-v1', natural=True, sab=False, render_mode='rgb_array')
blackjack_env = gym.wrappers.RecordEpisodeStatistics(blackjack_env, buffer_length=n_episodes)
qlAgent = QLearningAgent(blackjack_env, 1, initial_epsilon, initial_epsilon / n_episodes, 0.0000001)
q_values, optimal_policy = qlAgent.train_agent(n_episodes)


def get_win_rate(policy, num_games, env):
    state, _ = env.reset()
    num_wins = 0

    for _ in tqdm(range(num_games)):
        episode_complete_flag = False
        reward = 0

        while not episode_complete_flag:
            action = policy[state]
            state, reward, episode_complete_flag, truncated, info = env.step(action)

        if reward > 0:
            num_wins += 1

        state, info = env.reset()

    return num_wins * 100 / num_games


print("Q-Learning Agent's win rate: ", get_win_rate(optimal_policy, 100000, blackjack_env))

roll_length = 500


def make_training_plots(rolling_length, filename):
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Episode rewards")
    # compute and assign a rolling average of the data to provide a smoother graph
    reward_moving_average = (
            np.convolve(
                np.array(blackjack_env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
            )
            / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[1].set_title("Episode lengths")
    length_moving_average = (
            np.convolve(
                np.array(blackjack_env.length_queue).flatten(), np.ones(rolling_length), mode="same"
            )
            / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[2].set_title("Training Error")
    training_error_moving_average = (
            np.convolve(np.array(qlAgent.get_training_error()), np.ones(rolling_length), mode="same")
            / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


make_training_plots(roll_length, "qLearning_training_plots.png")


def generate_policy_grid(agent, usable_ace=False):
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(5, 22),
        np.arange(1, 11),
    )

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return policy_grid


def create_plots(policy_grid, title, filename):
    fig, ax = plt.subplots()
    im = ax.imshow(policy_grid, cmap='Pastel1')  # Set3 with black, Pastel1 with black

    ax.set_xticks(range(0, 17), labels=list(map(str, range(5, 22))))
    ax.set_yticks(range(0, 10), ["A"] + list(map(str, range(2, 11))))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(policy_grid.shape[0]):
        for j in range(policy_grid.shape[1]):
            ax.text(j, i, policy_grid[i, j],
                    ha="center", va="center", color="black")

    # ax.set_title("")
    fig.tight_layout()
    plt.title(title)
    plt.savefig(filename)
    plt.show()


# policy with usable ace (ace counts as 1 or 11)
policy_grid = generate_policy_grid(qlAgent, usable_ace=True)
create_plots(policy_grid, title="With usable ace", filename="qLearning_usable_ace_policy_heatmap.png")

# policy without usable ace
policy_grid = generate_policy_grid(qlAgent, usable_ace=False)
create_plots(policy_grid, title="Without usable ace", filename="qLearning_unusable_ace_policy_heatmap.png")
