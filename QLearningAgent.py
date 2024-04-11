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
    def __init__(self, env, gamma, epsilon, epsilon_decay_factor):
        self.env = env

        self.__reinitialize_variables()

        self.gamma = gamma
        self.original_epsilon = epsilon
        self.epsilon_decay_factor = epsilon_decay_factor

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

                self.__update_q_values(state, action, new_state, reward)

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

    def __update_q_values(self, state, action, new_state, reward):
        eta = 1 / (1 + self.num_value_updates[state][action])
        v_opt_new_state = max(self.q_values[new_state])
        temporal_difference = reward + self.gamma * v_opt_new_state - self.q_values[state][action]
        self.q_values[state][action] = self.q_values[state][action] + eta * temporal_difference
        self.training_error.append(temporal_difference)

    def __update_parameters(self, state, action, current_epsilon, new_state):
        self.num_value_updates[state][action] += 1
        current_epsilon *= self.epsilon_decay_factor
        return current_epsilon, new_state


n_episodes = 10000000
blackjack_env = gym.make('Blackjack-v1', natural=True, sab=False, render_mode='rgb_array')
blackjack_env = gym.wrappers.RecordEpisodeStatistics(blackjack_env, buffer_length=n_episodes)
qlAgent = QLearningAgent(blackjack_env, 1, 1, 0.999)
q_values, optimal_policy = qlAgent.train_agent(n_episodes)

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


make_training_plots(roll_length, "testing.png")


def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig


# state values & policy with usable ace (ace counts as 11)
value_grid, policy_grid = create_grids(qlAgent, usable_ace=True)
fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
plt.savefig("usable_ace_policy_gamma1_epsilon1.png")
plt.show()

value_grid, policy_grid = create_grids(qlAgent, usable_ace=False)
fig2 = create_plots(value_grid, policy_grid, title="Without usable ace")
plt.savefig("unusable_ace_policy_gamma1_epsilon1.png")
plt.show()
