import copy

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import gymnasium as gym
from gymnasium.envs.toy_text import blackjack
from gymnasium.envs.toy_text.blackjack import sum_hand, usable_ace as usable_ace_func
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

random.seed(666)
np.random.seed(666)


# class ExpectimaxAgent:
#     def __init__(self, env, gamma):
#         self.env = env
#
#         self.__reinitialize_variables()
#
#         self.gamma = gamma
#
#     def train_agent(self, num_episodes):
#         self.__reinitialize_variables()
#         optimal_policy = defaultdict(lambda: np.zeros(self.env.action_space.n))
#
#         state, info = self.env.reset()
#
#         for _ in tqdm(range(1, num_episodes + 1)):
#             episode_complete_flag = False
#
#             while not episode_complete_flag:
#                 action = self.__select_appropriate_action(state)
#                 new_state, reward, episode_complete_flag, truncated, info = self.env.step(action)
#
#                 self.__update_q_values(state, action, new_state, reward)
#
#                 state = new_state
#
#             state, info = self.env.reset()
#
#         for state in self.q_values.keys():
#             optimal_policy[state] = np.argmax(self.q_values[state])
#
#         return self.q_values, optimal_policy
#
#     def __select_appropriate_action(self, state):
#         return np.argmax(self.q_values[state])
#
#     def __reinitialize_variables(self):
#         self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
#         self.num_value_updates = defaultdict(lambda: np.zeros(self.env.action_space.n))
#         self.training_error = []
#
#     def get_training_error(self):
#         return self.training_error
#
#     def __update_q_values(self, state, action, new_state, reward):
#         eta = 1 / (1 + self.num_value_updates[state][action])
#         v_opt_new_state = max(self.q_values[new_state])
#         temporal_difference = reward + self.gamma * v_opt_new_state - self.q_values[state][action]
#         self.q_values[state][action] = self.q_values[state][action] + eta * temporal_difference
#         self.training_error.append(temporal_difference)
#
#
# n_episodes = 10000
# blackjack_env = gym.make('Blackjack-v1', natural=True, sab=False, render_mode='rgb_array')
# blackjack_env = gym.wrappers.RecordEpisodeStatistics(blackjack_env, buffer_length=n_episodes)
# exAgent = ExpectimaxAgent(blackjack_env, 1)
# q_values, optimal_policy = exAgent.train_agent(n_episodes)
#
# roll_length = 500

class CloneableBlackjackEnv(blackjack.BlackjackEnv):
    def __init__(self):
        super().__init__(natural=True, sab=False, render_mode='rgb_array')
        self.dealer = None
        self.player = None
        self.usable_ace = None

    def get_state(self):
        """Returns the current state."""
        return sum_hand(self.player), self.dealer[0], usable_ace_func(self.player)

    def set_state(self, player, dealer, usable_ace):
        """Sets the state to the provided values."""
        self.player = [player]
        self.dealer = [dealer]
        self.usable_ace = usable_ace

    def is_done(self):
        """Check if the game is done."""
        return self.dealer[0] + self.dealer[1] >= 17

    def get_reward(self):
        """
        Retrieve the current reward.

        Note: In Blackjack, the reward is returned as part of the environment's step method.
        """
        # Return 0 if the game is not done, as the reward is not yet available
        if not self.is_done():
            return 0

        # Get the reward from the last step
        _, reward, _, _, _ = self.step(0)  # Stand action to trigger the dealer's move
        return reward


# Now create an instance of your custom environment
blackjack_env = CloneableBlackjackEnv()


def clone_env(env):
    return env.get_state()


def restore_env(env, state):
    env.set_state(*state)


n_episodes = 10000
# blackjack_env = gym.make('Blackjack-v1', natural=True, sab=False, render_mode='rgb_array')
# blackjack_env = gym.wrappers.RecordEpisodeStatistics(blackjack_env, buffer_length=n_episodes)


def expectimax(env, depth, is_player_turn):
    if depth == 0 or env.is_done():
        return env.get_reward()

    initial_incoming_state = copy.deepcopy(clone_env(env))
    if is_player_turn:  # Player's decision node
        # Option to hit
        state, reward, done, _, _ = env.step(1)  # '1' is typically the action for 'hit'
        if done:
            hit_value = reward
        else:
            hit_value = expectimax(env, depth - 1, True)
        # Reset to the original state (we'll need a mechanism to store/replay states)
        env.reset()
        env.set_state(*initial_incoming_state)

        # Option to stand
        state, reward, done, _, _ = env.step(0)  # '0' for 'stand'
        if done:
            stand_value = reward
        else:
            stand_value = expectimax(env, depth - 1, False)
        env.reset()  # Reset to the original state

        return max(hit_value, stand_value)
    else:  # Dealer's turn (chance node)
        total_value = 0
        # Assuming equal probability for each card from 1 to 10 (simplified model)
        for card in range(1, 11):
            # Simulate the dealer drawing a specific card (pseudo-code; actual simulation might vary)
            state, reward, done, _, _ = env.step(card)  # This is not a valid call in Blackjack-v1
            if done:
                total_value += reward
            else:
                total_value += expectimax(env, depth - 1, False)
            env.reset()  # Reset to original state

        return total_value / 10  # Average of possible outcomes


def best_move(env):
    # Simulate hitting
    saved_state = copy.deepcopy(clone_env(env))  # Hypothetical method to clone the env state
    new_state, reward, done, _, _ = env.step(1)  # Hit
    if not done:
        hit_value = expectimax(env, 2, True)
    else:
        hit_value = reward

    # Simulate standing
    env.reset()
    env.set_state(*saved_state)
    _, reward, done, _, _ = env.step(0)  # Stand
    if not done:
        stand_value = expectimax(env, 2, False)
    else:
        stand_value = reward

    return 1 if hit_value > stand_value else 0


def play_game():
    done = False
    blackjack_env.reset()
    while not done:
        action = best_move(blackjack_env)
        print(f"Player decision: {action}")
        state, reward, done, _, _ = blackjack_env.step(action)
        print(f"Current state: {state}, Reward: {reward}")
        if done:
            print(f"Game over. Final reward: {reward}")


play_game()

# def expectimax(state, depth, env):
#     if depth == 0 or env.is_bust(state[0]):
#         return -1 if env.is_bust(state[0]) else env.get_score(state[0])
#
#     if env.turn(state) == 1:  # Player's turn
#         # Decision node: Hit or Stand
#         hit_value = expectimax(env.step(state, 'hit')[0], depth - 1, env)
#         stand_value = expectimax(env.step(state, 'stand')[0], depth - 1, env)
#         return max(hit_value, stand_value)
#     else:  # Dealer's turn
#         # Chance node: simulate possible cards
#         total_value = 0
#         possible_cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
#         for card in possible_cards:
#             next_state = env.step(state, card)
#             total_value += expectimax(next_state[0], depth - 1, env) / len(possible_cards)
#         return total_value
#
#
# def best_move(state, env):
#     hit_value = expectimax(env.step(state, 1)[0], 2, env)
#     stand_value = expectimax(env.step(state, 0)[0], 2, env)
#     return 'hit' if hit_value > stand_value else 'stand'
#
#
# def play_game():
#     state = blackjack_env.reset()
#     done = False
#     while not done:
#         action = best_move(state, blackjack_env)
#         print(f"Player decision: {action}")
#         state, reward, done, _ = blackjack_env.step(action)
#         print(f"Current state: {state}, Reward: {reward}")
#         if done:
#             print(f"Game over. Final reward: {reward}")
#
#
# play_game()


# def make_training_plots(rolling_length, filename):
#     fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
#     axs[0].set_title("Episode rewards")
#     # compute and assign a rolling average of the data to provide a smoother graph
#     reward_moving_average = (
#             np.convolve(
#                 np.array(blackjack_env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
#             )
#             / rolling_length
#     )
#     axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
#     axs[1].set_title("Episode lengths")
#     length_moving_average = (
#             np.convolve(
#                 np.array(blackjack_env.length_queue).flatten(), np.ones(rolling_length), mode="same"
#             )
#             / rolling_length
#     )
#     axs[1].plot(range(len(length_moving_average)), length_moving_average)
#     axs[2].set_title("Training Error")
#     training_error_moving_average = (
#             np.convolve(np.array(exAgent.get_training_error()), np.ones(rolling_length), mode="same")
#             / rolling_length
#     )
#     axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.show()
#
#
# make_training_plots(roll_length, "training_plots_expectimax.png")
#
#
# def generate_policy_grid(agent, usable_ace=False):
#     policy = defaultdict(int)
#     for obs, action_values in agent.q_values.items():
#         policy[obs] = int(np.argmax(action_values))
#
#     player_count, dealer_count = np.meshgrid(
#         # players count, dealers face-up card
#         np.arange(12, 22),
#         np.arange(1, 11),
#     )
#
#     # create the policy grid for plotting
#     policy_grid = np.apply_along_axis(
#         lambda obs: policy[(obs[0], obs[1], usable_ace)],
#         axis=2,
#         arr=np.dstack([player_count, dealer_count]),
#     )
#     return policy_grid
#
#
# def create_plots(policy_grid, title, filename):
#     fig, ax = plt.subplots()
#     im = ax.imshow(policy_grid, cmap='Pastel1')  # Set3 with black, Pastel1 with black
#
#     ax.set_xticks(range(0, 10), labels=list(map(str, range(12, 22))))
#     ax.set_yticks(range(0, 10), ["A"] + list(map(str, range(2, 11))))
#
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
#     for i in range(policy_grid.shape[0]):
#         for j in range(policy_grid.shape[1]):
#             ax.text(j, i, policy_grid[i, j],
#                     ha="center", va="center", color="black")
#
#     ax.set_title("Harvest of local farmers (in tons/year)")
#     fig.tight_layout()
#     plt.title(title)
#     plt.savefig(filename)
#     plt.show()
#
#
# # policy with usable ace (ace counts as 1 or 11)
# policy_grid = generate_policy_grid(exAgent, usable_ace=True)
# create_plots(policy_grid, title="With usable ace (Expectimax)", filename="usable_ace_policy_heatmap_expectimax.png")
#
# # policy without usable ace
# policy_grid = generate_policy_grid(exAgent, usable_ace=False)
# create_plots(policy_grid, title="Without usable ace (Expectimax)", filename="unusable_ace_policy_heatmap_expectimax.png")
