# agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import random


class SudokuAgent(nn.Module):
    def __init__(self):
        """
        Initializes the agent's neural network model. The network takes in a Sudoku board as input
        and outputs a set of probabilities or values for each potential action (cell, number pair).
        """
        super(SudokuAgent, self).__init__()

        # Define your neural network layers here
        pass

    def forward(self, state):
        """
        Forward pass through the neural network.

        :param state: The current Sudoku board state (flattened 9x9 array).
        :return: A set of values representing the action space (probabilities or Q-values).
        """
        pass


class Agent:
    def __init__(self, env):
        """
        Initializes the RL agent that will interact with the Sudoku environment.

        :param env: The Sudoku environment the agent will interact with.
        """
        self.env = env
        self.model = SudokuAgent()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()  # Loss function (for Q-learning, this could change)

        # Epsilon-greedy exploration parameters
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Discount factor for future rewards

    def select_action(self, state):
        """
        Selects an action based on the current state using epsilon-greedy strategy.

        :param state: The current Sudoku board state.
        :return: The action (row, col, num) that the agent selects.
        """
        pass

    def train(self, state, action, reward, next_state, done):
        """
        Performs one step of training on the agent's neural network (e.g., Q-learning update).

        :param state: The current Sudoku board state.
        :param action: The action the agent took.
        :param reward: The reward received for the action.
        :param next_state: The next state after the action was taken.
        :param done: Whether the episode has finished (i.e., puzzle is solved).
        """
        pass
