import random
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


class SudokuAgent(nn.Module):
    def __init__(self, input_dim, action_dim):
        """
        Neural network to predict Q-values for each action.

        :param input_dim: Input dimension (for Sudoku, it would be 9x9 = 81 flattened cells).
        :param action_dim: Number of possible actions (9x9 board with 9 possible numbers per cell).
        """
        super(SudokuAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 128)  # Second fully connected layer
        self.fc3 = nn.Linear(128, action_dim)  # Output layer

    def forward(self, x):
        """
        Forward pass for the network.

        :param x: Input state.
        :return: Predicted Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, env, input_dim, action_dim):
        """
        Initialize the agent.

        :param env: The Sudoku environment.
        :param input_dim: Input dimensions for the model.
        :param action_dim: Number of possible actions.
        """
        self.env = env
        self.model = SudokuAgent(input_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.empty_cells = []

    def update_empty_cells(self, state):
        """
        Update the list of empty cells.

        :param state: Current Sudoku board.
        """
        self.empty_cells = [(row, col) for row in range(9) for col in range(9) if state[row][col] == 0]

    def select_action(self, state):
        """
        Select an action based on epsilon-greedy strategy.

        :param state: Current Sudoku board.
        :return: Selected action as a tuple (row, col, num), or None if no empty cells.
        """
        self.update_empty_cells(state)
        if not self.empty_cells:
            return None
        if np.random.rand() <= self.epsilon:
            row, col = random.choice(self.empty_cells)
            num = random.randint(1, 9)
            return (row, col, num)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
            q_values = self.model(state_tensor)
            best_action = None
            best_q_value = float('-inf')
            for (row, col) in self.empty_cells:
                for num in range(1, 10):
                    action_idx = row * 81 + col * 9 + (num - 1)
                    q_value = q_values[0, action_idx].item()
                    if q_value > best_q_value:
                        best_q_value = q_value
                        best_action = (row, col, num)
            return best_action

    def train(self, state, action, reward, next_state, done):
        """
        Train the agent by updating the model based on the given experience.

        :param state: Current state (Sudoku board).
        :param action: The action taken by the agent.
        :param reward: Reward for the action.
        :param next_state: State after the action.
        :param done: Whether the episode is finished.
        """
        if action is None:
            return
        state_tensor = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).flatten().unsqueeze(0)
        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor).detach()
        target = reward + self.gamma * torch.max(next_q_values) * (1 - int(done))
        row, col, num = action
        action_idx = row * 81 + col * 9 + (num - 1)
        loss = self.criterion(q_values[0, action_idx], target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
