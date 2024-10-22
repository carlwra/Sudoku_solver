# environment.py

import numpy as np
from flask import current_app


class SudokuEnv:
    def __init__(self, puzzle_board, solution_board):
        """
        Initializes the Sudoku environment.

        :param puzzle_board: The initial state of the Sudoku puzzle (partially filled board).
        :param solution_board: The full solution to the puzzle.
        """
        self.reward = None
        self.puzzle_board = puzzle_board
        self.solution_board = solution_board
        self.current_board = np.copy(puzzle_board)  # Copy to track progress
        self.done = False

    def reset(self):
        """
        Resets the environment to the initial puzzle state.

        :return: The initial puzzle board (current_board).
        """
        self.current_board = np.copy(self.puzzle_board)
        self.done = False
        self.reward = 0
        return self.current_board

    def check_done(self):
        for i in self.current_board:
            for j in i:
                if j == 0:
                    return False
        return True

    def step(self, action):
        """
        Takes an action and updates the environment.
        :param action: A tuple (row, col, num) representing the cell to fill and the number to place.
        :return: A tuple (new_state, reward, done).
                 - new_state: The updated board after the action.
                 - reward: The reward based on the agent's action.
                 - done: A boolean indicating whether the puzzle is solved.
        """
        if action is not None:
            row, col, num = action
            if self.current_board[row][col] == 0:  # Make sure it's not a filled cell
                if num == self.solution_board[row][col]:
                    self.current_board[row][col] = num
                    reward = 80
                    done = self.check_done()
                    return self.current_board, reward, done
                else:
                    reward = -30
                    return self.current_board, reward, False
            else:
                # If the cell is already filled, return a negative reward
                reward = -1
                return self.current_board, reward, False
        # If action is None, return the current state with no reward
        return self.current_board, 0, False


    def render(self):
        """
        Renders the current state of the board (for debugging or visualization purposes).
        """
        print(self.current_board)
