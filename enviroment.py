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
            if self.current_board[row][col] != 0:
                print(f"Cell ({row}, {col}) is already filled. No action taken.")
                return self.current_board, self.reward, self.done

            if num == self.solution_board[row][col]:
                self.current_board[row][col] = num
                self.reward += 10
                self.done = self.check_done()
                self.render()
                return self.current_board, self.reward, self.done

            else:
                if self.reward >= 30:
                    self.reward -= 30
                else:
                    self.reward = 0
                self.render()
                return self.current_board, self.reward, self.done





    def render(self):
        """
        Renders the current state of the board (for debugging or visualization purposes).
        """
        print(self.current_board)
