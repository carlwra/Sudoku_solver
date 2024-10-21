import numpy as np
import random


def is_valid(board, row, col, num):
    """Check if placing num in board[row][col] is valid according to Sudoku rules."""
    if num in board[row]:
        return False

    if num in board[:, col]:
        return False

    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in board[start_row:start_row+3, start_col:start_col+3]:
        return False

    return True


class create_valid_sudoku_board:
    def __init__(self):
        pass

    def fill_board(self, board):
        """Fill the board with valid numbers, creating a complete Sudoku. Function is self recursive."""

        for row in range(9):
            for col in range(9):
                if board[row, col] == 0:

                    nums = list(range(1, 10))
                    random.shuffle(nums)  # Shuffle to add randomness till valid number

                    for num in nums:
                        if is_valid(board, row, col, num):
                            board[row, col] = num
                            if self.fill_board(board):
                                return True
                            board[row, col] = 0
                    return False
        return True

    def create_board(self):
        """This function creates a valid sudoku board.
        It returns a 9x9 ndarray."""
        board = np.zeros((9, 9), dtype=int)
        self.fill_board(board)
        return board


