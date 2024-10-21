import numpy as np
import random

class SudokuPuzzleGenerator:
    def __init__(self, clues=23):
        self.clues = clues  # number of cells to keep

    def create_puzzle(self, board):
        """Creates a Sudoku puzzle with a certain number of clues."""
        full_board = board.copy()

        puzzle_board = full_board.copy()

        total_cells = 9 * 9
        cells_to_remove = total_cells - self.clues

        for _ in range(cells_to_remove):
            row, col = random.randint(0, 8), random.randint(0, 8)
            while puzzle_board[row, col] == 0:
                row, col = random.randint(0, 8), random.randint(0, 8)
            puzzle_board[row, col] = 0

        return puzzle_board
