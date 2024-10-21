import numpy as np
import torch
import matplotlib.pyplot as plt
import create_valid_sudoku_board as create_sudoku
import puzzle_creator

sudoku_generator = create_sudoku.create_valid_sudoku_board()
finished_board = sudoku_generator.create_board()

stars = "*"*30

print(stars)
print("Initial filled out sodoku: \n")
print(finished_board)
print(stars)

puzzle_generator = puzzle_creator.SudokuPuzzleGenerator()
puzzle = puzzle_generator.create_puzzle(finished_board)
print(puzzle)
print(stars)
