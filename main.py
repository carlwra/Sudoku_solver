import torch
import numpy as np
import create_valid_sudoku_board as create_sudoku
import puzzle_creator
from agent import Agent, SudokuAgent
from enviroment import SudokuEnv


def load_pretrained_model(model_path, action_dim):
    """
    Load the pre-trained model from the specified path.

    :param model_path: Path to the pre-trained model file (e.g., model.pth).
    :param action_dim: Number of possible actions.
    :return: Loaded model.
    """
    model = SudokuAgent(action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def run_pretrained_agent(env, agent):
    """
    Run the pre-trained agent on the environment.

    :param env: The Sudoku environment.
    :param agent: The RL agent.
    """
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        if action is None:
            break
        state, reward, done = env.step(action)
        print(f"Action: {action}, Reward: {reward}")
        print(f"Current board:\n{state}")

if __name__ == "__main__":
    """
    Main entry point for running the Sudoku agent with a pre-trained model.
    """
    input_dim = 81
    action_dim = 9 * 9 * 9

    sudoku_generator = create_sudoku.create_valid_sudoku_board()
    finished_board = sudoku_generator.create_board()
    puzzle_generator = puzzle_creator.SudokuPuzzleGenerator()
    puzzle = puzzle_generator.create_puzzle(finished_board)

    env = SudokuEnv(puzzle, finished_board)

    agent = Agent(env, input_dim, action_dim)

    model_path = 'sudoku_agent.pth'
    model = load_pretrained_model(model_path, action_dim)
    agent.model = model

    run_pretrained_agent(env, agent)
