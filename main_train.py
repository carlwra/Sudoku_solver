import numpy as np
import torch
import matplotlib.pyplot as plt
import create_valid_sudoku_board as create_sudoku
import puzzle_creator
from enviroment import SudokuEnv
from agent import Agent

sudoku_generator = create_sudoku.create_valid_sudoku_board()
finished_board = sudoku_generator.create_board()

stars = "*" * 30
print(stars)
print("Initial filled-out Sudoku: \n")
print(finished_board)
print(stars)

puzzle_generator = puzzle_creator.SudokuPuzzleGenerator()
puzzle = puzzle_generator.create_puzzle(finished_board)
print("Sudoku puzzle with 23 clues: \n")
print(puzzle)
print(stars)

env = SudokuEnv(finished_board, puzzle)

input_dim = 9 * 9
action_dim = 9 * 9 * 9
agent = Agent(env, input_dim, action_dim)

num_episodes = 100000
max_steps_per_episode = 200
episode_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    for step in range(max_steps_per_episode):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.train(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {episode + 1}/{num_episodes} finished after {step + 1} steps")
            break

    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    episode_rewards.append(total_reward)

    print(f"Episode {episode + 1} reward: {total_reward}")

plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward over Episodes')
plt.show()

model_path = "sudoku_agent.pth"
torch.save(agent.model.state_dict(), model_path)
print(f"Model saved to {model_path}")
