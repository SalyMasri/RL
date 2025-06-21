# Reinforcement Learning: Q-Learning Agent

## Project Overview
This project implements a **Q-Learning agent** for a grid-based environment. The goal is to optimize the agent's decision-making process by balancing **exploration** and **exploitation** using reinforcement learning techniques. The agent learns an optimal policy to maximize rewards while navigating through various states.

## Technologies & Tools Used
- **Programming Language**: Python
- **Libraries**: NumPy
- **Algorithm**: Q-Learning
- **Optimization Methods**: Epsilon-Greedy Strategy, Bellman Update

## Key Features
- **State Representation**: The environment is represented as a **discretized grid** where actions correspond to movement.
- **Action Selection**: Implemented **Epsilon-Greedy Policy** to control exploration vs. exploitation.
- **Q-Table Initialization**: Uses **NaN masking** to prevent invalid moves.
- **Bellman Update**: Computes **Q-values** iteratively to converge towards the optimal policy.
- **Early Stopping Criterion**: Convergence is determined using **mean Q-value change**.

## Results
- The agent successfully **learns an optimal policy** to achieve the highest possible reward.  
- Performance is influenced by **learning rate (α), discount factor (γ), and epsilon decay**.  
- Different hyperparameters significantly affect the training efficiency and final policy stability.
