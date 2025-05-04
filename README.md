# Deep Q-Network (DQN) for Flappy Bird

This project implements a Deep Q-Network (DQN) reinforcement learning agent to play Flappy Bird using PyTorch.

## Features
- 🎮 Plays Flappy Bird using Deep Reinforcement Learning
- 🧠 Implements Experience Replay for stable training
- ⚙️ Configurable hyperparameters via YAML file
- 📊 Tracks training progress and metrics
- 🖥️ Supports both training and demonstration modes

## Requirements
- Python 3.8+
- PyTorch
- Gymnasium
- Flappy Bird Gymnasium environment
- PyYAML

## Installation
```bash
pip install torch gymnasium flappy-bird-gymnasium pyyaml
```
## Run Trained Model
```bash
python agent.py flappybird2
```
## Train the Model
Edit parameters in hyperparams.yaml and run the following command to train the model:
```bash
python agent.py flappybird1 --train
```
## File Structure
flappy_bird_DQN/
├── agent.py            # Main DQN agent implementation
├── dqn.py              # Neural network architecture
├── experience_replay.py # Experience replay buffer
├── hyperparameters.yml  # Training configuration
├── runs/               # folder for saving training logs and checkpoints
└── README.md           # This file

## Demo
https://github.com/yourusername/yourrepo/blob/main/video.mp4
