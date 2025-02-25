# PPO-LSTM: Deep Reinforcement Learning with Memory

A deep reinforcement learning framework implementing Proximal Policy Optimization (PPO) with LSTM memory for partially observable environments. This project focuses on training agents that can solve tasks requiring memory and sequential decision-making abilities.

## Overview

This project implements a PPO (Proximal Policy Optimization) algorithm with LSTM (Long Short-Term Memory) networks to solve reinforcement learning problems that require memory. The system is designed to work with various environments, with a particular focus on partially observable environments like MiniGrid memory tasks and modified CartPole environments.

Key features:
- PPO algorithm implementation with LSTM for memory retention
- Support for various environments including MiniGrid, custom mazes, and CartPole variants
- Visualization tools for memory analysis and agent behavior
- Configurable hyperparameters via command line or sweep configuration
- Integration with Weights & Biases (wandb) for experiment tracking

## Project Structure

```
ppo_lstm/
├── checkpoints/                  # Saved model checkpoints
├── configs/                      # Configuration files
│   ├── sweep_config.yaml         # Sweep configuration for MiniGrid
│   └── sweep_config cartpole.yaml # Sweep configuration for CartPole
├── images/                       # Project images and diagrams
├── important_checkpoints/        # Key model checkpoints
├── notebooks/                    # Development notebooks (for exploration)
├── outputs/                      # Generated outputs
│   ├── logs/                     # Log files
│   ├── videos/                   # Recorded agent videos
│   └── visualizations/           # Visualization outputs
├── scripts/                      # Utility scripts
│   ├── experiments.sh            # Script for running multiple experiments
│   └── run_sweep.sh              # Script for running hyperparameter sweeps
├── src/                          # Source code
│   ├── environments/             # Environment implementations
│   │   ├── cartpole_no_vel.py    # CartPole without velocity information
│   │   ├── minigrid_custom_maze.py # Custom maze environment
│   │   ├── minigrid_memory.py    # Memory task environments
│   │   └── minigrid_obstructed.py # Obstructed environment
│   ├── ppo_lstm/                 # Core PPO-LSTM implementation
│   │   ├── agent.py              # Agent implementation
│   │   ├── buffer.py             # Experience buffer
│   │   ├── config.py             # Configuration classes
│   │   ├── main.py               # Main entry point
│   │   └── networks.py           # Neural network architectures
├── tests/                        # Test files
│   └── test_main.py              # Main test file
├── visualisations/               # Visualization tools
│   └── frame_extraction.py       # Tool for extracting memory visualization frames
└── requirements.txt              # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ppo_lstm
```

2. Create and activate a conda environment:
```bash
conda env create -f base_environment.yml
conda activate ppo_lstm
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Training an Agent

### Basic Usage

To train an agent with default parameters:

```bash
python -m src.ppo_lstm.main
```

### Custom Configuration

To train with custom parameters:

```bash
python -m src.ppo_lstm.main \
  --env-id MiniGrid-MemoryS13-v0 \
  --hidden-size 8 \
  --hidden-layer-size 16 \
  --seq-len 8 \
  --learning-rate 8e-4 \
  --num-steps 1024 \
  --total-steps 10000000 \
  --num-envs 16
```

### Running Multiple Experiments

To run multiple experiments in parallel:

```bash
bash scripts/experiments.sh
```

### Hyperparameter Sweep

To run a hyperparameter sweep using Weights & Biases:

```bash
# Start a new sweep
wandb sweep configs/sweep_config.yaml

# Run the sweep (replace SWEEP_ID with the actual ID)
bash scripts/run_sweep.sh -s SWEEP_ID -g 0 -n 4
```

## Visualization

### Memory Visualization

Generate visualization frames for analysis:

```bash
python -m visualisations.frame_extraction 5  # Generate 5 separate runs
```

## Environments

The project includes several environments:

1. **CartPoleNoVel**: A variant of CartPole where the velocity observation is hidden, requiring the agent to infer velocity from position history.

2. **MiniGrid Memory Tasks**: Tasks where the agent must remember information from earlier in the episode to make optimal decisions.

3. **Custom Mazes**: Maze navigation tasks that test the agent's ability to explore and remember the environment layout.

## Configuration

Key configuration parameters in `config.py`:

- **project_name**: Name of the project
- **env_id**: Environment ID to use
- **view_size**: Size of agent's field of view
- **hidden_size**: Size of LSTM hidden state
- **hidden_layer_size**: Size of hidden layers in actor/critic networks
- **total_steps**: Total number of training steps
- **seq_len**: Sequence length for LSTM training
- **num_envs**: Number of parallel environments for training
- **learning_rate**: Learning rate for optimization