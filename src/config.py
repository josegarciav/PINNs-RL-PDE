
CONFIG = {
    # Training settings
    'learning_rate': 0.001,
    'num_epochs': 1000,
    'num_points': 100,
    'batch_size': 32,
    'log_interval': 100,

    # Neural network architecture
    'input_dim': 2,       # (x, t) or (S, t) for financial PDEs
    'hidden_dim': 64,     # Number of neurons per layer
    'output_dim': 1,      # Solution u(x, t)
    'activation': 'tanh', # Activation function

    # Reinforcement Learning settings
    'use_rl': True,       # Whether to use RL-based adaptive collocation
    'rl_learning_rate': 0.001,
    'rl_gamma': 0.99,     # Discount factor
}
