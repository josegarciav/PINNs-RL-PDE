CONFIG = {
    # Training settings
    'learning_rate': 0.001,
    'num_epochs': 5000,
    'num_points': 1000,             # Collocation points
    'num_boundary_points': 200,     # Boundary condition points
    'log_interval': 100,            # Log frequency
    'lambda_bc': 1.0,               # Boundary loss weight

    # Model architecture settings
    'input_dim': 2,                 # PDE input dimension (x, t)
    'hidden_dim': 64,
    'output_dim': 1,
    'num_layers': 4,
    'activation': 'tanh',

    # Reinforcement Learning settings
    'use_rl': False,
    'rl_learning_rate': 0.0005,
    'rl_gamma': 0.99,
    'rl_hidden_dim': 64,

    # Device settings
    'device': 'mps',                # Explicitly set device ('mps', 'cuda', 'cpu')

    # Visualization and checkpointing
    'log_interval': 100,            # Epoch logging interval
    'model_save_path': 'results/models/pinn_trained.pth',
    'results_path': 'results/',
}
