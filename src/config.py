
CONFIG = {
    # Training settings
    'learning_rate': 0.0001,
    'num_epochs': 3000,
    'num_points': 5000,             # Collocation points
    'num_boundary_points': 4000,    # Boundary condition points
    'log_interval': 100,            # Log frequency of iterations
    'lambda_bc': 10,                # Boundary loss weight

    # Model architecture settings
    'input_dim': 2,                 # PDE input dimension (x, t)
    'hidden_dim': 126,
    'output_dim': 1,
    'num_layers': 6,
    'activation': 'tanh',

    # Reinforcement Learning settings
    'use_rl': True,
    'rl_learning_rate': 0.0005,
    'rl_gamma': 0.99,
    'rl_hidden_dim': 64,

    # Device settings and paths
    'device': 'mps', # Explicitly set device ('mps', 'cuda', 'cpu')
    'model_save_path': 'results/models/pinn_trained.pth',
    'results_path': 'results/',
}
