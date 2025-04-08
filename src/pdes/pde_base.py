# Base class for PDE implementations
# Provides common functionality for all PDEs

import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Union, Set
import numpy as np
from scipy.stats import qmc
from src.rl_agent import RLAgent
import matplotlib.pyplot as plt
import logging
import os


@dataclass
class PDEConfig:
    """Configuration dataclass for PDE parameters."""

    name: str
    domain: Union[
        Tuple[float, float], List[Tuple[float, float]]
    ]  # For 1D, list of tuples for higher dimensions
    time_domain: Tuple[float, float]
    parameters: Dict[str, float]
    boundary_conditions: Dict[str, Dict[str, Any]]
    initial_condition: Dict[str, Any]
    exact_solution: Dict[str, Any]
    dimension: int = 1  # Default to 1D
    input_dim: Optional[int] = None  # Input dimensions for the NN (spatial + temporal)
    output_dim: Optional[int] = (
        None  # Output dimensions for the NN (solution components)
    )
    architecture: Optional[str] = None  # Neural network architecture for this PDE
    device: Optional[torch.device] = None
    training: Optional[Dict[str, Any]] = None  # Training configuration


class PDEBase:
    """
    Base class for defining PDEs for Physics-Informed Neural Networks (PINNs).
    Provides common functionality for all PDE implementations.
    """

    @staticmethod
    def create(
        pde_type: str, config: Optional[PDEConfig] = None, **kwargs
    ) -> "PDEBase":
        """
        Factory method to create PDE instances from type and parameters.

        :param pde_type: Type of PDE to create (e.g., 'heat', 'wave', 'burgers')
        :param config: Optional PDEConfig instance, if None one will be created from kwargs
        :param kwargs: Parameters specific to the PDE type
        :return: Instantiated PDE object
        """
        from importlib import import_module
        from inspect import isclass

        # Convert pde_type to class name format (e.g., 'heat' -> 'HeatEquation')
        if "_" in pde_type:
            # Handle names like 'heat_equation' -> 'HeatEquation'
            pde_class_name = "".join(word.capitalize() for word in pde_type.split("_"))
            if not pde_class_name.endswith("Equation"):
                pde_class_name += "Equation"
        else:
            # Handle names like 'heat' -> 'HeatEquation'
            pde_class_name = pde_type.capitalize() + "Equation"

        # Try alternative class name formats if needed
        alternative_class_names = [
            pde_class_name,
            pde_type.capitalize(),  # For cases like 'Pendulum' instead of 'PendulumEquation'
            "".join(
                word.capitalize() for word in pde_type.split("_")
            ),  # CamelCase without 'Equation'
        ]

        # Try to import and instantiate the appropriate PDE class
        for class_name in alternative_class_names:
            try:
                # Import from module, assuming module name is lowercase
                module_path = f"src.pdes.{pde_type.lower().replace('equation', '')}"
                if module_path.endswith("_"):
                    module_path = module_path[:-1]

                module = import_module(module_path)

                # Get the PDE class
                pde_class = getattr(module, class_name)

                # Check if it's a proper class that inherits from PDEBase
                if isclass(pde_class) and issubclass(pde_class, PDEBase):
                    # If no config is provided, create one from kwargs
                    if config is None:
                        # Extract required parameters for PDEConfig
                        config_params = {
                            "name": kwargs.pop("name", f"{class_name}"),
                            "domain": kwargs.pop("domain", [(0.0, 1.0)]),
                            "time_domain": kwargs.pop("time_domain", (0.0, 1.0)),
                            "parameters": kwargs.pop("parameters", {}),
                            "boundary_conditions": kwargs.pop(
                                "boundary_conditions", {}
                            ),
                            "initial_condition": kwargs.pop("initial_condition", {}),
                            "exact_solution": kwargs.pop("exact_solution", {}),
                            "dimension": kwargs.pop("dimension", 1),
                            "input_dim": kwargs.pop("input_dim", None),
                            "output_dim": kwargs.pop("output_dim", None),
                            "architecture": kwargs.pop("architecture", None),
                            "device": kwargs.pop("device", None),
                            "training": kwargs.pop("training", None),
                        }
                        config = PDEConfig(**config_params)

                    # Instantiate PDE with config and remaining kwargs
                    return pde_class(config=config, **kwargs)
            except (ImportError, AttributeError) as e:
                # Continue trying with other class names
                continue

        # If we get here, no valid PDE class was found
        raise ValueError(f"Could not find PDE implementation for type: {pde_type}")

    def __init__(self, config: PDEConfig, rl_agent=None):
        """
        Initialize PDE with given configuration.

        :param config: Configuration dataclass
        :param rl_agent: Optional RL agent for adaptive sampling
        """
        self.config = config
        self.domain = config.domain
        self.rl_agent = rl_agent  # Store RL agent

        # Handle domain configuration in both old and new formats
        if isinstance(self.domain, list):
            if len(self.domain) > 0:
                if isinstance(self.domain[0], (list, tuple)):
                    # Convert any lists to tuples for consistency
                    self.domain = [(float(d[0]), float(d[1])) for d in self.domain]
                else:
                    # Old format: [xmin, xmax] -> [(xmin, xmax)]
                    self.domain = [(float(self.domain[0]), float(self.domain[1]))]
        else:
            # Default domain if none provided
            self.domain = [(0.0, 1.0)]

        # Update config with normalized domain format
        self.config.domain = self.domain

        # Handle time_domain/t_domain property
        if hasattr(config, "time_domain"):
            self.time_domain = config.time_domain
        elif hasattr(config, "t_domain"):
            self.time_domain = config.t_domain
        else:
            self.time_domain = [0.0, 1.0]

        # Convert time_domain to tuple if it's a list
        if isinstance(self.time_domain, list):
            self.time_domain = tuple(self.time_domain)

        # Setup device - Handle device explicitly and carefully
        if hasattr(config, 'device') and config.device is not None:
            # If config.device is already a torch.device object, use it directly
            if isinstance(config.device, torch.device):
                self.device = config.device
            else:
                # Try to convert string to torch.device
                try:
                    self.device = torch.device(str(config.device))
                except:
                    print(f"Warning: Invalid device '{config.device}', falling back to CPU")
                    self.device = torch.device("cpu")
        else:
            # Fallback to CPU if no device specified
            self.device = torch.device("cpu")
            
        # Log the device being used
        print(f"PDE initialized with device: {self.device}")
        
        # Ensure config has the same device
        self.config.device = self.device

        # Store dimensionality
        self.dimension = config.dimension

        # Make sure parameters exist but preserve any existing parameters
        if not hasattr(config, "parameters"):
            config.parameters = {}
        elif config.parameters is None:
            config.parameters = {}

        # Setup boundary and initial conditions
        self._setup_boundary_conditions()

        # Setup validation points
        self._setup_validation_points()

        # For tracking point distribution
        self.collocation_history = []

        # Setup neural network parameters if not specified
        if self.config.input_dim is None:
            self.config.input_dim = self.dimension + 1  # Spatial dimensions + time

        if self.config.output_dim is None:
            self.config.output_dim = 1  # Default to single output (u)

    def _setup_boundary_conditions(self):
        """Set up boundary condition functions from configuration."""
        self.boundary_conditions = {}

        if (
            hasattr(self.config, "boundary_conditions")
            and self.config.boundary_conditions
        ):
            for bc_type, params in self.config.boundary_conditions.items():
                self.boundary_conditions[bc_type] = self._create_boundary_condition(
                    bc_type, params
                )

        # Set up initial condition as a boundary condition if not already defined
        if "initial" not in self.boundary_conditions and hasattr(
            self.config, "initial_condition"
        ):
            self.boundary_conditions["initial"] = self._create_boundary_condition(
                "initial", self.config.initial_condition
            )

    def _setup_validation_points(self):
        """Set up validation points for exact solution comparison."""
        self.validation_points = None

    def _create_boundary_condition(
        self, bc_type: str, params: Dict[str, Any]
    ) -> callable:
        """
        Create boundary condition function from parameters.

        :param bc_type: Type of boundary condition
        :param params: Parameters for the boundary condition
        :return: Boundary condition function
        """
        # Map left/right to dirichlet for backward compatibility
        if bc_type in ["left", "right"]:
            bc_type = "dirichlet"

        if bc_type == "dirichlet":
            value = params.get("value", 0.0)
            return lambda x, t: torch.full_like(x[:, 0:1], value)

        elif bc_type == "neumann":
            value = params.get("value", 0.0)
            return lambda x, t: torch.full_like(x[:, 0:1], value)

        elif bc_type == "periodic":
            if self.dimension == 1:
                return lambda x, t: torch.sin(2 * torch.pi * x[:, 0:1])
            else:
                return lambda x, t: torch.sin(
                    2 * torch.pi * torch.sum(x, dim=1, keepdim=True)
                )

        elif bc_type == "initial":
            # Handle different types of initial conditions
            ic_type = params.get("type", "sine")

            if ic_type == "sine":
                amplitude = params.get("amplitude", 1.0)
                frequency = params.get("frequency", 1.0)
                return lambda x, t: amplitude * torch.sin(
                    frequency * torch.pi * x[:, 0:1]
                )

            elif ic_type == "tanh":
                epsilon = params.get("epsilon", 0.1)
                return lambda x, t: torch.tanh(x[:, 0:1] / epsilon)

            elif ic_type == "gaussian":
                mean = params.get("mean", 0.0)
                std = params.get("std", 0.1)
                return lambda x, t: torch.exp(-((x[:, 0:1] - mean) ** 2) / (2 * std**2))

            elif ic_type == "fixed":
                value = params.get("value", 0.0)
                return lambda x, t: torch.full_like(x[:, 0:1], value)

            elif ic_type == "random":
                amplitude = params.get("amplitude", 0.1)
                return lambda x, t: amplitude * (2 * torch.rand_like(x[:, 0:1]) - 1)

            else:
                # Default to zero if type not recognized
                print(
                    f"Warning: Unrecognized initial condition type '{ic_type}'. Defaulting to zero."
                )
                return lambda x, t: torch.zeros_like(x[:, 0:1])
        else:
            print(
                f"Warning: Unsupported boundary condition type '{bc_type}'. Defaulting to zero."
            )
            return lambda x, t: torch.zeros_like(x[:, 0:1])

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the PDE residual. Must be implemented by subclasses.

        :param model: Neural network model
        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Residual tensor
        """
        raise NotImplementedError("Subclasses must implement compute_residual")

    def compute_derivatives(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        temporal_derivatives: List[int] = None,
        spatial_derivatives: Set[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute derivatives of the solution with respect to time and space.

        Args:
            model: Neural network model
            x: Spatial coordinates
            t: Time coordinates
            temporal_derivatives: List of temporal derivative orders to compute
            spatial_derivatives: Set of spatial derivative orders to compute

        Returns:
            Dictionary containing computed derivatives

        Raises:
            ValueError: If derivative order is invalid (>2 for temporal, >4 for spatial)
        """
        # Validate derivative orders
        if temporal_derivatives:
            max_temporal_order = max(temporal_derivatives)
            if max_temporal_order > 2:
                raise ValueError(
                    f"Temporal derivative order {max_temporal_order} is not supported. Maximum order is 2."
                )

        if spatial_derivatives:
            max_spatial_order = max(spatial_derivatives)
            if max_spatial_order > 4:
                raise ValueError(
                    f"Spatial derivative order {max_spatial_order} is not supported. Maximum order is 4."
                )

        # Ensure tensors require gradients and are on the correct device
        x = x.detach().to(self.device).requires_grad_(True)
        t = t.detach().to(self.device).requires_grad_(True)

        # Ensure model parameters require gradients
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad_(True)
            # Set model to training mode
            model.train()
            # Forward pass through the model
            inputs = torch.cat([x, t], dim=1)
            u = model(inputs)
        else:
            # If model is a tensor (for testing), use it directly
            u = model

        # Ensure output requires gradients
        u = u.to(self.device).requires_grad_(True)

        derivatives = {}

        # Compute temporal derivatives if requested
        if temporal_derivatives:
            u_t_prev = u
            for i in sorted(temporal_derivatives):
                if i == 0:
                    continue  # Skip 0th order derivative

                if i == 1:
                    # First time derivative
                    grad_outputs = torch.ones_like(
                        u, device=self.device
                    ).requires_grad_(True)
                    u_t = torch.autograd.grad(
                        u,
                        t,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        allow_unused=True,
                        retain_graph=True,
                    )[0]
                    if u_t is None:
                        u_t = torch.zeros_like(u, device=self.device)
                    derivatives["dt"] = u_t.requires_grad_(True)
                    u_t_prev = u_t
                else:
                    # Higher-order time derivatives
                    key = f"dt{i}"
                    grad_outputs = torch.ones_like(
                        u_t_prev, device=self.device
                    ).requires_grad_(True)
                    u_t_higher = torch.autograd.grad(
                        u_t_prev,
                        t,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        allow_unused=True,
                        retain_graph=True,
                    )[0]
                    if u_t_higher is None:
                        u_t_higher = torch.zeros_like(u, device=self.device)
                    derivatives[key] = u_t_higher.requires_grad_(True)
                    u_t_prev = u_t_higher

        # Compute spatial derivatives if requested
        if spatial_derivatives:
            if self.dimension == 1:
                u_x_prev = u
                for i in sorted(spatial_derivatives):
                    if i == 0:
                        continue  # Skip 0th order derivative

                    if i == 1:
                        # First derivative
                        grad_outputs = torch.ones_like(
                            u, device=self.device
                        ).requires_grad_(True)
                        u_x = torch.autograd.grad(
                            u,
                            x,
                            grad_outputs=grad_outputs,
                            create_graph=True,
                            allow_unused=True,
                            retain_graph=True,
                        )[0]
                        if u_x is None:
                            u_x = torch.zeros_like(u, device=self.device)
                        derivatives["dx"] = u_x.requires_grad_(True)
                        u_x_prev = u_x
                    else:
                        # Higher-order derivatives
                        key = f"dx{i}"
                        grad_outputs = torch.ones_like(
                            u_x_prev, device=self.device
                        ).requires_grad_(True)
                        u_x_higher = torch.autograd.grad(
                            u_x_prev,
                            x,
                            grad_outputs=grad_outputs,
                            create_graph=True,
                            allow_unused=True,
                            retain_graph=True,
                        )[0]
                        if u_x_higher is None:
                            u_x_higher = torch.zeros_like(u, device=self.device)
                        derivatives[key] = u_x_higher.requires_grad_(True)
                        u_x_prev = u_x_higher
            else:
                # Multi-dimensional case
                for dim in range(self.dimension):
                    u_x_prev = u
                    dim_name = f"x{dim+1}" if self.dimension > 1 else "x"

                    for order in sorted(spatial_derivatives):
                        if order == 0:
                            continue  # Skip 0th order derivative

                        # Compute derivatives recursively for each dimension
                        for i in range(1, order + 1):
                            if i == 1:
                                # First derivative
                                grad_outputs = torch.ones_like(
                                    u, device=self.device
                                ).requires_grad_(True)
                                u_x = torch.autograd.grad(
                                    u,
                                    x[:, dim : dim + 1],
                                    grad_outputs=grad_outputs,
                                    create_graph=True,
                                    allow_unused=True,
                                    retain_graph=True,
                                )[0]
                                if u_x is None:
                                    u_x = torch.zeros_like(u, device=self.device)
                                derivatives[f"d{dim_name}"] = u_x.requires_grad_(True)
                                u_x_prev = u_x
                            else:
                                # Higher-order derivatives
                                key = f"d{dim_name*i}"
                                grad_outputs = torch.ones_like(
                                    u_x_prev, device=self.device
                                ).requires_grad_(True)
                                u_x_higher = torch.autograd.grad(
                                    u_x_prev,
                                    x[:, dim : dim + 1],
                                    grad_outputs=grad_outputs,
                                    create_graph=True,
                                    allow_unused=True,
                                    retain_graph=True,
                                )[0]
                                if u_x_higher is None:
                                    u_x_higher = torch.zeros_like(u, device=self.device)
                                derivatives[key] = u_x_higher.requires_grad_(True)
                                u_x_prev = u_x_higher

        # Compute Laplacian (∇²u) for convenience
        if spatial_derivatives and 2 in spatial_derivatives:
            if self.dimension == 1:
                # 1D Laplacian is just the second derivative
                derivatives["laplacian"] = derivatives["dx2"]
            else:
                # Multi-dimensional Laplacian is the sum of second derivatives in each dimension
                laplacian = torch.zeros_like(u, device=self.device)
                for dim in range(self.dimension):
                    dim_name = f"x{dim+1}" if self.dimension > 1 else "x"
                    laplacian += derivatives[f"d{dim_name*2}"]
                derivatives["laplacian"] = laplacian.requires_grad_(True)

        return derivatives

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute exact analytical solution. Must be implemented by subclasses.

        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Exact solution tensor
        """
        raise NotImplementedError("Subclasses must implement exact_solution")

    def generate_collocation_points(
        self, num_points: int, strategy: str = "uniform"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate collocation points for training.

        :param num_points: Number of points to generate
        :param strategy: Sampling strategy ('uniform', 'latin_hypercube', 'sobol', 'adaptive')
        :return: Tuple of spatial and temporal points
        """
        if strategy == "uniform":
            if self.dimension == 1:
                # For 1D, domain is a list with one tuple
                x = torch.linspace(
                    self.domain[0][0],
                    self.domain[0][1],
                    int(np.sqrt(num_points)),
                    device=self.device,
                ).reshape(-1, 1)
                t = torch.linspace(
                    self.time_domain[0],
                    self.time_domain[1],
                    int(np.sqrt(num_points)),
                    device=self.device,
                ).reshape(-1, 1)

                # Create meshgrid for even coverage
                X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
                x = X.reshape(-1, 1)
                t = T.reshape(-1, 1)

                # Add some noise for better training
                x_noise = (self.domain[0][1] - self.domain[0][0]) * 0.01
                t_noise = (self.time_domain[1] - self.time_domain[0]) * 0.01
                x = x + torch.randn_like(x) * x_noise
                t = t + torch.randn_like(t) * t_noise

                # Clip to domain
                x = torch.clamp(x, self.domain[0][0], self.domain[0][1])
                t = torch.clamp(t, self.time_domain[0], self.time_domain[1])

            else:
                # For multi-dimensional domains
                grid_points = []
                # Calculate points per dimension to ensure we get enough points
                # Use a number slightly higher to account for possible pruning
                points_per_dim = max(
                    2, int((num_points) ** (1 / (self.dimension + 1))) + 1
                )

                for dim in range(self.dimension):
                    grid_dim = torch.linspace(
                        self.domain[dim][0],
                        self.domain[dim][1],
                        points_per_dim,
                    )
                    grid_points.append(grid_dim)

                # Add time dimension
                grid_points.append(
                    torch.linspace(
                        self.time_domain[0],
                        self.time_domain[1],
                        points_per_dim,
                    )
                )

                # Create meshgrid
                grid_tensors = torch.meshgrid(*grid_points, indexing="ij")

                # Reshape to points
                points = torch.stack([g.reshape(-1) for g in grid_tensors], dim=1)

                # If we have more points than requested, sample exactly num_points
                if len(points) > num_points:
                    # Random indices without replacement
                    indices = torch.randperm(len(points))[:num_points]
                    points = points[indices]
                # If we have fewer points, add more by sampling with replacement
                elif len(points) < num_points:
                    additional_indices = torch.randint(
                        0, len(points), (num_points - len(points),)
                    )
                    additional_points = points[additional_indices]
                    points = torch.cat([points, additional_points], dim=0)

                # Add noise for better training
                noise_scale = 0.01
                noise = torch.randn_like(points) * noise_scale
                points = points + noise

                # Clip to domain
                for dim in range(self.dimension):
                    points[:, dim] = torch.clamp(
                        points[:, dim], self.domain[dim][0], self.domain[dim][1]
                    )
                points[:, -1] = torch.clamp(
                    points[:, -1],
                    self.time_domain[0],
                    self.time_domain[1],
                )

                # Extract x and t
                x = points[:, : self.dimension]
                t = points[:, -1].reshape(-1, 1)

        elif strategy == "latin_hypercube":
            sampler = qmc.LatinHypercube(d=self.dimension + 1)  # +1 for time dimension
            sample = sampler.random(n=num_points)

            # Scale spatial coordinates
            x = []
            for dim in range(self.dimension):
                x.append(
                    torch.tensor(
                        qmc.scale(
                            sample[:, dim].reshape(-1, 1),
                            l_bounds=[float(self.domain[dim][0])],
                            u_bounds=[float(self.domain[dim][1])],
                        ),
                        dtype=torch.float32,
                    )
                )
            x = torch.cat(x, dim=1)

            # Scale time coordinate
            t = torch.tensor(
                qmc.scale(
                    sample[:, -1].reshape(-1, 1),
                    l_bounds=[float(self.time_domain[0])],
                    u_bounds=[float(self.time_domain[1])],
                ),
                dtype=torch.float32,
            )

        elif strategy == "sobol":
            # Sobol sequence for low-discrepancy sampling
            sampler = qmc.Sobol(d=self.dimension + 1)  # +1 for time dimension
            sample = sampler.random(n=num_points)

            # Scale spatial coordinates
            x = []
            for dim in range(self.dimension):
                x.append(
                    torch.tensor(
                        qmc.scale(
                            sample[:, dim].reshape(-1, 1),
                            l_bounds=[float(self.domain[dim][0])],
                            u_bounds=[float(self.domain[dim][1])],
                        ),
                        dtype=torch.float32,
                    )
                )
            x = torch.cat(x, dim=1)

            # Scale time coordinate
            t = torch.tensor(
                qmc.scale(
                    sample[:, -1].reshape(-1, 1),
                    l_bounds=[float(self.time_domain[0])],
                    u_bounds=[float(self.time_domain[1])],
                ),
                dtype=torch.float32,
            )

        elif strategy == "adaptive":
            # Use RL agent for adaptive sampling if available
            if self.rl_agent is not None:
                # Use the adaptive sampling from RL agent
                grid_size = min(100, max(10, int(np.sqrt(num_points))))

                # Create a grid of points for the RL agent to sample from
                if self.dimension == 1:
                    x_grid = torch.linspace(
                        self.domain[0][0],
                        self.domain[0][1],
                        grid_size,
                        device=self.device,
                    )
                    t_grid = torch.linspace(
                        self.time_domain[0],
                        self.time_domain[1],
                        grid_size,
                        device=self.device,
                    )
                    X, T = torch.meshgrid(x_grid, t_grid, indexing="ij")
                    points = torch.stack([X.flatten(), T.flatten()], dim=1)
                else:
                    # Generate multi-dimensional grid
                    grid_points = []
                    for dim in range(self.dimension):
                        grid_points.append(
                            torch.linspace(
                                self.domain[dim][0],
                                self.domain[dim][1],
                                grid_size,
                                device=self.device,
                            )
                        )
                    grid_points.append(
                        torch.linspace(
                            self.time_domain[0],
                            self.time_domain[1],
                            grid_size,
                            device=self.device,
                        )
                    )

                    grid_tensors = torch.meshgrid(*grid_points, indexing="ij")
                    points = torch.stack([g.flatten() for g in grid_tensors], dim=1)

                # Get sampling probabilities from RL agent
                with torch.no_grad():
                    probs = self.rl_agent.select_action(points)
                    probs = torch.abs(probs)  # Ensure positive probabilities
                    probs = probs / torch.sum(probs)  # Normalize

                # Sample points based on probabilities
                selected_indices = torch.multinomial(
                    probs.flatten(), min(num_points, len(points)), replacement=True
                )
                selected_points = points[selected_indices]

                # If we have fewer points than requested (rare case), add more with replacement
                if len(selected_points) < num_points:
                    additional_indices = torch.randint(
                        0,
                        len(selected_points),
                        (num_points - len(selected_points),),
                        device=self.device,
                    )
                    additional_points = selected_points[additional_indices]
                    selected_points = torch.cat(
                        [selected_points, additional_points], dim=0
                    )

                # Add some small noise to avoid exact grid points
                noise_scale = min(
                    0.01,
                    min(
                        [
                            (self.domain[d][1] - self.domain[d][0]) / grid_size
                            for d in range(self.dimension)
                        ]
                    ),
                    (self.time_domain[1] - self.time_domain[0]) / grid_size,
                )
                noise = torch.randn_like(selected_points) * noise_scale
                selected_points = selected_points + noise

                # Clip to domain bounds
                for dim in range(self.dimension):
                    selected_points[:, dim] = torch.clamp(
                        selected_points[:, dim],
                        self.domain[dim][0],
                        self.domain[dim][1],
                    )
                selected_points[:, -1] = torch.clamp(
                    selected_points[:, -1],
                    self.time_domain[0],
                    self.time_domain[1],
                )

                # Split into spatial and temporal coordinates
                if self.dimension == 1:
                    x = selected_points[:, 0].reshape(-1, 1)
                else:
                    x = selected_points[:, : self.dimension]
                t = selected_points[:, -1].reshape(-1, 1)

                # Store points for visualization
                self.collocation_history.append(selected_points.cpu().numpy())

                # Reward the RL agent based on the diversity of selected points
                if len(self.collocation_history) > 1:
                    prev_points = torch.tensor(
                        self.collocation_history[-2], device=self.device
                    )
                    reward = torch.mean(
                        torch.min(torch.cdist(selected_points, prev_points), dim=1)[0]
                    )
                    self.rl_agent.update_epsilon(
                        len(self.collocation_history)
                    )  # Update exploration rate
            else:
                # Fallback to uniform sampling if no RL agent
                return self.generate_collocation_points(num_points, strategy="uniform")
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        # Ensure the points are on the correct device before returning
        x = x.to(self.device)
        t = t.to(self.device)
        
        return x, t

    def compute_loss(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components for training.

        :param model: Neural network model
        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Dictionary of loss components
        """
        # Compute PDE residual
        residual = self.compute_residual(model, x, t)
        residual_loss = torch.mean(residual**2)

        # Compute boundary condition loss
        if self.dimension == 1:
            x_boundary = torch.tensor(
                [self.domain[0][0], self.domain[0][1]],
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1, 1)
        else:
            x_boundary = []
            for dim in range(self.dimension):
                x_boundary.extend([self.domain[dim][0], self.domain[dim][1]])
            x_boundary = torch.tensor(
                x_boundary, dtype=torch.float32, device=self.device
            ).reshape(-1, 1)

        t_boundary = torch.linspace(
            self.time_domain[0],
            self.time_domain[1],
            100,
            device=self.device,
        ).reshape(-1, 1)

        # Create boundary points by combining spatial and temporal coordinates
        x_boundary = x_boundary.repeat_interleave(len(t_boundary), dim=0)
        t_boundary = t_boundary.repeat(len(x_boundary) // len(t_boundary), 1)

        # Compute boundary condition loss
        boundary_loss = torch.tensor(0.0, device=self.device)
        for bc_type, bc_func in self.boundary_conditions.items():
            u_boundary = model(torch.cat([x_boundary, t_boundary], dim=1))
            u_target = bc_func(x_boundary, t_boundary)
            boundary_loss += torch.mean((u_boundary - u_target) ** 2)

        # Compute initial condition loss
        x_initial = torch.linspace(
            self.domain[0][0], self.domain[0][1], 100, device=self.device
        ).reshape(-1, 1)
        t_initial = torch.zeros_like(x_initial, device=self.device)

        u_initial = model(torch.cat([x_initial, t_initial], dim=1))

        # Use existing boundary condition functions to handle initial conditions
        if "initial" in self.boundary_conditions:
            # If there's a specific initial condition function already created
            u_target = self.boundary_conditions["initial"](x_initial, t_initial)
        else:
            # Handle sine wave case as default (for backward compatibility)
            ic_type = self.config.initial_condition.get("type", "sine")
            if (
                ic_type == "sine"
                and "amplitude" in self.config.initial_condition
                and "frequency" in self.config.initial_condition
            ):
                u_target = self.config.initial_condition["amplitude"] * torch.sin(
                    self.config.initial_condition["frequency"] * torch.pi * x_initial
                )
            else:
                # For other types, create a temporary boundary condition function
                temp_bc = self._create_boundary_condition(
                    "initial", self.config.initial_condition
                )
                u_target = temp_bc(x_initial, t_initial)

        initial_loss = torch.mean((u_initial - u_target) ** 2)
        
        # Initialize smoothness loss
        smoothness_loss = torch.tensor(0.0, device=self.device)
        
        # Get smoothness weight from config if available
        if hasattr(self.config.training, "loss_weights") and self.config.training.loss_weights:
            smoothness_weight = self.config.training.loss_weights.get("smoothness", 0.0)
        else:
            smoothness_weight = 0.0
        
        # Create dictionary of losses
        losses = {
            "residual": residual_loss,
            "boundary": boundary_loss,
            "initial": initial_loss,
            "smoothness": smoothness_loss
        }

        # Use adaptive weights if enabled
        if (
            hasattr(self.config.training, "adaptive_weights")
            and self.config.training.adaptive_weights.enabled
        ):
            # The total loss will be computed by the trainer using adaptive weights
            # We just return the individual components
            losses["total"] = (
                residual_loss
                + boundary_loss
                + initial_loss
                + smoothness_weight * smoothness_loss
            )
        else:
            # Otherwise use fixed weights from config
            # Map 'pde' key to 'residual' for backward compatibility
            residual_weight = self.config.training.loss_weights.get("pde", self.config.training.loss_weights.get("residual", 1.0))
            boundary_weight = self.config.training.loss_weights.get("boundary", 10.0)
            initial_weight = self.config.training.loss_weights.get("initial", 10.0)
            
            total_loss = (
                residual_weight * residual_loss
                + boundary_weight * boundary_loss
                + initial_weight * initial_loss
                + smoothness_weight * smoothness_loss
            )
            losses["total"] = total_loss

        return losses

    def build_model(self, override_config=None):
        """
        Build a neural network model using the PDE-specific architecture settings
        from the configuration.

        :param override_config: Optional dictionary to override architecture parameters.
        :return: Instantiated neural network model.
        """
        try:
            # Import the PINNModel class.
            from src.neural_networks import PINNModel

            # Determine the current PDE type. Default to "wave" if not specified.
            pde_type = self.config.get("pde_type", "wave")
            # Get the PDE-specific configuration.
            pde_conf = self.config.get("pde_configs", {})
            pde_conf = pde_conf.get(pde_type, {})

            # Build the base architecture configuration using PDE-specific input/output dimensions
            # and the device from the class.
            arch_config = {
                "input_dim": pde_conf.get("input_dim", 2),
                "output_dim": pde_conf.get("output_dim", 1),
                "device": self.device,
            }

            # Get the architecture type specified for this PDE.
            arch_type = pde_conf.get("architecture", None)
            if arch_type is not None:
                # Look up the detailed architecture settings in the global "architectures" section.
                architectures_dict = self.config.get("architectures", {})
                if arch_type in architectures_dict:
                    # Merge in the PDE-specific architecture details.
                    arch_config.update(architectures_dict[arch_type])
                else:
                    # Fallback: if no detailed settings exist, simply store the architecture name.
                    arch_config["architecture"] = arch_type

            # Apply any user-specified override configuration.
            if override_config:
                arch_config.update(override_config)

            # Instantiate and return the model.
            return PINNModel(**arch_config)

        except ImportError:
            print(
                "Could not import PINNModel. Make sure neural_networks module is available."
            )
            return None

    def validate(
        self, model: torch.nn.Module, num_points: int = 1000
    ) -> Dict[str, float]:
        """
        Validate the model's solution against exact solution.

        :param model: Neural network model
        :param num_points: Number of validation points
        :return: Dictionary of error metrics
        """
        x, t = self.generate_collocation_points(num_points)
        u_pred = model(torch.cat([x, t], dim=1))
        u_exact = self.exact_solution(x, t)
        error = torch.abs(u_pred - u_exact)
        return {
            "l2_error": torch.mean(error**2).item(),
            "max_error": torch.max(error).item(),
            "mean_error": torch.mean(error).item(),
        }

    def plot_solution(
        self,
        model: torch.nn.Module,
        num_points: int = 1000,
        save_path: Optional[str] = None,
    ):
        """
        Plot the model's solution and exact solution.

        :param model: Neural network model
        :param num_points: Number of points for plotting
        :param save_path: Optional path to save the plot
        """
        x, t = self.generate_collocation_points(num_points)
        u_pred = model(torch.cat([x, t], dim=1))
        u_exact = self.exact_solution(x, t)

        plt.figure(figsize=(10, 6))
        plt.scatter(
            x.cpu().numpy(), u_pred.detach().cpu().numpy(), label="Predicted", alpha=0.5
        )
        plt.scatter(x.cpu().numpy(), u_exact.cpu().numpy(), label="Exact", alpha=0.5)
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(f"{self.config.name} Solution")
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def save_state(self, path: str):
        """
        Save the PDE state to a file.

        :param path: Path to save the state
        """
        state = {
            "config": self.config,
            "collocation_history": self.collocation_history,
            "validation_points": self.validation_points,
        }
        torch.save(state, path)

    def load_state(self, path: str):
        """
        Load the PDE state from a file.

        :param path: Path to load the state from
        """
        state = torch.load(path)
        self.config = state["config"]
        self.collocation_history = state["collocation_history"]
        self.validation_points = state["validation_points"]
        self._setup_boundary_conditions()

    def update_sampling_strategy(
        self, x: torch.Tensor, t: torch.Tensor, residual: torch.Tensor
    ):
        """
        Update the RL agent's sampling strategy based on the residual.

        :param x: Spatial coordinates
        :param t: Time coordinates
        :param residual: PDE residual at these points
        """
        # Compute reward based on residual magnitude
        reward = torch.mean(torch.abs(residual))

        # Update RL agent
        state = torch.cat([x, t], dim=1)
        self.rl_agent.update(state, reward)

    def visualize_sampling_strategy(self, num_points: int = 1000):
        """
        Visualize the current sampling strategy.

        :param num_points: Number of points to generate for visualization
        :return: Tuple of (x, t) coordinates and their acceptance probabilities
        """
        x = torch.rand((num_points, self.dimension), device=self.device)
        t = torch.rand((num_points, 1), device=self.device)

        # Scale to domain
        x = self._scale_to_domain(x, self.domain)
        t = self._scale_to_domain(t, self.time_domain)

        # Get RL agent's actions for all points
        states = torch.cat([x, t], dim=1)
        actions = torch.zeros(num_points, device=self.device)
        for i in range(num_points):
            actions[i] = self.rl_agent.select_action(states[i : i + 1])

        return x, t, actions

    def visualize_collocation_evolution(
        self,
        points_history=None,
        epoch=None,
        save_path: Optional[str] = None,
        num_snapshots: int = 5,
    ):
        """
        Visualize the evolution of collocation points during training.

        :param points_history: History of collocation points
        :param epoch: Current epoch number
        :param save_path: Path to save the visualization
        :param num_snapshots: Number of snapshots to visualize
        """
        # Use provided points_history or class attribute
        history = (
            points_history if points_history is not None else self.collocation_history
        )

        if not history or len(history) < 2:
            print("Not enough collocation history to visualize evolution")
            return

        import matplotlib

        # Use 'Agg' backend if running in a non-main thread
        if not matplotlib.is_interactive():
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import os

        # Create output directory if it doesn't exist
        os.makedirs("visualizations", exist_ok=True)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        plt.suptitle(
            (
                f"Evolution of Collocation Points (Epoch {epoch})"
                if epoch
                else "Evolution of Collocation Points"
            ),
            fontsize=16,
        )

        # Define custom colormap for evolution
        colors = ["#0d3b66", "#1b9aaa", "#ef476f", "#ffc43d"]
        cmap = LinearSegmentedColormap.from_list(
            "evolution_cmap", colors, N=len(history)
        )

        # Select snapshots to visualize
        indices = np.linspace(0, len(history) - 1, num_snapshots).astype(int)

        # First plot: Progression of points over time
        ax = axes[0, 0]
        for i, idx in enumerate(indices):
            points = history[idx]
            if self.dimension == 1:
                x_pts = points[:, 0]
                y_pts = points[:, 1]  # time dimension
            else:
                # For 2D problems, use the first two spatial dimensions
                x_pts = points[:, 0]
                y_pts = points[:, 1]

            alpha = 0.3 + 0.7 * (i / (len(indices) - 1))  # Increasing alpha
            size = 5 + 20 * (i / (len(indices) - 1))  # Increasing size
            ax.scatter(
                x_pts,
                y_pts,
                alpha=alpha,
                s=size,
                color=cmap(i / (len(indices) - 1)),
                label=f"Snapshot {idx}",
            )

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("t", fontsize=12)
        ax.set_title("Progression of Points", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)

        # Second plot: Density evolution (first snapshot)
        points = history[0]
        self._plot_density_snapshot(
            axes[0, 1], points, "Initial Distribution", cmap="Blues"
        )

        # Third plot: Density evolution (middle snapshot)
        mid_idx = len(history) // 2
        points = history[mid_idx]
        self._plot_density_snapshot(
            axes[1, 0],
            points,
            f"Intermediate Distribution (Snapshot {mid_idx})",
            cmap="Greens",
        )
        # Fourth plot: Density evolution (last snapshot)
        points = history[-1]
        self._plot_density_snapshot(
            axes[1, 1], points, "Final Distribution", cmap="Reds"
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            save_name = (
                f"collocation_evolution_epoch_{epoch}.png"
                if epoch
                else "collocation_evolution.png"
            )
            plt.savefig(f"visualizations/{save_name}", dpi=300)
            # Also save as latest for real-time viewing
            plt.savefig("visualizations/latest_collocation_evolution.png", dpi=300)
        plt.close()

    def _plot_density_snapshot(self, ax, points, title, cmap="viridis"):
        """Helper method to plot density snapshot with clearer visualization."""
        if self.dimension == 1:
            x_pts = points[:, 0]
            y_pts = points[:, 1]  # time dimension
        else:
            # For 2D problems, use the first two spatial dimensions
            x_pts = points[:, 0]
            y_pts = points[:, 1]

        # Define grid for density estimation
        xmin, xmax = np.min(x_pts), np.max(x_pts)
        ymin, ymax = np.min(y_pts), np.max(y_pts)

        # Add small padding
        x_padding = (xmax - xmin) * 0.05
        y_padding = (ymax - ymin) * 0.05 if ymax > ymin else 0.05

        xmin -= x_padding
        xmax += x_padding
        ymin -= y_padding
        ymax += y_padding

        # Create a simple 2D histogram
        hist, xedges, yedges = np.histogram2d(
            x_pts,
            y_pts,
            bins=(20, 20),  # Reduced number of bins for clearer visualization
            range=[[xmin, xmax], [ymin, ymax]],
        )

        # Plot heatmap
        im = ax.imshow(
            hist.T,
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",  # Sharp transitions between density levels
        )

        # Add colorbar with simplified labels
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Number of Points", fontsize=10)

        # Overlay scatter plot of actual points with increased visibility
        ax.scatter(x_pts, y_pts, s=10, c="white", alpha=0.5, marker=".")

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("t", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, linestyle="--")
