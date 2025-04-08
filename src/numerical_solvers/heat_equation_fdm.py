import numpy as np
from typing import Dict
from src.numerical_solvers.finite_difference_base import FiniteDifferenceSolver, FDMConfig
import os
import logging

class HeatEquationFDM(FiniteDifferenceSolver):
    """Finite difference solver for the heat equation."""
    
    def check_stability(self) -> None:
        """Check the stability condition for FTCS scheme."""
        alpha = self.config.parameters.get('alpha', 0.1)
        stability = alpha * self.dt / (self.dx * self.dx)
        if stability > 0.5:
            print(f"Warning: Scheme might be unstable. Stability parameter = {stability} > 0.5")

    def discretize_equation(self) -> None:
        """Implement FTCS scheme for the heat equation."""
        alpha = self.config.parameters.get('alpha', 0.1)
        r = alpha * self.dt / (self.dx * self.dx)
        
        for n in range(0, self.config.nt - 1):
            # Compute approximations
            terms = self.approximate_differential_terms(self.u[n], n)
            
            # Update solution
            self.u[n + 1, 1:-1] = self.u[n, 1:-1] + r * terms['d2u_dx2']
            
            # Apply boundary conditions
            self.apply_boundary_conditions(n + 1)

    def approximate_differential_terms(self, u: np.ndarray, n: int) -> Dict[str, np.ndarray]:
        """Approximate spatial derivatives for the heat equation."""
        # Second derivative in space (central difference)
        d2u_dx2 = (u[2:] - 2 * u[1:-1] + u[:-2])
        
        return {'d2u_dx2': d2u_dx2}

    def apply_boundary_conditions(self, n: int) -> None:
        """Apply boundary conditions for the heat equation."""
        bc_type = next(iter(self.config.boundary_conditions))  # Get first BC type
        
        if bc_type == 'periodic':
            # Periodic boundary conditions
            self.u[n, 0] = self.u[n, -2]  # Left boundary
            self.u[n, -1] = self.u[n, 1]  # Right boundary
        elif bc_type == 'dirichlet':
            # Dirichlet boundary conditions
            value = self.config.boundary_conditions[bc_type].get('value', 0.0)
            self.u[n, 0] = value  # Left boundary
            self.u[n, -1] = value  # Right boundary
        else:
            raise ValueError(f"Unsupported boundary condition type: {bc_type}")

    def plot_solution(self, save_path=None, num_time_steps=5):
        """
        Visualize and save the numerical solution of the heat equation.
        
        Args:
            save_path (str): Path where to save the image. If None, no save is performed.
            num_time_steps (int): Number of time steps to visualize.
        """
        try:
            import matplotlib.pyplot as plt
            import os
            
            # Create figure to display multiple time steps
            plt.figure(figsize=(12, 8))
            
            # Select uniformly distributed time steps
            time_steps = np.linspace(0, self.config.nt-1, num_time_steps, dtype=int)
            
            # Create x matrix for horizontal axis
            x = np.linspace(self.config.x_domain[0], self.config.x_domain[1], self.config.nx)
            
            # Plot each time step
            for i, t_idx in enumerate(time_steps):
                plt.subplot(num_time_steps, 1, i+1)
                plt.plot(x, self.u[t_idx], 'b-', linewidth=2)
                plt.title(f'FDM Solution at t = {t_idx * self.dt:.3f}')
                plt.ylabel('u(x,t)')
                if i == num_time_steps - 1:
                    plt.xlabel('x')
                plt.grid(True)
                plt.ylim(-1.2, 1.2)  # Adjust according to expected values
            
            plt.tight_layout()
            
            # Save the figure if a path is provided
            if save_path:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Save the figure
                plt.savefig(save_path, format='png', dpi=300)
                plt.close()
                print(f"FDM solution saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError as e:
            print(f"Error importing necessary libraries for plotting: {e}")
        except Exception as e:
            print(f"Error plotting FDM solution: {e}")

    def plot_solution_comparison(self, pinn_model=None, save_path=None, device='cpu'):
        """
        Visualize and save a comparison of the FDM numerical solution with the PINN solution.
        
        Args:
            pinn_model (torch.nn.Module): Trained PINN model to compare.
            save_path (str): Path where to save the image. If None, no save is performed.
            device (str): Device for PyTorch calculations.
        """
        try:
            import matplotlib.pyplot as plt
            import torch
            import os
            
            # Create figure for comparison
            plt.figure(figsize=(15, 10))
            
            # Select time steps for comparison
            num_time_steps = 4
            time_steps = np.linspace(0, self.config.nt-1, num_time_steps, dtype=int)
            
            # Create x matrix for horizontal axis
            x = np.linspace(self.config.x_domain[0], self.config.x_domain[1], self.config.nx)
            
            # Plot each time step with both solutions
            for i, t_idx in enumerate(time_steps):
                current_t = t_idx * self.dt
                plt.subplot(2, 2, i+1)
                
                # Plot FDM solution
                plt.plot(x, self.u[t_idx], 'b-', linewidth=2, label='FDM')
                
                # Plot PINN solution if available
                if pinn_model is not None:
                    # Prepare data for PINN model
                    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)
                    t_tensor = torch.full_like(x_tensor, current_t).to(device)
                    
                    # Predict with PINN model
                    with torch.no_grad():
                        inputs = torch.cat([x_tensor, t_tensor], dim=1)
                        pinn_pred = pinn_model(inputs).cpu().numpy()
                    
                    # Plot PINN solution
                    plt.plot(x, pinn_pred, 'r--', linewidth=2, label='PINN')
                
                plt.title(f'Comparison at t = {current_t:.3f}')
                plt.ylabel('u(x,t)')
                plt.xlabel('x')
                plt.grid(True)
                plt.legend()
                plt.ylim(-1.2, 1.2)  # Adjust according to expected values
            
            plt.tight_layout()
            
            # Save the figure if a path is provided
            if save_path:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Save the figure
                plt.savefig(save_path, format='png', dpi=300)
                plt.close()
                print(f"Solutions comparison saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError as e:
            print(f"Error importing necessary libraries for plotting: {e}")
        except Exception as e:
            print(f"Error plotting solutions comparison: {e}")
            
    def plot_solution_3d(self, save_path=None):
        """
        Create a 3D surface plot of the finite difference solution.
        
        Args:
            save_path (str): Path where to save the image. If None, no save is performed.
        """
        try:
            import plotly.graph_objects as go
            import os
            
            # Create meshgrid for 3D plotting
            x = np.linspace(self.config.x_domain[0], self.config.x_domain[1], self.config.nx)
            t = np.linspace(self.config.t_domain[0], self.config.t_domain[1], self.config.nt)
            T, X = np.meshgrid(t, x, indexing='ij')
            
            # Get FDM solution
            Z = self.u
            
            # Create 3D surface plot
            fig = go.Figure(data=[go.Surface(
                z=Z,
                x=X, 
                y=T, 
                colorscale='Viridis',
                colorbar=dict(
                    title=dict(
                        text="u(x,t)",
                        font=dict(size=14)
                    ),
                    lenmode="fraction", 
                    len=0.6
                ),
                lighting=dict(
                    ambient=0.8,
                    diffuse=0.9,
                    roughness=0.5,
                    specular=0.2
                )
            )])
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text='Finite Difference Solution (1D)',
                    font=dict(size=16)
                ),
                scene=dict(
                    xaxis_title='x',
                    yaxis_title='t',
                    zaxis_title='u(x,t)',
                    xaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)'),
                    yaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)'),
                    zaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)'),
                    aspectmode='manual',
                    aspectratio=dict(x=1.5, y=1.5, z=0.8),
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=-0.1),
                        eye=dict(x=1.5, y=-1.5, z=0.8)
                    )
                ),
                width=800,
                height=600,
                paper_bgcolor='rgb(240, 245, 250)',
                margin=dict(l=65, r=50, b=65, t=90)
            )
            
            # Save the figure if a path is provided
            if save_path:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Save to HTML for interactive viewing
                html_path = save_path.replace('.png', '.html')
                fig.write_html(html_path)
                print(f"Interactive 3D visualization saved to: {html_path}")
                
                # Save to PNG for static viewing
                fig.write_image(save_path, width=800, height=600, scale=2)
                print(f"3D solution plot saved to: {save_path}")
            else:
                # Just display the figure
                fig.show()
                
        except ImportError:
            print("Plotly is required for 3D visualization. Install with: pip install plotly")
        except Exception as e:
            print(f"Error creating 3D visualization: {e}")

    @staticmethod
    def generate_fdm_comparison_plots(pde, model, device, viz_dir, logger=None):
        """
        Generate FDM comparison plots for the heat equation.
        
        Args:
            pde: PDE object with configuration
            model: Trained PINN model
            device: Computing device (cpu, cuda, mps)
            viz_dir: Directory to save visualizations
            logger: Logger object (optional)
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        
        import numpy as np
        
        # Get PDE parameters
        if hasattr(pde, 'domain'):
            spatial_domain = pde.domain
        elif hasattr(pde.config, 'domain'):
            spatial_domain = pde.config.domain
        else:
            spatial_domain = [(0.0, 1.0)]
            
        # Unified time domain retrieval - check all possible attributes
        time_domain = None
        if hasattr(pde, 'time_domain') and pde.time_domain is not None:
            time_domain = pde.time_domain
            logger.info(f"Using time_domain from pde.time_domain: {time_domain}")
        elif hasattr(pde, 't_domain') and pde.t_domain is not None:
            time_domain = pde.t_domain
            logger.info(f"Using time_domain from pde.t_domain: {time_domain}")
        elif hasattr(pde.config, 'time_domain') and pde.config.time_domain is not None:
            time_domain = pde.config.time_domain
            logger.info(f"Using time_domain from pde.config.time_domain: {time_domain}")
        elif hasattr(pde.config, 't_domain') and pde.config.t_domain is not None:
            time_domain = pde.config.t_domain
            logger.info(f"Using time_domain from pde.config.t_domain: {time_domain}")
        else:
            time_domain = [0.0, 1.0]
            logger.warning(f"No time_domain or t_domain found, using default: {time_domain}")
            
        # Ensure time_domain is a list of two floats
        if isinstance(time_domain, tuple):
            time_domain = list(time_domain)
        
        logger.info(f"Domain: {spatial_domain}, Time domain: {time_domain}")
            
        # Get diffusion coefficient
        alpha = 0.1  # Default
        if hasattr(pde, 'parameters') and pde.parameters and 'alpha' in pde.parameters:
            alpha = pde.parameters['alpha']
        elif hasattr(pde, 'alpha'):
            alpha = pde.alpha
        elif hasattr(pde.config, 'parameters') and pde.config.parameters and 'alpha' in pde.config.parameters:
            alpha = pde.config.parameters['alpha']
            
        logger.info(f"Using diffusion coefficient alpha: {alpha}")
            
        # Get boundary conditions
        bc_type = "dirichlet"
        bc_value = 0.0
        if hasattr(pde.config, 'boundary_conditions'):
            if 'dirichlet' in pde.config.boundary_conditions:
                bc_value = pde.config.boundary_conditions['dirichlet'].get('value', 0.0)
            elif 'periodic' in pde.config.boundary_conditions:
                bc_type = "periodic"
        
        # Configure FDM parameters
        nx = 100  # Number of spatial points
        dx = (spatial_domain[0][1] - spatial_domain[0][0]) / nx
        
        # Ensure stability: dt <= dx²/(2*alpha)
        stability_factor = 0.4  # Keep below 0.5
        dt = stability_factor * dx * dx / alpha
        
        # Calculate number of time steps
        t_final = time_domain[1]
        nt = int(t_final / dt) + 1
        
        logger.info(f"FDM Parameters - nx: {nx}, nt: {nt}, dx: {dx:.5f}, dt: {dt:.5f}")
        logger.info(f"Stability parameter: {alpha*dt/(dx*dx):.5f} (should be < 0.5)")
        
        # Create FDM configuration
        fdm_config = FDMConfig(
            nx=nx,
            nt=nt,
            x_domain=(spatial_domain[0][0], spatial_domain[0][1]),
            t_domain=(time_domain[0], time_domain[1]),
            parameters={"alpha": alpha},
            boundary_conditions={bc_type: {"value": bc_value} if bc_type == "dirichlet" else {}},
            initial_condition=None,
            device=device
        )
        
        logger.info(f"Created FDM config with t_domain: {fdm_config.t_domain}")
        
        # Create FDM solver
        fdm_solver = HeatEquationFDM(fdm_config)
        
        # Set initial condition based on PDE configuration
        if hasattr(pde, 'initial_condition_fn'):
            # Use PDE's initial condition function
            x = np.linspace(spatial_domain[0][0], spatial_domain[0][1], nx)
            try:
                ic_values = pde.initial_condition_fn(x).detach().cpu().numpy()
                fdm_solver.set_initial_condition(lambda x: ic_values)
                logger.info("Using PDE's initial condition function")
            except Exception as e:
                logger.warning(f"Error using PDE's initial condition function: {e}")
                # Fall back to default
                ic_amplitude = 1.0
                ic_frequency = 1.0
                fdm_solver.set_initial_condition(lambda x: ic_amplitude * np.sin(ic_frequency * np.pi * x))
                logger.info("Falling back to default initial condition")
        else:
            # Default to sine function
            ic_amplitude = 1.0
            ic_frequency = 1.0
            
            # Try to get initial condition parameters from config
            if hasattr(pde.config, 'initial_condition'):
                if isinstance(pde.config.initial_condition, dict):
                    ic_amplitude = pde.config.initial_condition.get('amplitude', 1.0)
                    ic_frequency = pde.config.initial_condition.get('frequency', 1.0)
            
            # Define initial condition function
            def initial_condition(x):
                return ic_amplitude * np.sin(ic_frequency * np.pi * x)
                
            fdm_solver.set_initial_condition(initial_condition)
            logger.info(f"Using sine initial condition with amplitude {ic_amplitude} and frequency {ic_frequency}")
        
        # Solve the equation
        logger.info("Solving heat equation with FDM...")
        fdm_solver.solve()
        
        # Create and save visualizations
        logger.info("Generating FDM visualizations...")
        
        # 1. Plot solution at multiple time steps (stacked view)
        fdm_plot_path = os.path.join(viz_dir, "fdm_solution.png")
        fdm_solver.plot_solution(save_path=fdm_plot_path, num_time_steps=5)
        
        # 2. Plot solution at multiple time steps (multiline view)
        try:
            fdm_multiline_path = os.path.join(viz_dir, "fdm_solution_multiline.png")
            fdm_solver.plot_solution_multiline(save_path=fdm_multiline_path)
        except Exception as e:
            logger.warning(f"Failed to generate multiline plot: {e}")
        
        # 3. Plot comparison with analytical solution if available
        if hasattr(pde, 'exact_solution'):
            # Create wrapper for exact solution
            def analytical_solution(x, t):
                try:
                    import torch
                    x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(device)
                    t_tensor = torch.tensor(t, dtype=torch.float32).to(device)
                    result = pde.exact_solution(x_tensor, t=t_tensor).cpu().numpy().flatten()
                    return result
                except Exception as e:
                    logger.warning(f"Error computing analytical solution: {e}")
                    return np.zeros_like(x)
                
            fdm_analytical_path = os.path.join(viz_dir, "fdm_vs_analytical.png")
            try:
                fdm_solver.plot_analytical_comparison(
                    analytical_solution, 
                    save_path=fdm_analytical_path
                )
                logger.info("Generated analytical comparison plot")
            except Exception as e:
                logger.warning(f"Failed to generate analytical comparison: {e}")
        
        # 4. Generate 3D visualization
        fdm_3d_path = os.path.join(viz_dir, "fdm_solution_3d.png")
        try:
            fdm_solver.plot_solution_3d(save_path=fdm_3d_path)
            logger.info("Generated 3D visualization")
        except Exception as e:
            logger.warning(f"Failed to generate 3D visualization: {e}")
        
        # 5. Plot comparison with PINN solution
        fdm_pinn_path = os.path.join(viz_dir, "fdm_vs_pinn.png")
        try:
            # Usar el método de la clase base FiniteDifferenceSolver
            fig = fdm_solver.plot_comparison_with_pinn(
                pinn_model=model,
                save_path=fdm_pinn_path,
                device=device,
                title="Heat Equation: FDM vs PINN Comparison",
                figsize=(15, 12)
            )
            
            # También generar una comparación simple con plot_solution_comparison
            simple_comparison_path = os.path.join(viz_dir, "fdm_pinn_simple_comparison.png")
            fdm_solver.plot_solution_comparison(
                pinn_model=model,
                save_path=simple_comparison_path,
                device=device
            )
            
            logger.info("Generated PINN comparison plots")
        except Exception as e:
            logger.warning(f"Failed to generate PINN comparison: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("FDM comparison plots generated successfully")
        
        return fdm_solver  # Return the solver in case further analysis is needed
