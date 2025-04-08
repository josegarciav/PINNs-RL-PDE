"""
Numerical solvers for differential equations.
"""

from .finite_difference_base import FiniteDifferenceSolver, FDMConfig
from .heat_equation_fdm import HeatEquationFDM

__all__ = ["FiniteDifferenceSolver", "FDMConfig", "HeatEquationFDM"]
