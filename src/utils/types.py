from typing import Union, List, Tuple
import numpy as np
import torch

# Type aliases for better readability
Array = Union[np.ndarray, torch.Tensor]
ArrayLike = Union[Array, List[float], List[int], Tuple[float, ...], Tuple[int, ...]]
