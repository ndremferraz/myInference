import jax.numpy as jnp
import numpy as np

class Module:
    def __init__(self):
        pass

    def __call__(self, x: jnp.ndarray):
        raise NotImplementedError

class Linear(Module):
    def __init__(self, weights: np.ndarray):
        super().__init__()
        self.weights = weights

    def __call__(self, x: jnp.ndarray):
        return jnp.dot(x, self.weights)

