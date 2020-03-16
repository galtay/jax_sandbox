import jax.numpy as np
from jax import grad, jit, vmap
from jax import random


key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)


size = 3000
x = random.normal(key, (size, size), dtype=np.float32)
np.dot(x, x.T).block_until_ready()  # runs on the GP
