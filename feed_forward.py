"""
A feed forward neural network
"""
import itertools
import time

from jax.api import jit, grad
from jax.scipy.special import logsumexp
import jax.numpy as jnp

import numpy as onp
import tensorflow as tf


random_seed = 12
layer_sizes = [784, 1024, 1024, 10]
param_scale = 0.1
step_size = 0.001
num_epochs = 40
batch_size = 128

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
num_train = x_train.shape[0]
num_test = x_test.shape[0]
num_pixels = x_train.shape[1] * x_train.shape[2]

# flatten pixels
x_train = x_train.reshape(num_train, num_pixels)
x_test = x_test.reshape(num_test, num_pixels)

# one-hot encode targets
y_train = onp.eye(10)[y_train]
y_test = onp.eye(10)[y_test]


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def init_weights_and_biases(scale, layer_sizes, random_seed):
    rng = onp.random.RandomState(random_seed)
    return [
        (scale * rng.randn(m, n), scale * rng.randn(n))
        for m, n, in pairwise(layer_sizes)]


def get_number_of_batches(num_samples, batch_size):
    num_complete_batches, leftover_samples = divmod(num_samples, batch_size)
    assert(num_complete_batches * batch_size + leftover_samples == num_samples)
    if leftover_samples > 0:
        num_incomplete_batches = 1
    else:
        num_incomplete_batches = 0
    num_batches = num_complete_batches + num_incomplete_batches
    return num_batches


def infinite_data_stream(x_train, y_train, num_batches, random_seed):
    num_train = x_train.shape[0]
    rng = onp.random.RandomState(random_seed)
    while True:
        permuted_idxs = rng.permutation(num_train)
        for ii in range(num_batches):
            batch_idxs = permuted_idxs[ii * batch_size: (ii + 1) * batch_size]
            yield x_train[batch_idxs], y_train[batch_idxs]


def predict(params, inputs):
    activations = inputs
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jnp.tanh(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits - logsumexp(logits, axis=1, keepdims=True)


def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))

def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)

@jit
def update(params, batch):
    grads = grad(loss)(params, batch)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]


num_batches = get_number_of_batches(num_train, batch_size)
batches = infinite_data_stream(x_train, y_train, num_batches, random_seed)
params = init_weights_and_biases(param_scale, layer_sizes, random_seed)

for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
        batch = next(batches)
        params = update(params, batch)

    epoch_time = time.time() - start_time
    train_acc = accuracy(params, (x_train, y_train))
    test_acc = accuracy(params, (x_test, y_test))
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
