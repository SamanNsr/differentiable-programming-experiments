# A simple MLP trained with non-differentiable approach (genetic algorithm) zero-order optimization

import jax
import jax.numpy as jnp
from jax import jit as jjit
from jax import random as jrandom
from jax.flatten_util import ravel_pytree
from functools import partial


def init_network(rand_key, layer_sizes):
    keys = jrandom.split(rand_key, len(layer_sizes) - 1)
    params = []
    for k, (n_in, n_out) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        W = jrandom.uniform(k, (n_in, n_out), minval=-0.1, maxval=0.1)
        b = jnp.zeros((n_out,))
        params.append((W, b))
    return params


def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0, x)


@jjit
def predict(params, inputs):
    for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.tanh(outputs)
    return outputs


# fitness function for genetic algorithm
@partial(jjit, static_argnames=['unflatten_fn'])
def loss(flat_params, unflatten_fn, inputs, targets):
    params = unflatten_fn(flat_params)
    preds = predict(params, inputs)
    return jnp.mean((preds - targets) ** 2)


batch_loss = jax.vmap(loss, in_axes=(0, None, None, None))


@partial(jjit, static_argnames=['unflatten_fn', 'top_k'])
def genetic_update(population, unflatten_fn, inputs, targets, key, mutation_rate=0.05, top_k=10):
    # Evaluate population fitness
    losses = batch_loss(population, unflatten_fn, inputs, targets)

    # Selection (Elitism)
    best_indices = jnp.argsort(losses)[:top_k]
    parents = population[best_indices]

    # Reproduction
    pop_size = population.shape[0]
    repeats = pop_size // top_k
    next_generation = jnp.tile(parents, (repeats, 1))

    # Mutation
    noise = jrandom.normal(key, next_generation.shape) * mutation_rate

    # Strict Elitism
    mask = (jnp.arange(pop_size) >= top_k).astype(jnp.float32)
    noise = noise * mask[:, None]

    updated_population = next_generation + noise

    return updated_population, losses[best_indices[0]]


if __name__ == "__main__":
    main_key = jrandom.PRNGKey(42)
    main_key, pop_key = jrandom.split(main_key)

    layer_sizes = [2, 10, 10, 1]
    initial_params = init_network(main_key, layer_sizes)

    flat_blueprint, unflatten_fn = ravel_pytree(initial_params)
    NUM_PARAMS = len(flat_blueprint)

    pop_size = 100
    population = jrandom.normal(pop_key, (pop_size, NUM_PARAMS)) * 0.1

    X = jnp.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    Y = jnp.array([[0.], [1.], [1.], [0.]])

    generations = 50
    for gen in range(generations):
        main_key, step_key = jrandom.split(main_key)
        population, best_loss = genetic_update(
            population, unflatten_fn, X, Y, step_key, mutation_rate=0.05, top_k=10)

        if gen % 10 == 0:
            print(f"Generation {gen:03d} | Best Loss: {best_loss:.4f}")
