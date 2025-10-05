import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import jit as jjit
from jax import lax
import time

def init_mlp_params(rand_key, layer_sizes):
    keys = jrandom.split(rand_key, len(layer_sizes) - 1)
    params = []
    for k, (n_in, n_out) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        W = jrandom.uniform(k, (n_in, n_out), minval=-0.1, maxval=0.1)
        b = jnp.zeros((n_out,))
        params.append((W, b))
    return params

@jjit
def predict(params, inputs):
    for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.tanh(outputs)  
    return outputs               

@jjit
def loss(params, inputs, targets):
    logits = predict(params, inputs)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(targets * log_probs, axis=-1))

@jjit
def update(params, x, y, lr):
    loss_val, grads = jax.value_and_grad(loss)(params, x, y)
    new_params = [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]
    return new_params, loss_val

def train_body(epoch, carry):
    params, x, y, lr = carry
    params, _ = update(params, x, y, lr)
    return (params, x, y, lr)


if __name__ == "__main__":
    SEED = 0
    rand_key = jrandom.PRNGKey(SEED)
    in_dim = 784
    n_classes = 10
    layer_sizes = [in_dim, 512, 256, n_classes]
    params = init_mlp_params(rand_key, layer_sizes)
    

    key, k1, k2 = jrandom.split(rand_key, 3)
    x_batch = jrandom.normal(k1, (128, in_dim))
    y_int = jrandom.randint(k2, (128,), 0, n_classes)
    y_batch = jax.nn.one_hot(y_int, n_classes)


    lr = 1e-3
    num_epochs = 1000

    carry = (params, x_batch, y_batch, lr)

    # 2.1381 seconds python for loop
    # 0.1138 seconds jax fori loop
    start_time = time.time()
  
    carry = (params, x_batch, y_batch, lr)
    carry = lax.fori_loop(0, num_epochs, train_body, carry)
    params, _, _, _ = carry

    end_time = time.time()

    final_loss = loss(params, x_batch, y_batch)
    print(f"final loss: {final_loss:.4f}")
    print(f"Training time for {num_epochs} epochs: {end_time - start_time:.4f} seconds")