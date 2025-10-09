import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import jit as jjit
from jax import lax as jlax
from jax import value_and_grad
import tensorflow_datasets as tfds

### the training loss stuck at 2.3, why?
### because the weights are not initialized properly

def make_datasets(batch_size):
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()

    train_ds = ds_builder.as_dataset(split='train', as_supervised=True)
    test_ds = ds_builder.as_dataset(split='test', as_supervised=True)

    train_ds = (
        train_ds
        .shuffle(10_000)
        .batch(batch_size)
        .prefetch(2)
    )

    test_ds = (
        test_ds
        .batch(batch_size)
        .prefetch(2)
    )

    return train_ds, test_ds


# for the conv layer
def conv2d(x, w, b, stride=(1, 1), padding='SAME'):
    # channels are the third layers of our kernel
    # w shape: (out_ch, in_ch, kh, kw)

    # reordering to HWIO format for jax -> (kernel_height, kernel_width, in_channels, out_channels)
    w_nhwc = jnp.transpose(w, (2, 3, 1, 0))

    y = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=w_nhwc,
        window_strides=stride,
        padding=padding,
        # NHWC --> (batch, height, width, channels)
        # input tensor / kernel tensor /  output tensor
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    return y + b


def relu(x): return jnp.maximum(0, x)


def max_pool(x, pool=(2, 2), stride=(2, 2)):
    # NHWC
    # 1 for the batch axis N → we do not pool across batch.
    # pool[0] for height H → e.g. 2 for a 2×2 pool.
    # pool[1] for width W → e.g. 2 for a 2×2 pool.
    # 1 for channels C → we do not pool across channels (pool is spatial only).

    # 1 for batch (no movement).
    # stride[0] down the height.
    # stride[1] across the width.
    # 1 for channels.

    return jlax.reduce_window(x,
                              -jnp.inf,
                              jax.lax.max,
                              window_dimensions=(1, pool[0], pool[1], 1),
                              window_strides=(1, stride[0], stride[1], 1),
                              padding=((0, 0), (0, 0), (0, 0), (0, 0))
                              )

# convert Conv layer to Dense layer
def flatten(x):
    return x.reshape((x.shape[0], -1))

# for the dense layer
def linear(x, w, b):
    return x @ w.T + b

def forward(params, x):
    w_conv1 = params['w_conv1']
    b_conv1 = params['b_conv1']
    w_conv2 = params['w_conv2']
    b_conv2 = params['b_conv2']
    w_dense = params['w_dense']
    b_dense = params['b_dense']

    # Conv1
    x = conv2d(x, w_conv1, b_conv1)
    x = relu(x)
    x = max_pool(x, pool=(2,2), stride=(2,2))

    # Conv2
    x = conv2d(x, w_conv2, b_conv2)
    x = relu(x)
    x = max_pool(x, pool=(2,2), stride=(2,2))

    # Dense
    x = flatten(x)
    x = linear(x, w_dense, b_dense)

    return x

def cross_entropy_loss(params, x, y_onehot):
    logits = forward(params, x)
    log_probs = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    loss = -jnp.mean(jnp.sum(y_onehot * log_probs, axis=-1))
    return loss


def sgd_update(params, x, y_onehot, lr=0.01):
    loss_val, grads = value_and_grad(cross_entropy_loss)(params, x, y_onehot)
    # manual param update (simple SGD)
    new_params = {}
    for k in params:
        new_params[k] = params[k] - lr * grads[k]
    return new_params, loss_val



def one_hot(labels, num_classes=10):
    return jnp.eye(num_classes)[labels]


if __name__ == "__main__":
    params = {
        'w_conv1': jnp.ones((8, 1, 3, 3)),
        'b_conv1': jnp.zeros((8,)),

        'w_conv2': jnp.ones((16, 8, 3, 3)), 
        'b_conv2': jnp.zeros((16,)),

        'w_dense': jnp.ones((10, 7 * 7 * 16)),
        'b_dense': jnp.zeros((10,)),
    }
    train, test = make_datasets(batch_size=32)

    for images, labels in train.take(100):
        images_np = images.numpy()  # TensorFlow → NumPy
        labels_np = labels.numpy()
        
        images_jax = jnp.array(images_np)  # NumPy → JAX
        labels_jax = jnp.array(labels_np)
        
        images_jax = images_jax.astype(jnp.float32) / 255.0

        labels_onehot = one_hot(labels_jax, num_classes=10)

        new_params, loss_val = sgd_update(params, images_jax, labels_onehot, lr=0.01)
        print("Loss:", loss_val)
        

    