import time
import sys
import numpy as np
import random

SEED = 42


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def warm_tensorflow():
    """Import TensorFlow and run a couple of small traced functions to initialize kernels and backend."""
    print("Importing TensorFlow...")
    try:
        import tensorflow as tf
        import tensorflow.keras as keras
    except Exception as e:
        print("Failed to import TensorFlow. Ensure it is installed.")
        print("Error:", e)
        sys.exit(1)

    print(f"TensorFlow {tf.__version__} imported.")

    # Try to enable memory growth for GPUs if present
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
    except Exception:
        pass

    @tf.function
    def matmul_trace(x, y):
        return tf.matmul(x, y)

    a = tf.random.uniform((64, 64))
    b = tf.random.uniform((64, 64))
    _ = matmul_trace(a, b)
    _ = matmul_trace(a, b)

    @tf.function
    def conv_trace(x, w):
        return tf.nn.conv2d(x, w, strides=1, padding="SAME")

    x = tf.random.uniform((1, 32, 32, 3))
    w = tf.random.uniform((3, 3, 3, 8))
    _ = conv_trace(x, w)

    # Touch Keras backend to finish warm-up
    try:
        _ = keras.backend.zeros((1,))
        _ = keras.layers.Dense(1)
    except Exception:
        pass

    print("TensorFlow warm-up complete.")
    return {"tf_version": tf.__version__}


def main():
    set_seeds(SEED)
    warm_tensorflow()


if __name__ == "__main__":
    main()
