from tensorflow.keras.datasets import mnist

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize to [0,1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # Add channel dimension (N, 1, 28, 28)
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test  = x_test.reshape(-1, 1, 28, 28)

    return x_train, y_train, x_test, y_test
