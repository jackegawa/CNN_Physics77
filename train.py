import numpy as np
from data import load_mnist
from model import CNN
from operations import SoftMaxOp
from tensor import Tensor
from optim import SGD


def train_no_batching(num_epochs=1, lr=0.01):

    # 1. Load data
    x_train, y_train, x_test, y_test = load_mnist()

    # 2. Build model
    model = CNN()
    softmax = SoftMaxOp()
    optim = SGD(model.params(), lr=lr)

    n_samples = x_train.shape[0]

    for epoch in range(num_epochs):
        print(f"\nEPOCH {epoch+1}/{num_epochs}")

        # Shuffle dataset
        idx = np.random.permutation(n_samples)
        x_train = x_train[idx]
        y_train = y_train[idx]

        running_loss = 0

        for i in range(n_samples):
            if i%50 == 0:
                print("WORKING.....")
            # -----------------------
            # Make input Tensor
            # -----------------------
            x = Tensor(x_train[i:i+1])   # shape stays (1,1,28,28)
            # -----------------------
            # Forward pass
            # -----------------------
            logits = model.forward(x)

            # -----------------------
            # Loss (softmax + CE)
            # -----------------------
            y = y_train[i:i+1]  # shape (1,)
            loss = softmax(logits, y)
            # -----------------------
            # Backprop
            # -----------------------
            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.data

            if i % 100 == 0:
                print(f" sample {i}/{n_samples} - loss {loss.data:.4f}")

        print(f" Average loss: {running_loss / n_samples:.4f}")

        # -----------------------
        # Evaluate on test set
        # -----------------------
        correct = 0
        for j in range(x_test.shape[0]):
            x = Tensor(x_test[j:j+1])
            logits = model.forward(x)
            pred = np.argmax(logits.data)
            if pred == y_test[j]:
                correct += 1

        acc = correct / x_test.shape[0]
        print(f" Test Accuracy: {acc * 100:.2f}%")


train_no_batching()