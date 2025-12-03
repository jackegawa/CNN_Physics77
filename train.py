import numpy as np
from data import load_mnist
from model import CNN
from operations import SoftMaxOp
from tensor import Tensor
from optim import SGD
import matplotlib.pyplot as plt

def train_no_batching(num_epochs=1, lr=0.01):

    x_train, y_train, x_test, y_test = load_mnist()

    model = CNN()
    softmax = SoftMaxOp()
    optim = SGD(model.params(), lr=lr)

    n_samples = x_train.shape[0]

    avg_losses = []    # store per-epoch loss
    l = []
    for epoch in range(num_epochs):
        print(f"\nEPOCH {epoch+1}/{num_epochs}")

        idx = np.random.permutation(n_samples)
        x_train = x_train[idx]
        y_train = y_train[idx]

        running_loss = 0.0

        for i in range(n_samples):

            x = Tensor(x_train[i:i+1])
            logits = model.forward(x)

            y = y_train[i:i+1]
            loss = softmax(logits, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.data.item()

            if i % 1000 == 0:
                print(f" sample {i}/{n_samples} - loss {loss.data.item():.4f}")
                if loss.data.item() < 3.0:
                    l.append(loss.data.item())
        epoch_loss = running_loss / n_samples
        avg_losses.append(epoch_loss)

        print(f" Average loss: {epoch_loss:.4f}")
    plt.plot(l, label="loss")
    plt.xlabel("samples")
    plt.ylabel("loss")

    # Plot after all epochs
    plt.plot(avg_losses)
    plt.title("Training Loss Per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Cross Entropy Loss")
    plt.show()

train_no_batching(num_epochs=1,lr=0.005)

