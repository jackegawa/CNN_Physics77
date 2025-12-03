import numpy as np
from data import load_mnist
from model import CNN
from operations import SoftMaxOp
from tensor import Tensor
from optim import SGD
import matplotlib.pyplot as plt
import time
import tracemalloc

def train_no_batching(num_epochs=1, lr=0.007):

    tracemalloc.start()
    total_start = time.time()

    # 1. Load data
    x_train, y_train, x_test, y_test = load_mnist()

    # 2. Build model
    model = CNN()
    softmax = SoftMaxOp()
    optim = SGD(model.params(), lr=lr)

    n_samples = x_train.shape[0]

    step_times = []
    epoch_times = []

    for epoch in range(num_epochs):
        print(f"\nEPOCH {epoch+1}/{num_epochs}")

        epoch_start = time.time()

        # Shuffle dataset
        idx = np.random.permutation(n_samples)
        x_train = x_train[idx]
        y_train = y_train[idx]

        running_loss = 0
        l = []

        for i in range(n_samples):

            step_start = time.perf_counter()

            x = Tensor(x_train[i:i+1])   
            logits = model.forward(x)

            y = y_train[i:i+1]
            loss = softmax(logits, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.data
            l.append(loss.data)

            step_end = time.perf_counter()
            step_times.append((step_end - step_start) * 1000)  # ms

            if i % 100 == 0:
                print(f" sample {i}/{n_samples} - loss {loss.data:.4f}")

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f" Epoch Time: {epoch_time:.4f} sec")

        print(f" Average loss: {running_loss / n_samples:.4f}")
        plt.plot(np.arange(n_samples), l)
        plt.show()

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

    # END TRAINING — final summary
    total_time = time.time() - total_start
    current, peak = tracemalloc.get_traced_memory()

    print("\n================== TRAINING SUMMARY ==================")
    print(f"Total training time: {total_time:.2f} sec\n")

    print(f"Epoch times:")
    for i, t in enumerate(epoch_times):
        print(f"  Epoch {i+1}: {t:.2f} sec")
    print()

    print(f"Avg step time: {np.mean(step_times):.3f} ms")
    print(f"Min step time: {np.min(step_times):.3f} ms")
    print(f"Max step time: {np.max(step_times):.3f} ms\n")

    print(f"Current memory: {current/1e6:.3f} MB")
    print(f"Peak memory:    {peak/1e6:.3f} MB")
    print("======================================================")

train_no_batching()
