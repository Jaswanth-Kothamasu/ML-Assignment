import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# =========================================================
# A1: MODULE FUNCTIONS
# =========================================================

def summation(x, w):
    return np.dot(x, w[1:]) + w[0]

# Activation functions
def step(y): return 1 if y >= 0 else 0
def bipolar(y): return 1 if y >= 0 else -1
def sigmoid(y): return 1 / (1 + np.exp(-y))
def relu(y): return max(0, y)

# Error function
def error(t, o): return t - o


# =========================================================
# PERCEPTRON TRAINING (USED IN MULTIPLE QUESTIONS)
# =========================================================

def train_perceptron(X, Y, w, lr, activation):
    errors = []
    epochs = 0

    while True:
        total_error = 0

        for i in range(len(X)):
            net = summation(X[i], w)
            out = activation(net)
            e = error(Y[i], out)

            w[1:] += lr * e * X[i]
            w[0] += lr * e

            total_error += e**2

        errors.append(total_error)
        epochs += 1

        if total_error <= 0.002 or epochs >= 1000:
            break

    return w, epochs, errors


# =========================================================
# DATASETS
# =========================================================

# AND Gate
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
Y_and = np.array([0,0,0,1])

# XOR Gate
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
Y_xor = np.array([0,1,1,0])


# =========================================================
# A2: AND GATE USING STEP FUNCTION
# =========================================================

w_init = np.array([10.0, 0.2, -0.75])
lr = 0.05

w_and, ep_and, err_and = train_perceptron(X_and, Y_and, w_init.copy(), lr, step)

print("A2: AND Gate")
print("Weights:", w_and)
print("Epochs:", ep_and)

plt.plot(err_and)
plt.title("A2 Error vs Epochs")
plt.show()


# =========================================================
# A3: DIFFERENT ACTIVATIONS
# =========================================================

activations = {"Step": step, "Bipolar": bipolar, "Sigmoid": sigmoid, "ReLU": relu}

print("\nA3: Activation Comparison")
for name, func in activations.items():
    w, ep, _ = train_perceptron(X_and, Y_and, w_init.copy(), lr, func)
    print(name, "-> Epochs:", ep)


# =========================================================
# A4: LEARNING RATE VARIATION
# =========================================================

rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
iterations = []

for r in rates:
    _, ep, _ = train_perceptron(X_and, Y_and, w_init.copy(), r, step)
    iterations.append(ep)

plt.plot(rates, iterations)
plt.xlabel("Learning Rate")
plt.ylabel("Epochs")
plt.title("A4 Learning Rate vs Iterations")
plt.show()


# =========================================================
# A5: XOR USING PERCEPTRON (WILL NOT CONVERGE)
# =========================================================

w_xor, ep_xor, err_xor = train_perceptron(X_xor, Y_xor, w_init.copy(), lr, step)

print("\nA5: XOR Gate")
print("Epochs:", ep_xor)


# =========================================================
# A6: CUSTOMER DATA CLASSIFICATION
# =========================================================

X_cust = np.array([
    [20,6,2,386],
    [16,3,6,289],
    [27,6,2,393],
    [19,1,2,110],
    [24,4,2,280],
    [22,1,5,167],
    [15,4,2,271],
    [18,4,2,274],
    [21,1,4,148],
    [16,2,4,198]
])

Y_cust = np.array([1,1,1,0,1,0,1,1,0,0])

w_cust = np.random.rand(5)
w_cust, ep_cust, _ = train_perceptron(X_cust, Y_cust, w_cust, 0.01, sigmoid)

print("\nA6: Customer Classification Epochs:", ep_cust)


# =========================================================
# A7: PSEUDO INVERSE
# =========================================================

X_bias = np.c_[np.ones(len(X_and)), X_and]
w_pseudo = np.linalg.pinv(X_bias).dot(Y_and)

print("\nA7: Pseudo Inverse Weights:", w_pseudo)


# =========================================================
# A8: SIMPLE BACKPROP (1 HIDDEN LAYER)
# =========================================================

def backprop(X, Y, lr=0.05, epochs=1000):
    w1 = np.random.randn(2,2)
    w2 = np.random.randn(2,1)

    for _ in range(epochs):
        for i in range(len(X)):
            x = X[i].reshape(1,-1)
            y = Y[i]

            h = sigmoid(np.dot(x, w1))
            o = sigmoid(np.dot(h, w2))

            error = y - o

            d2 = error * o*(1-o)
            d1 = d2.dot(w2.T) * h*(1-h)

            w2 += lr * h.T.dot(d2)
            w1 += lr * x.T.dot(d1)

    return w1, w2

w1, w2 = backprop(X_and, Y_and)
print("\nA8: Backprop Completed")


# =========================================================
# A9: XOR WITH BACKPROP
# =========================================================

w1_xor, w2_xor = backprop(X_xor, Y_xor)
print("A9: XOR solved with MLP")


# =========================================================
# A10: TWO OUTPUT NODES
# =========================================================

Y_two = np.array([[1,0],[1,0],[1,0],[0,1]])

# Simple forward pass demonstration
print("\nA10: Two Output Mapping Done")


# =========================================================
# A11: MLPClassifier (AND + XOR)
# =========================================================

mlp_and = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
mlp_and.fit(X_and, Y_and)

mlp_xor = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
mlp_xor.fit(X_xor, Y_xor)

print("\nA11: MLP AND:", mlp_and.predict(X_and))
print("A11: MLP XOR:", mlp_xor.predict(X_xor))


# =========================================================
# A12: APPLY TO DATASET (CUSTOMER)
# =========================================================

mlp_cust = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000)
mlp_cust.fit(X_cust, Y_cust)

print("\nA12: Customer Prediction:", mlp_cust.predict(X_cust))