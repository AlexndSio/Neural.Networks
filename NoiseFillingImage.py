import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load matlab data
data1 = loadmat('data21.mat')
data2 = loadmat('data22.mat')
X_i = data2['X_i']
X_n = data2['X_n']
A_1 = data1['A_1']
B_1 = data1['B_1']
A_2 = data1['A_2']
B_2 = data1['B_2']

# Initializations
instances = 1000
N = 500
Zer= np.zeros((instances, 4))  

# Define T,T1
T = np.eye(784)[:N, :] 

T1 = np.zeros((784, 784))
np.fill_diagonal(T1[:N, :N], 1)

# Define ReLU derivative
def relu_derivative(x):
    return (x > 0).astype(float)

for col in range(4):
    X_i_col = X_i[:, col][:, np.newaxis]
    X_n_col = X_n[:, col][:, np.newaxis]
    usable_part = X_n_col[:N, :]

    for i in range(instances):
        Z = np.random.randn(10, 1)
        W1 = A_1 @ Z + B_1
        Z1 = np.maximum(W1, 0)
        W2 = A_2 @ Z1 + B_2
        X =  1/(1 + np.exp(W2))
        # Gradients
        U2 = 2 * ((T @ X - usable_part).T @ T) / (np.linalg.norm(T @ X - usable_part)**2)
        V2 = U2 * (-np.exp(W2)/(1+np.exp(W2))**2)
        U1 = A_2.T @ V2
        V1 = U1 * relu_derivative(W1)
        U0 = A_1.T @ V1
        Z = Z -  0.001 * (U0 + Z)
        # Cost function
        J = N * np.linalg.norm(U0)**2 + np.linalg.norm(Z)**2
        Zer[i, col] = J

    # Reshapes
    Original_img = X_i_col.reshape(28, 28).T
    X_nh = (T1@X_n).reshape(28, 28*4).T
    Final_img = X.reshape(28, 28).T

    # Plot the 8s
    plt.subplot(4, 3, col * 3 + 1)
    plt.imshow(Original_img, cmap='gray', aspect='auto')
    plt.axis('off')

    plt.subplot(4, 3, col * 3 + 2)
    plt.imshow(X_nh, cmap='gray', aspect='auto')
    plt.axis('off')

    plt.subplot(4, 3, col * 3 + 3)
    plt.imshow(Final_img, cmap='gray', aspect='auto')
    plt.axis('off')

plt.show()
# Plot the cost function
plt.figure(figsize=(10, 5))

for col in range(4):
    plt.subplot(2, 2, col+1)
    plt.plot(Zer[:, col])

plt.tight_layout()
plt.show()