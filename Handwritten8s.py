import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load matlab data
data = loadmat('data21.mat')
A_1 = data['A_1']
B_1 = data['B_1']
A_2 = data['A_2']
B_2 = data['B_2']

plt.figure()

for i in range(100):
    # Create input with mean = 0 and deviation = 1
    Z = np.random.randn(10, 1)
    # Use the data to create 8s
    W1 = A_1 @ Z  + B_1
    Z1 = np.maximum(W1, 0)
    W2 = A_2 @ Z1 + B_2
    X = 1 / (1 + np.exp(W2))

    X2D = X.reshape(28, 28).T

    # Display the results
    plt.subplot(10, 10, i+1)
    plt.imshow(X2D, cmap='gray')
    plt.axis('off')

plt.show()