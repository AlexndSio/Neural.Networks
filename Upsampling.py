import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load matlab data
data1 = loadmat('data21.mat')
data2 = loadmat('data23.mat')
X_i = data2['X_i']
X_n = data2['X_n']
A_1 = data1['A_1']
B_1 = data1['B_1']
A_2 = data1['A_2']
B_2 = data1['B_2']

# Initialazations
instances = 1000
Z = np.random.randn(10, 1)
imgs = X_n.shape[1]
M = X_n.shape[0]
Zer = np.zeros((instances, imgs))

# Define T
size = 7
T = np.zeros((size**2, 28*28))
for i in range(1, size**2+1):
    org_row = 4 * ((i - 1) // size)
    org_col = 4 * ((i - 1) % size)
    for j in range(4):
        for k in range(4):
            org = (org_row + j) * 28 + (org_col + k)
            T[i-1, org] = 1 / 16
# Plots
fig, axs = plt.subplots(imgs, 3, figsize=(10, imgs*3))
fig2, axs2 = plt.subplots(imgs, 1, figsize=(10, imgs*3))

# For each image
for img_i in range(imgs):
    usable_part = X_n[:, img_i][:, np.newaxis]
    
    for i in range(instances):
        W1 = A_1 @ Z + B_1
        Z1 = np.maximum(W1, 0)
        W2 = A_2 @ Z1 + B_2
        X =  1/(1 + np.exp(W2))
        
        # Gradients
        U2 = (1 / np.linalg.norm(T @ X - usable_part)**2) * (2 * T.T @ (T @ X - usable_part))
        F2 = (-np.exp(W2) / (1 + np.exp(W2))**2).T
        F1 = (W1 > 0).astype(float)
        V2 = U2 * F2
        U1 = A_2.T @ V2
        V1 = U1 * F1
        U0 = A_1.T @ V1
        gradient = M * U0 + 2 * Z
        
        # Adam
        lr = 0.01
        b1 = 0.9
        b2 = 0.999
        epsilon = 1e-8
        m = np.zeros_like(Z)
        v = np.zeros_like(Z)
        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * (gradient**2)
        Z = Z - lr * m / (np.sqrt(v) + epsilon)
        Z = np.mean(Z, axis=1)[:, np.newaxis]  

        # Cost function
        J = M * np.log(np.linalg.norm(T @ X - usable_part)**2) + np.linalg.norm(Z)**2
        Zer[i, img_i] = J

    # Reshapes
    img_r = X.reshape(28, 28).T
    img_d = X_n[:, img_i].reshape(size, size).T
    img_ide = X_i[:, img_i].reshape(28, 28).T

    # Plot the images
    axs[img_i, 0].imshow(img_ide, cmap='gray')
    axs[img_i, 1].imshow(img_d, cmap='gray')
    axs[img_i, 2].imshow(img_r, cmap='gray')

    # Plot the cost function
    axs2[img_i].plot(np.arange(1, instances+1), Zer[:, img_i])

plt.tight_layout()
plt.show()