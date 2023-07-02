# 03-mle-python-surface-bd.py

import numpy as np
from scipy.optimize import minimize

# Define a variable N with a new numerical value
N = 100

# Create an array S ranging from 1 to N
S = np.arange(1, N+1)

# Create an array theta ranging from 0.1 to 0.9 with 100 elements
theta = np.linspace(0.1, 0.9, 100)

# Create a 2D grid of S and theta values
S_grid, theta_grid = np.meshgrid(S, theta)

# Calculate a modified likelihood function L(theta|S) with new formula
L = S_grid * np.log(theta_grid) + (N - S_grid) * np.log(1 - theta_grid)

# Creating a 3D plot to visualize the likelihood function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
s = ax.plot_surface(S_grid, theta_grid, L, cmap='jet')

# Labeling the axes
ax.set_xlabel('S')
ax.set_ylabel('theta')
ax.set_zlabel('L')
ax.set_title('Likelihood Function L(theta|S)')
ax.view_init(65, 15)

# Saving the plot to a file
plt.savefig('s-theta-L.png')

# Uncomment the line below to display the plot
# plt.show()

# Define the surface boundary estimation function
def surface_boundary_estimation(data):
    # Define the likelihood function
    def likelihood(parameters):
        a, b, c = parameters
        predicted = a * data[:, 0] + b * data[:, 1] + c
        probabilities = 1 / (1 + np.exp(-predicted))
        loglikelihood = np.sum(data[:, 2] * np.log(probabilities) + (1 - data[:, 2]) * np.log(1 - probabilities))
        return -loglikelihood

    # Perform maximum likelihood estimation
    result = minimize(likelihood, [0.0, 0.0, 0.0], method='BFGS')

    # Retrieve the estimated parameters
    a_hat, b_hat, c_hat = result.x

    return a_hat, b_hat, c_hat

# Example usage
data = np.array([[1.2, 2.5, 0],
                 [2.5, 3.7, 1],
                 [3.7, 4.8, 1],
                 [4.8, 5.6, 0]])  # Replace with your own data

a, b, c = surface_boundary_estimation(data)
print("Estimated parameters: a =", a, "b =", b, "c =", c)
