# 04a-mle-python-surface-bd-exr.py

import numpy as np
from scipy.optimize import minimize


def L(S, o):
    # Sample implementation of L(S, o)
    return np.sin(S) * np.cos(o)

# Generate data for heatmap
S = np.linspace(0, 1, 100)
o = np.linspace(0, 1, 100)
S, o = np.meshgrid(S, o)
Z = L(S, o)

# Plot p2 (heatmap)
plt.subplot(2, 1, 1)
plt.imshow(Z, extent=(0, 1, 0, 1), origin='lower', cmap='jet')
plt.xlabel('S')
plt.ylabel('o')
plt.title("Bird's eye view")
plt.axvline(10, color='black')  # Add vertical line

# Plot p3 (curve)
plt.subplot(2, 1, 2)
plt.plot(o, L(25, o))
plt.xlabel('o')
plt.title('L(o|S=10)')

# Adjust spacing between subplots
plt.tight_layout()

# Save the figure
plt.savefig('python_fig.png')

# Display the figure
plt.show()

# Define the surface boundary estimation function with an extra parameter for regularization
def surface_boundary_estimation(data, regularization_param):
    # Define the likelihood function with regularization
    def likelihood(parameters):
        a, b, c = parameters[:3]
        regularization_term = regularization_param * np.sum(parameters[3:]**2)
        predicted = a * data[:, 0] + b * data[:, 1] + c
        probabilities = 1 / (1 + np.exp(-predicted))
        loglikelihood = np.sum(data[:, 2] * np.log(probabilities) + (1 - data[:, 2]) * np.log(1 - probabilities))
        return -loglikelihood + regularization_term

    # Perform maximum likelihood estimation with regularization
    initial_parameters = np.zeros(6)  # Replace with the appropriate number of parameters
    result = minimize(likelihood, initial_parameters, method='BFGS')

    # Retrieve the estimated parameters
    a_hat, b_hat, c_hat = result.x[:3]

    return a_hat, b_hat, c_hat

# Example usage
data = np.array([[1.2, 2.5, 0],
                 [2.5, 3.7, 1],
                 [3.7, 4.8, 1],
                 [4.8, 5.6, 0]])  # Replace with your own data
regularization_param = 0.1  # Replace with your desired regularization parameter

a, b, c = surface_boundary_estimation(data, regularization_param)
print("Estimated parameters for 04a: a =", a, "b =", b, "c =", c)
