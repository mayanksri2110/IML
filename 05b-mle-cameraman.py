import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cv2

# Read the image as an array
image = cv2.imread('/content/space.tif')

# Convert the image to grayscale and normalize the values between 0 and 1
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255

# Set the number of repetitions for the image
T = 100

# Repeat the grayscale image T times along the third axis
repeated_image = np.repeat(gray_image[:, :, np.newaxis], T, axis=2)

# Generate random values following a Poisson distribution based on the repeated image
random_poisson = np.random.poisson(repeated_image)

# Create a binary mask based on a threshold value
threshold = 1
binary_mask = (random_poisson >= threshold).astype(float)

# Estimate the parameter using the binary mask
estimated_parameter = -np.log(1 - np.mean(binary_mask, axis=2))

# Display the estimated parameter as a grayscale image
plt.imshow(estimated_parameter, cmap='gray')
plt.show()
