"""
=========================================================
DOOFENSHMIRTZ EVIL INCORPORATED - STARTER KIT
=========================================================
"Behold! The Starter-Kit-Inator! It provides all the basic 
code you need to interact with my data and my discount AI."

This script demonstrates how to:
1. Load a numpy image (.npy)
2. Plot the image
3. Plot a histogram of the pixel values
4. Load the pre-trained Oracle model (.pkl)
5. Run inference (predict and predict_proba)
6. Rotate an image using scikit-image
=========================================================
"""

import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage.transform import rotate

# --- Set your paths here ---
ROOT = "./" # Change this to the directory where you downloaded the data and model
IMAGE_PATH = os.path.join(ROOT, "subtask1/0.npy")
MODEL_PATH = os.path.join(ROOT, "doof_oracle.pkl")

# =======================================================
# 1. How to load an image from a numpy file
# =======================================================
print("1. Loading image from disk...")
# The images are stored as 2D numpy arrays of shape (28, 28)
img = np.load(IMAGE_PATH)
print(f"Loaded image with shape: {img.shape} and data type: {img.dtype}")

# =======================================================
# 2. How to plot the image
# =======================================================
print("2. Plotting image...")
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap="gray")
plt.title("Original Security Camera Image")
plt.axis("off") # Hides the axes for a cleaner look
plt.show()

# =======================================================
# 3. How to plot a histogram of the pixel values
# =======================================================
print("3. Plotting pixel value histogram...")
plt.figure(figsize=(6, 4))
# .ravel() flattens the 2D image into a 1D array for the histogram
plt.hist(img.ravel(), bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.title("Histogram of Pixel Intensities")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.5)
plt.show()

# =======================================================
# 4. How to load the pickle model
# =======================================================
print("4. Loading the Discount Oracle AI...")
with open(MODEL_PATH, "rb") as f:
    oracle = pickle.load(f)
print("Model loaded successfully!")
# The model classes are: 0 (Upright), 1 (Upside-Down), 2 (Sideways)
print(f"Oracle known classes: {oracle.classes_}")

# =======================================================
# 5. How to predict and use predict_proba()
# =======================================================
print("\n5. Running Inference...")

# IMPORTANT: scikit-learn models expect a 2D array of shape (n_samples, n_features).
# Since our image is (28, 28), we must flatten it to (1, 784) before feeding it to the model.
flat_img = img.reshape(1, -1)

# .predict() gives you the hard class label (0, 1, or 2)
predicted_class = oracle.predict(flat_img)[0]
print(f"Hard Prediction (Class ID): {predicted_class}")

# .predict_proba() gives you the probability distribution across all 3 classes
probabilities = oracle.predict_proba(flat_img)[0]
print("Probabilities:")
print(f"  - Upright (0)     : {probabilities[0]:.4f}")
print(f"  - Upside-Down (1) : {probabilities[1]:.4f}")
print(f"  - Sideways (2)    : {probabilities[2]:.4f}")

# =======================================================
# 6. How to rotate an image
# =======================================================
print("\n6. Rotating the image...")

# Let's rotate the image by 45 degrees.
# resize=False ensures the output image stays 28x28.
rotated_img = rotate(img, angle=45, resize=False, order=1, preserve_range=True)

plt.figure(figsize=(5, 5))
plt.imshow(rotated_img, cmap="gray")
plt.title("Image Rotated by 45 Degrees")
plt.axis("off")
plt.show()

# You can now flatten this rotated image and feed it back into the oracle
flat_rotated = rotated_img.reshape(1, -1)
rot_probs = oracle.predict_proba(flat_rotated)[0]
print("Probabilities after 45-degree rotation:")
print(f"  - Upright (0)     : {rot_probs[0]:.4f}")
print(f"  - Upside-Down (1) : {rot_probs[1]:.4f}")
print(f"  - Sideways (2)    : {rot_probs[2]:.4f}")

print("\nStarter Kit execution complete! Now go stop Perry the Platypus!")

"""
# CODE FOR SUBMISSION PREPARATION

import pandas as pd

import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage.transform import rotate

ROOT = "./" # Change this to the directory where you downloaded the data and model

task1_path = os.path.join(ROOT, "subtask1")
task2_path = os.path.join(ROOT, "subtask2") 
task3_path = os.path.join(ROOT, "subtask3")


MODEL_PATH = os.path.join(ROOT, "doof_oracle.pkl")
with open(MODEL_PATH, "rb") as f:
    oracle = pickle.load(f)

result = []

# Iterating through the files in the task1 directory to load the images and make predictions

for path in os.listdir(task1_path):
    if not path.endswith('.npy'):
        continue
    IMAGE_PATH = os.path.join(task1_path, path)
    img = np.load(IMAGE_PATH)
    flat_img = img.reshape(1, -1)
    predicted_class = oracle.predict(flat_img)[0] # This should be replaced with logic to solve the task  
    # Extracting the ID from the filename (assuming format "ID.npy")                                        
    datapoint_id = int(path.split('.')[0])
    # Adding the prediction for Task1 to the result list
    result.append({
        'subtaskID': 1,             # ID-ul subtask-ului
        'datapointID': datapoint_id, # ID-ul datapoint-ului extras din numele fișierului
        'answer': predicted_class    # Răspunsul pentru Task1
    })

### DO THE SAME FOR TASK2 AND TASK3, REPLACING THE PREDICTION LOGIC WITH THE APPROPRIATE ONE FOR EACH TASK ###

# Create a DataFrame from the result list
df_output = pd.DataFrame(result)

# Display the first 5 rows of the resulting DataFrame
df_output.head()

# Save the results to a CSV file that can be submitted on the platform
df_output.to_csv('submission.csv', index=False)

"""