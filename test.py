import os
import numpy as np
from PIL import Image

# Set the path to your image directory
image_dir = "/path/to/your/image/directory"

# Initialize empty arrays for face vectors and labels
face_vector = np.empty((0, 128))  # Assuming 128-dimensional feature vectors
y_train = []

# Loop over each image in the directory and extract features
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        # Read in the image using PIL
        img = Image.open(os.path.join(image_dir, filename)).convert('L')  # Convert to grayscale

        # Extract features from the image (e.g., using LBP, HOG, or CNNs)
        features = extract_features(img)

        # Add the features to the face vector array
        face_vector = np.vstack((face_vector, features))

        # Add the corresponding label to the labels array
        label = extract_label(filename)  # Extract the label from the filename
        y_train.append(label)

# Convert the labels array to a NumPy array
y_train = np.array(y_train)
