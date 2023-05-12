import os
import cv2
import matplotlib.pyplot as plt
from face_recognition import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



#read the images
directory = 'images'

face = []
labels = []  # to store the filenames

for foldername in os.listdir(directory):
       for filename in os.listdir(directory + '/' + foldername):
              face_image = cv2.imread(directory + '/' + foldername + '/' + filename, cv2.IMREAD_GRAYSCALE)
              face.append(face_image)
              labels.append(foldername)

face = np.asarray(face)
labels = np.asarray(labels)

X_train, X_test, y_train, y_test = train_test_split(face, labels, test_size=0.4, random_state=42)

face_vector = []

X_train = np.asarray(X_train)
for i in range(len(X_train)):
    face_vector.append(X_train[i].flatten())

face_vector = np.asarray(face_vector)
face_vector = face_vector.transpose()

weights, avg_face_vector, eigen_faces = PCA(face_vector)

prediction = []
for i in range(len(X_test)):
    test_img = X_test[i]
    index = test(test_img, weights, avg_face_vector, eigen_faces) 
    prediction.append(y_train[index])

prediction = np.asarray(prediction)

accuracy = accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)

plot_multiclass_roc(y_test, prediction,5)