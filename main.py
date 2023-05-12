import os
import cv2
import matplotlib.pyplot as plt
from face_recognition import *



#read the images
directory = 'detected faces 250'

face_vector = []
face = []
filenames = []  # to store the filenames
for filename in os.listdir(directory):
       face_image = cv2.imread(directory + '/' + filename, cv2.IMREAD_GRAYSCALE)
       face.append(face_image)
       face_image = face_image.flatten()
       face_vector.append(face_image)
       filenames.append(filename)

face = np.asarray(face)
face_vector = np.asarray(face_vector)
face_vector = face_vector.transpose()


weights, avg_face_vector, eigen_faces = PCA(face_vector)
test_img = cv2.imread('images_for_predection/saeid (15).pngface.jpg',cv2.IMREAD_GRAYSCALE)
index = test(test_img, weights,avg_face_vector, eigen_faces)
matched_image_filename = filenames[index]  # get the filename of the matched image
matched_image_name = matched_image_filename.split("(")[0]  # extract the person name
print("Matched image: ", matched_image_name)
plt.imshow(face[index], cmap="gray")
plt.show()