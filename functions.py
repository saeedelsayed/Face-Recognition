import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
counter = 0

def get_reference_images(directory = 'dataset3'):
    # Set up a list of reference faces
    reference_faces = []
    reference_labels = []

    number = -1

    # Access the directory that contains all the images folders
    for foldername in os.listdir(directory):
        number += 1
        # Access all the images files in each folder
        for filename in os.listdir(directory + '/' + foldername):
                # Read each face image and store it and its label in the lists
                face_image = cv2.imread(directory + '/' + foldername + '/' + filename, cv2.IMREAD_GRAYSCALE)
                reference_faces.append(face_image)
                reference_labels.append(number)

    # Convert the lists to arrays
    reference_faces = np.asarray(reference_faces)
    reference_labels = np.asarray(reference_labels)

    reference_faces_vector = []

    for face in reference_faces:
        reference_faces_vector.append(face.flatten())
    reference_faces_vector = np.asarray(reference_faces_vector).transpose()
    
    return reference_faces,reference_labels,reference_faces_vector


def apply_pca(reference_faces_vector):
    """
    Apply the PCA model to the referecnce faces
    
    ## Parameters 
    face_vector : contain the training images where each column contain one image of size (N^2 * 1)
    
    ## Returns
    weights : vector contain the similarity between each image and eigen vectors

    avg_face_vector : the mean image, used in testing

    eigen_faces : contain eigen vectors, used in testing
    """
    #get the mean image
    avg_face_vector = reference_faces_vector.mean(axis=1)
    avg_face_vector = avg_face_vector.reshape(reference_faces_vector.shape[0], 1)
    #subtract the mean image from the images
    normalized_face_vector = reference_faces_vector - avg_face_vector

    #calculate covariance matrix
    covariance_matrix = np.cov(np.transpose(normalized_face_vector)) 

    #get eigen values and eigen vectors
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    #select best k eigen vectors . this variable is changable according to the data set
    k = 40
    index_of_max_k_eigen_values = np.argpartition(eigen_values, -k)[-k:]
    k_eigen_vectors = []

    for i in range(len(index_of_max_k_eigen_values)):
        k_eigen_vectors.append(eigen_vectors[index_of_max_k_eigen_values[i]])

    k_eigen_vectors = np.asarray(k_eigen_vectors)

    eigen_faces = k_eigen_vectors.dot(normalized_face_vector.T)
    weights = (normalized_face_vector.T).dot(eigen_faces.T)

    return weights, avg_face_vector, eigen_faces


def recognize_face(input_img, avg_face_vector, eigen_faces,model):
    """
    Recognize faces in test image

    ## Parameters
    test_img : image to recognize faces in , it is in gray scale

    weights : the weights resulting from PCA model

    ave_face_vector : average face from the training images

    eigen_faces : eigen vector resulting from PCA model
    
    ## Returns
    index : index of the matched image

    """
    input_img = input_img.reshape(input_img.shape[0] * input_img.shape[1], 1)
    test_normalized_face_vector = input_img - avg_face_vector
    test_weight = (test_normalized_face_vector.T).dot(eigen_faces.T)
    
    return model.predict(test_weight)[0]

def detect_faces(input_image, pca_parameters, model):

    gray_input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)

    # Draw a bounding box around the detected face and label it with the closest reference face
    faces_detected = face_cascade.detectMultiScale(gray_input_image, scaleFactor = 1.3, minNeighbors=5)

    font_size = 30
    font_scale = min(input_image.shape[:2]) / (20 * font_size)
    thickness = max(3, int(font_scale))

    if len(faces_detected) > 0:

        for (x, y, w, h) in faces_detected:
            cropped_face_img = gray_input_image[y:y+h, x:x+w]
            resized_cropped_face_img = cv2.resize(cropped_face_img,(250,250))

            label = recognize_face(resized_cropped_face_img, pca_parameters[1], pca_parameters[2], model)

            names = ['Abdelrahman', 'Diaa', 'Saeed', 'Sarta', 'Sherif']

            cv2.rectangle(input_image, (x, y), (x+w, y+h), (0, 0, 255), thickness)
            cv2.putText(input_image, f"{names[label]}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
    return input_image

def draw_roc_curve(y_test, y_prob,n_classes):
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot the ROC curves for each class
    plt.figure()

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
            label='ROC curve of class {0} (area = {1:0.2f})'
            ''.format(i, roc_auc[i]))
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multinomial logistic regression')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")