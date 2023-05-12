from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


"""
description: Creating the PCA model
parameters : face_vector ---> contain the training images where each column contain one image
             of size (N^2 * 1)
return value : weights ----> vector contain the similarity between each image and eigen vectors
               avg_face_vector ----> the mean image ,used in testing
               eigen_faces ----> contain eigen vectors ,used in testing

"""

def PCA(face_vector):
    #get the mean image
    avg_face_vector = face_vector.mean(axis=1)
    avg_face_vector = avg_face_vector.reshape(face_vector.shape[0], 1)
    #subtract the mean image from the images
    normalized_face_vector = face_vector - avg_face_vector

    #calculate covariance matrix
    covariance_matrix = np.cov(np.transpose(normalized_face_vector)) 

    #get eigen values and eigen vectors
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    #select best k eigen vectors . this variable is changable according to the data set
    k = 100
    index_of_max_k_eigen_values = np.argpartition(eigen_values, -k)[-k:]
    k_eigen_vectors = []

    for i in range(len(index_of_max_k_eigen_values)):
        k_eigen_vectors.append(eigen_vectors[index_of_max_k_eigen_values[i]])

    k_eigen_vectors = np.asarray(k_eigen_vectors)

    eigen_faces = k_eigen_vectors.dot(normalized_face_vector.T)
    weights = (normalized_face_vector.T).dot(eigen_faces.T)

    return weights, avg_face_vector, eigen_faces

"""
description : recognize faces in test image
parameters  : test_img ---> image to recognize faces in , it is in gray scale
              weights ---> the weights resulting from PCA model
              ave_face_vector ---> average face from the training images
              eigen_faces ---> eigen vector resulting from PCA model
return value : index ---> index of the matched image

"""

def test(test_img, weights, avg_face_vector, eigen_faces):
    test_img = test_img.reshape(test_img.shape[0] * test_img.shape[1], 1)
    test_normalized_face_vector = test_img - avg_face_vector
    test_weight = (test_normalized_face_vector.T).dot(eigen_faces.T)
    index =  np.argmin(np.linalg.norm(test_weight - weights, axis=1))

    return index



def plot_roc_curve(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()