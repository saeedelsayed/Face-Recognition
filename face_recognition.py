# from sklearn.decomposition import PCA
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle



def PCA(face_vector):
    """
    Creating the PCA model
    
    ## Parameters 
    face_vector : contain the training images where each column contain one image of size (N^2 * 1)
    
    ## Returns
    weights : vector contain the similarity between each image and eigen vectors

    avg_face_vector : the mean image, used in testing

    eigen_faces : contain eigen vectors, used in testing
    """
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
    k = 30
    index_of_max_k_eigen_values = np.argpartition(eigen_values, -k)[-k:]
    k_eigen_vectors = []

    for i in range(len(index_of_max_k_eigen_values)):
        k_eigen_vectors.append(eigen_vectors[index_of_max_k_eigen_values[i]])

    k_eigen_vectors = np.asarray(k_eigen_vectors)

    eigen_faces = k_eigen_vectors.dot(normalized_face_vector.T)
    weights = (normalized_face_vector.T).dot(eigen_faces.T)

    return weights, avg_face_vector, eigen_faces

def test(test_img, weights, avg_face_vector, eigen_faces):
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
    test_img = test_img.reshape(test_img.shape[0] * test_img.shape[1], 1)
    test_normalized_face_vector = test_img - avg_face_vector
    test_weight = (test_normalized_face_vector.T).dot(eigen_faces.T)
    index =  np.argmin(np.linalg.norm(test_weight - weights, axis=1))

    return index


def plot_multiclass_roc(y_true, y_score, n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot ROC curves for each class
    plt.figure(figsize=(8,6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blueviolet', 'darkgreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    # Plot macro-average ROC curve
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # Add labels and legend
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for multiclass classification')
    plt.legend(loc="lower right")
    plt.show()