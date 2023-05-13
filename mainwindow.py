import cv2
import sys
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from functions import get_reference_images, apply_pca, recognize_face,calculate_accuracy

# Get the reference images and the their labels
reference_faces,reference_labels, reference_faces_vector = get_reference_images()

# Get the weights, eigen vectors and eigen values of the vector
weights, avg_face_vector, eigen_faces = apply_pca(reference_faces_vector)


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("PCA Face Recognition")
        self.resize(200,200)

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.StartBtn = QPushButton("Start")
        self.StartBtn.clicked.connect(self.start_webcam)

        self.StopBtn = QPushButton("Stop")
        self.StopBtn.clicked.connect(self.stop_webcam)

        self.VBL.addWidget(self.StartBtn)
        self.VBL.addWidget(self.StopBtn)

        self.setLayout(self.VBL)

    def start_webcam(self):
        # Create a video capture object
        self.capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

        # Set the frame size to match the label size
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        # Create a timer to update the video stream
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)  # 5 milliseconds

    def stop_webcam(self):
        self.capture.release()
        sys.exit()

    def update_frame(self):
        # Read the frame
        stream, frame = self.capture.read()
        flipped_frame = cv2.flip(frame, 1)

        # Calculate the PCA features of the frame

        gray_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)

        # Load the cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

        # Draw a bounding box around the detected face and label it with the closest reference face
        faces_detected = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        if len(faces_detected) > 0:
            # Initialize an empty list to store the cropped face images

            for (x, y, w, h) in faces_detected:
                face_cropped_img = gray_frame[y:y+h, x:x+w]
                resized_face_cropped_img = cv2.resize(face_cropped_img,(250,250))
                closest_idx = recognize_face(resized_face_cropped_img, weights, avg_face_vector, eigen_faces)
                calculated_accuracy = calculate_accuracy(reference_faces,reference_labels)*100

                cv2.rectangle(flipped_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(flipped_frame, f"{reference_labels[closest_idx]} - {calculated_accuracy}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Convert the frame to RGB format
        flipped_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

        # Create a QImage from the frame data
        img = QImage(flipped_frame, flipped_frame.shape[1], flipped_frame.shape[0], QImage.Format_RGB888)

        # Create a QPixmap from the QImage
        pix = QPixmap.fromImage(img)

        # Set the pixmap on the label to display the video stream
        self.FeedLabel.setPixmap(pix)