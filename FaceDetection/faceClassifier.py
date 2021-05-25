import cv2
import numpy as np
import os

from numpy.core.fromnumeric import shape

# KNN CODE

# Euclidean Distance


def dist(x1, x2):
    return np.sqrt(sum((x1-x2)**2))


def knn(train, test, k=5):

    distances = []
    m = train.shape[0]

    for i in range(m):

        ix = train[i, :-1]
        iy = train[i, -1]
        d = dist(test, ix)
        distances.append((d, iy))

    distances = sorted(distances, key=lambda x: x[0])

    # Nearest first k distance
    distances = distances[:k]

    distances = np.array(distances)

    new_dist = np.unique(distances[:, -1], return_counts=True)

    index = new_dist[1].argmax()

    prediction = new_dist[0][index]

    # print(new_dist)
    return prediction

####### Initialise Camera  ########


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

face_data = []
labels = []

dataset_path = "./FaceData/"

class_id = 0  # Labels for the file
names = {}  # Mapping b/w id and names

for fx in os.listdir(dataset_path):

    # Loading and opening the data file
    if fx.endswith(".npy"):
        names[class_id] = fx[:-4]
        print("Loaded: "+fx)
        data = np.load(dataset_path+fx)
        face_data.append(data)

        # Create label for the file or data
        target = class_id*np.ones((data.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)

train_dataset = np.concatenate((face_dataset, face_labels), axis=1)
print(train_dataset.shape)

# Testing

while True:
    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for face in faces:
        x, y, w, h = face

        # Getting the Region Of Interest(ROI)
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        # Returns Predicted label
        output = knn(train_dataset, face_section.flatten())

        # Display the name and rectangle around the face
        predictedName = names[int(output)]
        cv2.putText(gray_frame, predictedName, (x, y-10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(gray_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Face", gray_frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
