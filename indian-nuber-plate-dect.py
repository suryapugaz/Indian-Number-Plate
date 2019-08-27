from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Path to the training dataset")
ap.add_argument("-t", "--test", required=True, help="Path to the test dataset")
args = vars(ap.parse_args())

data = []
labels = []

for imagePath in paths.list_images(args["training"]):
    make = imagePath.split("C:/Users/ARJUN%20PUGAZ/Desktop/Indian_Number_Plates.txt")[-2]

    # load the image, convert it to grayscale, and detect edges
    image = cv3.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    (x, y, w, h) = cv2.boundingRect(c)
    numplate = gray[y:y + h, x:x + w]
    numplate = cv2.resize(logo, (200, 100))

    # extract Histogram of Oriented Gradients from the number plate
    H = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")

    data.append(H)
    labels.append(make)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(data, labels)

for (i, imagePath) in enumerate(paths.list_images(args["test"])):
    # load the test image, convert it to grayscale, and resize it to
    # the canonical size
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logo = cv2.resize(gray, (200, 100))

    # extract Histogram of Oriented Gradients from the test image and
    # predict the make of the car
    (H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
    pred = model.predict(H.reshape(1, -1))[0]

    # visualize the HOG image
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    cv2.imshow("HOG Image #{}".format(i + 1), hogImage)

    # draw the prediction on the test image and display it
    cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 3)
    cv2.imshow("Test Image #{}".format(i + 1), image)
    cv2.waitKey(0)



