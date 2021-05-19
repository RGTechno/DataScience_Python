# Program to read and display images from OpenCV

import cv2

img = cv2.imread("../images/tbbt.jpg")
gray_img = cv2.imread("../images/tbbt.jpg",cv2.IMREAD_GRAYSCALE)

cv2.imshow("The Big bang Theory",img)
cv2.imshow("The Big bang Theory grey",gray_img)

cv2.waitKey(0)  # Holds the window for infinite time until closed
cv2.destroyAllWindows()
