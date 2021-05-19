# We will be capturing video stream from our webcam with OpenCV

import cv2

capture = cv2.VideoCapture(0)  # 0 is the default web cam and if we want to use diff cam we have to provide with id

while True:
    retBool,frame = capture.read()   # Frame is just an image and retBool is a boolean values that is returned if the video is not captured properly or webcam is not started then it return retBool as false
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if retBool == False:
        continue

    cv2.imshow("My Video",frame)
    cv2.imshow("My Gray Video",gray_frame)


    # Key Press action to stop or break the loop/process
    key_pressed = cv2.waitKey(1) & 0xFF
    # Basically cv2.waitKey() returns a 32 bit and 0xFF is a 8Bit 11111111 which on and operation gives the ascii value of the keyPressed

    if key_pressed == ord("q"):  # ord is python function that return char's ascii value
        break

capture.release()
cv2.destroyAllWindows()