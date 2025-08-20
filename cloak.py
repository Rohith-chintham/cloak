import cv2
import numpy as np
import time

# Start the webcam
cap = cv2.VideoCapture(0)
time.sleep(3)  # allow camera to adjust

# Capture the background (without you in it)
for i in range(30):
    ret, background = cap.read()
background = np.flip(background, axis=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally (mirror effect)
    frame = np.flip(frame, axis=1)

    # Convert color from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the red cloak color range
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    red_mask = mask1 + mask2

    # Refine mask (remove noise)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    red_mask = cv2.dilate(red_mask, np.ones((3, 3), np.uint8))

    # Invert mask (everything except cloak)
    inverse_mask = cv2.bitwise_not(red_mask)

    # Segment out non-red parts
    res1 = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Segment cloak part from background
    res2 = cv2.bitwise_and(background, background, mask=red_mask)

    # Final output (combine both)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow("Harry Potter Cloak", final_output)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
