import cv2
import Hand_Tracking_Module_ as htm
import time
import math
import numpy as np

cap = cv2.VideoCapture(0)

cTime = 0
pTime = 0

# Hand Detector
detector = htm.handDetector(detectionCon=0.8)

# Find Function
# x is the raw distance y is the value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

coff = np.polyfit(x, y, 2)    # y = Ax^2 + BX + C


while True:
    Success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPositon(img)

    if lmList:
        # print(lmList)

        # find the distance between landmark no 5 and 17
        _1, x1, y1 = lmList[5]
        _2, x2, y2 = lmList[17]
        _3, x3, y3 = lmList[20]

        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        A, B, C = coff
        distanceCM = int(A*distance**2 + B*distance + C)

        x, y = x3-20, y3-40
        cv2.rectangle(img, (x, y), (x+170, y+250), (0, 255, 255), 2)
        cv2.putText(img, f"{distanceCM} CM", (x, y-20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

        # print(_3, x3, y3)
        # print(distanceCM, distance)
        # print(abs(x2 - x1), distance)

    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS:{int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == 80 or key == 113:
        break

cap.release()
cv2.destroyAllWindows()

print("Code Completed!")