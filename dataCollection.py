import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0

# Ensure the folder exists
if not os.path.exists(folder):
    os.makedirs(folder)

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image from webcam. Exiting...")
            break

        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Calculate valid cropping coordinates
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]
            imgCropShape = imgCrop.shape

            # Ensure imgCrop is not empty
            if imgCrop.size > 0:
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = min(math.ceil(k * w), imgSize)  # Ensure wCal <= imgSize
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = (imgSize - wCal) // 2
                    imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
                else:
                    k = imgSize / w
                    hCal = min(math.ceil(k * h), imgSize)  # Ensure hCal <= imgSize
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = (imgSize - hCal) // 2
                    imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)
            else:
                print("Empty imgCrop, skipping this frame.")

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            save_path = f'{folder}/Image_{time.time()}.jpg'
            cv2.imwrite(save_path, imgWhite)
            print(f"Saved {save_path}, Count: {counter}")

except KeyboardInterrupt:
    print("\nProgram interrupted. Exiting...")

finally:
    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()
