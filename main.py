import cv2
import os

haar_classifier = cv2.CascadeClassifier("haarcascade_stop_sign.xml")

testDirectoryName = "test_images"

for image in os.listdir(testDirectoryName):

    img = cv2.imread(f"{testDirectoryName}/{image}")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    founded_objects = haar_classifier.detectMultiScale(img_gray, minSize =(80, 80))

    amount_found = len(founded_objects)

    if amount_found != 0:
        
        for (x, y, width, height) in founded_objects:
            cv2.rectangle(img, (x, y), (x + height, y + width), (0, 255, 0), 5)

    cv2.imshow(image, img)

while True:
    k=cv2.waitKey(1) & 0xFF
    if k==27:
        break

cv2.destroyAllWindows()