import os
import cv2
Names = [['./training-images','train'], ['./test-images','test']]

for name in Names:
    FileList = []
    for dirname in os.listdir(name[0])[1:]: # [1:] Excludes .DS_Store from Mac OS
        path = os.path.join(name[0],dirname)
        for filename in os.listdir(path):
            if filename.endswith(".png"):
                print(filename)
                gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                gray = cv2.resize(gray, (28, 28))
                cv2.imwrite(filename, gray)