import cv2
import numpy as np


kernel = np.ones((5, 5), np.uint8)
im = cv2.imread('/home/robesafe/Oscar/Imagenes_20200226/Imagen_carretera_nueva.jpg')
mask = cv2.inRange(im, (0, 200, 0), (130, 256, 90)) #Quitar reflejos
dilation = cv2.dilate(mask, kernel, iterations=1)
img = cv2.erode(dilation, kernel, iterations=1)




size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

ret, img = cv2.threshold(img, 127, 255, 0)

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
done = False

while (not done):
    eroded = cv2.erode(img, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(img, temp)
    skel = cv2.bitwise_or(skel, temp)
    img = eroded.copy()

    zeros = size - cv2.countNonZero(img)
    if zeros == size:
        done = True




cv2.imshow("skel", skel)
cv2.waitKey(0)
cv2.destroyAllWindows()