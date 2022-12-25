import cv2
import numpy as np

rootDir = "D:\\SYS_File\\Desktop\\Python\\image\\low\\"
filename = '175139.jpg'

image = cv2.imread(rootDir + filename)
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
line_vertical = []
line_horizontal = []

ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((10, 10), np.uint8)
# 3图像的开闭运算
cv0pen = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)  # 开运算
cv2.imwrite("test1.jpg", th2)

edges = cv2.Canny(cv0pen, 50, 150, apertureSize=3)
cv2.imwrite("test2.jpg", edges)