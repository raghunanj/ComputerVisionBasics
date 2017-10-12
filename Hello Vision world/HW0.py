import numpy as np
import cv2

#Load an image 
img = cv2.imread('MTC.jpg', -1)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

scalar = input('Enter a scalar value(within 0-255) : ')
x=np.uint8([scalar])
#x = [25];
dst = np.add(img,x)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

dst_sub = np.subtract(img,x)
cv2.imshow('dst_sub',dst_sub)
cv2.waitKey(0)
cv2.destroyAllWindows()

dst_mul = np.multiply(img,x)
cv2.imshow('dst_mul',dst_mul)
cv2.waitKey(0)
cv2.destroyAllWindows()

dst_div = np.divide(img,x)
cv2.imshow('dst_div',dst_div)
cv2.waitKey(0)
cv2.destroyAllWindows()

print ("resize image")

small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
cv2.imshow('small',small)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
