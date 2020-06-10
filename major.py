import cv2
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt

# loading the image - let's take a look at it
image = cv2.imread("main.jpg")
#cv2.imshow('image',image)


hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#cv2.imshow('hsv',hsv)

threshold, binary = cv2.threshold(hsv[:, :, 2], 120, 310, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow('binary',binary)


denoised = filters.median(binary, selem=np.ones((5, 5)))
cv2.imshow('binary',denoised)

denoised = filters.median(binary, selem=np.ones((5, 5)))
cv2.imshow('binary',denoised)

binary=denoised

#make a kernal of 4x4 for mainpulating the image
kernel = np.ones((4, 4), dtype='uint8')
cv2.imshow('binary',kernel)


erosion=cv2.erode(binary,kernel,iterations=1)
cv2.imshow('binary',erosion)

closed_image = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel,iterations=2)
cv2.imshow('binary',closed_image)

dist_transform = cv2.distanceTransform(denoised, cv2.DIST_L1,cv2.DIST_MASK_PRECISE)
cv2.imshow('binary',dist_transform)

template= cv2.imread("template.jpg")
template= cv2.resize(template, (64, 64))
template= cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

_,kernel2 = cv2.threshold(template[:, :, 2], 120, 310, cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)


denoised = filters.median(kernel2, selem=np.ones((5, 5)))
cv2.imshow('binary',denoised)

dist_trans_t = cv2.distanceTransform(denoised, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,)
cv2.imshow('binary',dist_trans_t)

template_matched = cv2.matchTemplate(dist_transform, dist_trans_t, cv2.TM_CCOEFF_NORMED)
cv2.imshow('binary',template_matched)

mn, mx, _, _ = cv2.minMaxLoc(template_matched)
th, p = cv2.threshold(template_matched, 0.10, 0.60, cv2.THRESH_BINARY_INV)

cv2.imshow('binary',p)

# let's go for the peak value in the template matched image..!

x= cv2.convertScaleAbs(p)
cv2.imshow('binary',x)

contours, hierarchy = cv2.findContours(x, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
x= cv2.convertScaleAbs(p)
cv2.imshow('binary',p)

duplicate= cv2.imread("main.jpg")

count = 0

for i in range(len(contours)):
    
    if cv2.contourArea(contours[i]) < 150:
        continue
        
    x, y, w, h = cv2.boundingRect(contours[i])    

    cv2.rectangle(duplicate, (x, y), (x+w, y+h), (140, 185, 205), 2)
    #cv2.drawContours(duplicate, contours, i, (10, 10, 255), 2)
    
    count += 1

print('Number of trees : ', count)

cv2.imshow('binary',duplicate)



cv2.waitKey(0)
cv2.destroyAllWindows()


