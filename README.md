# DIP
**1)Develop a program to display gray scale image using a read and write operation.

**Description:

import cv2
image=cv2.imread('image2.jpg')
cv2.imshow('original',image)
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('grayscale',gray_image)
cv2.imwrite(‘sample.jpg,gray’)
cv2.waitKey(0)
cv2.destroyAllWindows()

**Output:

**2)Develop a program to perform linear transformation on an image.
     a)Scaling
     b)Rotating

**a) Scaling:

import cv2
File_name='image2.jpg'
try:
    image=cv2.imread(File_name)
    (height,width)=image.shape[:2]
    res=cv2.resize(image,(int(width/2),int(height/2)),interpolation =cv2.INTER_CUBIC)
    cv2.imwrite('result.jpg',res)
    cv2.imshow('image',image)	  
    cv2.imshow('result',res)
        cv2.waitKey(0)
except IOError:
    print('Error while reading file!!!')
    cv2.waitKey(0)
    cv2.destroyAllWindows(0)
    
**Output:    

**b) Rotating:

import cv2
File_name='image2.jpg'
image=cv2.imread(File_name)
(rows,cols)=image.shape[:2]
M=cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
res=cv2.warpAffine(image,M,(cols,rows))
cv2.imshow('image',image)
cv2.imshow('result',res)
cv2.waitKey(0)

**Output:

**3)Develop a program to find the sum of mean of set of images.

import cv2
import os
path="E:\images"
images=[]
dirs=os.listdir(path)
for file in dirs:
    fpath=path+"\\"+file
    images.append(cv2.imread(fpath))
    i=0
for im in images:
    cv2.imshow(dirs[i],images[i])
    i=i+1
print(i) 
cv2.imshow('sum',len(im))
cv2.imshow('mean',len(im)/im)
cv2.waitKey()

**Output:

**4)Develop  a program to convert a color image to gray scale and binary.

import cv2 
originalImage = cv2.imread('d2.jpg')
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Black white image', blackAndWhiteImage)
cv2.imshow('Original image',originalImage)
cv2.imshow('Gray image', grayImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

**Output:

**5)Develop a program to convert given color image to color space.

import cv2
img=cv2.imread('d6.jpg')
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
cv2.imshow('YUV image', yuv_img)
cv2.waitKey()
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV image', hsv_img)
cv2.waitKey()
hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
cv2.imshow('HLS image', hls_img)
cv2.waitKey()
cv2.destroyAllWindows()

**Output:

**6)Develop  a program to create an image from 2D array.

import numpy as np
from PIL import Image
import cv2 as c
array=np.zeros([100,200,3],dtype=np.uint8)
array[:,:100]=[150,128,0]#orange left side
array[:,100:]=[0,0,255]#blue right side
img=Image.fromarray(array)
img.save('img.png')
img.show()
c.waitKey(0)





