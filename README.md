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

![image](https://user-images.githubusercontent.com/72377303/104425653-ca43ca00-5535-11eb-9cc2-777c38e412ef.png)

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

![image](https://user-images.githubusercontent.com/72377303/104425905-13941980-5536-11eb-8bb5-f5acc2883fe7.png)

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

![image](https://user-images.githubusercontent.com/72377303/104426122-5fdf5980-5536-11eb-8254-26b81202d6f8.png)

**3. Develop a program to find sum and mean of a set of images.
Create n number of images and read the directory and perform operation.

import cv2
import os
path = 'C:\Pictures'
imgs = []

files = os.listdir(path)
for file in files:
    filepath=path+"\\"+file
    imgs.append(cv2.imread(filepath))
i=0
im = []
for im in imgs:
    #cv2.imshow(files[i],imgs[i])
    im+=imgs[i]
    i=i+1
cv2.imshow("sum of four pictures",im)
meanImg = im/len(files)
cv2.imshow("mean of four pictures",meanImg)
cv2.waitKey(0)

**Output:

![image](https://user-images.githubusercontent.com/72377303/104428061-d8dfb080-5538-11eb-94a8-42d0231a7cc3.png)

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

![image](https://user-images.githubusercontent.com/72377303/104425173-3245e080-5535-11eb-94e9-936a3b05562e.png)

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

![image](https://user-images.githubusercontent.com/72377303/104425012-fd398e00-5534-11eb-8c40-0ba6cc50b18c.png)

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

**Output:

![image](https://user-images.githubusercontent.com/72377303/104424762-a5028c00-5534-11eb-884f-0896bf097b2d.png)





