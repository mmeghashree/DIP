# DIP
**1)Develop a program to display gray scale image using a read and write operation.

**Description:
Grayscale:Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.
Importance of grayscaling :
*Dimension reduction: For e.g. In RGB images there are three color channels and has three dimensions while grayscaled images are single dimensional.
Reduces model complexity: Consider training neural article on RGB images of 10x10x3 pixel.The input layer will have 300 input nodes. On the other hand, the same neural network will need only 100 input node for grayscaled images.
For other algorithms to work: There are many algorithms that are customized to work only on grayscaled images e.g. Canny edge detection function pre-implemented in OpenCV library works on Grayscaled images only.

**Program:

import cv2
image=cv2.imread('image2.jpg')
cv2.imshow('original',image)
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('grayscale',gray_image)
cv2.imwrite(‘sample.jpg,gray’)
cv2.waitKey(0)
cv2.destroyAllWindows()

**cv2.resize()-> method refers to the scaling of images. Scaling comes handy in many image processing as well as machine learning applications. It helps in reducing the number of pixels from an image 
cv2.cvtColor()-> method is used to convert an image from one color space to another. 
np.hstack()->function is used to stack the sequence of input arrays horizontally (i.e. column wise) to make a single array.
np.concatenate->Concatenation refers to joining. This function is used to join two or more arrays of the same shape along a specified axis.
cv2.imwrite()->method is used to save an image to any storage device. This will save the image according to the specified format in current working directory.

**Output:

![image](https://user-images.githubusercontent.com/72377303/104425653-ca43ca00-5535-11eb-9cc2-777c38e412ef.png)

**2)Develop a program to perform linear transformation on an image.
     a)Scaling
     b)Rotating

**Discription:
Scaling:Image resizing refers to the scaling of images. Scaling comes handy in many image processing as well as machine learning applications. It helps in reducing the number of pixels from an image 
Rotation:Image rotation is a common image processing routine used to rotate images at any desired angle. This helps in image reversal, flipping, and obtaining an intended view of the image. Image rotation has applications in matching, alignment, and other image-based algorithms. OpenCV is a well-known library used for image processing.

**Program:
a) Scaling:

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
    
**cv2.resize()->method refers to the scaling of images. Scaling comes handy in many image processing as well as machine learning applications. It helps in reducing the number of pixels from an image 
imshow()->function in pyplot module of matplotlib library is used to display data as an image
    
**Output: 

![image](https://user-images.githubusercontent.com/72377303/104425905-13941980-5536-11eb-8bb5-f5acc2883fe7.png)

**Progarm:
b) Rotating:

import cv2
File_name='image2.jpg'
image=cv2.imread(File_name)
(rows,cols)=image.shape[:2]
M=cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
res=cv2.warpAffine(image,M,(cols,rows))
cv2.imshow('image',image)
cv2.imshow('result',res)
cv2.waitKey(0)

**cv2.getRotationMatrix2D Perform the counter clockwise rotation
warpAffine() function is the size of the output image, which should be in the form of (width, height). Remember width = number of columns, and height = number of rows.

**Output:

![image](https://user-images.githubusercontent.com/72377303/104426122-5fdf5980-5536-11eb-8254-26b81202d6f8.png)

**3. Develop a program to find sum and mean of a set of images.
Create n number of images and read the directory and perform operation.

**Discription:
Sum:You can add two images with the OpenCV function, cv. add(), or simply by the numpy operation res = img1 + img2.
Mean:The function mean calculates the mean value M of array elements, independently for each channel, and return it:" This mean it should return you a scalar for each layer of you image

**Program:

import cv2
import os
path = 'E:\photos'
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

**The append() method in python adds a single item to the existing list.
listdir() method in python is used to get the list of all files and directories in the specified directory.

**Output:

![image](https://user-images.githubusercontent.com/72377303/104428061-d8dfb080-5538-11eb-94a8-42d0231a7cc3.png)

**4)Develop  a program to convert a color image to gray scale and binary.

**Discription:
Grayscale image:Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.
Binary image:A binary image is a monochromatic image that consists of pixels that can have one of exactly two colors, usually black and white.

import cv2 
originalImage = cv2.imread('d2.jpg')
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Black white image', blackAndWhiteImage)
cv2.imshow('Original image',originalImage)
cv2.imshow('Gray image', grayImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

**v2.threshold->works as, if pixel value is greater than a threshold value, it is assigned one value (may be white), else it is assigned another value (may be black). 
destroyAllWindows()->simply destroys all the windows we created. To destroy any specific window, use the function cv2. destroyWindow() where you pass the exact window name.

**Output:

![image](https://user-images.githubusercontent.com/72377303/104425173-3245e080-5535-11eb-94e9-936a3b05562e.png)

**5)Develop a program to convert given color image to color space.

**Discription:
Color space:Color spaces are a way to represent the color channels present in the image that gives the image that particular hue
BGR color space: OpenCV’s default color space is RGB. 
HSV color space: It stores color information in a cylindrical representation of RGB color points. It attempts to depict the colors as perceived by the human eye. Hue value varies from 0-179, Saturation value varies from 0-255 and Value value varies from 0-255. 
HLS color space:The HSL color space, also called HLS or HSI, stands for:Hue : the color type Ranges from 0 to 360° in most applications 
YUV color space:Y refers to the luminance or intensity, and U/V channels represent color information. This works well in many applications because the human visual system perceives intensity information very differently from color information.

**Program:

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

**cv2.cvtColor()->method is used to convert an image from one color space to another. 

**Output:

![image](https://user-images.githubusercontent.com/72377303/104425012-fd398e00-5534-11eb-8c40-0ba6cc50b18c.png)

**6)Develop  a program to create an image from 2D array.

**Discription:
Two dimensional array:2D array can be defined as an array of arrays. The 2D array is organized as matrices which can be represented as the collection of rows and columns. However, 2D arrays are created to implement a relational database look alike data structure.

**Program:

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

**numpy.zeros()->function returns a new array of given shape and type, with zeros.
Image.fromarray(array)->this is used to create image object of above array.
unit8->is an unsigned 8-bit integer that can represent values 0-255.

**Output:

![image](https://user-images.githubusercontent.com/72377303/104424762-a5028c00-5534-11eb-884f-0896bf097b2d.png)

**7)Program to find the of neighbourhood values of matrix.

**Discription:
An array of (i,j) where i indicates row and j indicates column.
For every given cell index (i,j),findind sum of all matrix elements except the elements present in the i'th row and/or j'th column.
**Program:

import numpy as np
M = [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]] 
M = np.asarray(M)
N = np.zeros(M.shape)
def sumNeighbors(M,x,y):
    l = []
    for i in range(max(0,x-1),x+2): # max(0,x-1), such that no negative values in range() 
        for j in range(max(0,y-1),y+2):
            try:
                t = M[i][j]
                l.append(t)
            except IndexError: # if entry doesn't exist
                pass
    return sum(l)-M[x][y] # exclude the entry itself
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        N[i][j] = sumNeighbors(M, i, j)
print ("Original matrix:\n", M)
print ("Summed neighbors matrix:\n", N)

**Output:

![image](https://user-images.githubusercontent.com/72377303/104438861-56a9b900-5545-11eb-95a0-9a35933af0e4.png)

**8)Program for operator overloading.

**Program:

#include <iostream>
using namespace std;
class matrix
{
 int r1, c1, i, j, a1;
 int a[10][10];
public:int get()
 {
  cout << "Enter the row and column size for the  matrix\n";
  cin >> r1;
  cin >> c1;
   cout << "Enter the elements of the matrix\n";
  for (i = 0; i < r1; i++)
  {
   for (j = 0; j < c1; j++)
   {
    cin>>a[i][j];
   }
  }
 };
 void operator+(matrix a1)
 {
 int c[i][j];
   for (i = 0; i < r1; i++)
   {
    for (j = 0; j < c1; j++)
    {
     c[i][j] = a[i][j] + a1.a[i][j];
    }
  }
  cout<<"addition is\n";
  for(i=0;i<r1;i++)
  {
   cout<<" ";
   for (j = 0; j < c1; j++)
   {
    cout<<c[i][j]<<"\t";
   }
   cout<<"\n";
  }
 };
  void operator-(matrix a2)
 {
 int c[i][j];
   for (i = 0; i < r1; i++)
   {
    for (j = 0; j < c1; j++)
    {
     c[i][j] = a[i][j] - a2.a[i][j];
    }   
  }
  cout<<"subtraction is\n";
  for(i=0;i<r1;i++)
  {
   cout<<" ";
   for (j = 0; j < c1; j++)
   {
    cout<<c[i][j]<<"\t";
   }
   cout<<"\n";
  }
 };
 void operator*(matrix a3)
 {
  int c[i][j];
  for (i = 0; i < r1; i++)
  {
   for (j = 0; j < c1; j++)
   {
    c[i][j] =0;
    for (int k = 0; k < r1; k++)
    {
     c[i][j] += a[i][k] * (a3.a[k][j]);
    }
  }
  }
  cout << "multiplication is\n";
  for (i = 0; i < r1; i++)
  {
   cout << " ";
   for (j = 0; j < c1; j++)
   {
    cout << c[i][j] << "\t";
   }
   cout << "\n";
  }
 };
};
int main()
{
 matrix p,q;
 p.get();
 q.get();
 p + q;
 p - q;
 p * q;
return 0;
}

**Output:
Enter the row and column size for the  matrix
2
2
Enter the elements of the matrix
6
7
5
8
Enter the row and column size for the  matrix
2
2
Enter the elements of the matrix
2
3
1
4
addition is
 8      10
 6      12
subtraction is
 4      4
 4      4
multiplication is
 19     46
 18     47


