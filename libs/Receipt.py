#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import imutils
import random
import pytesseract


# In[3]:


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


# In[4]:

def get_edged_image(image):
    edges = cv2.Canny(image, 100, 200)
    return edges


def denoise_filter2D(image, kernel=None):
    if kernel is None:
        kernel = np.ones((2,2), np.float32)/4    
        
    denoised_image = cv2.filter2D(image, -1, kernel)
    
    return denoised_image


# In[5]:


def standard_thresholding(image):
    
    ret3,threshold_image = cv2.threshold(image,160,255,cv2.THRESH_BINARY)
    
    return threshold_image


# In[6]:


def get_receipt_contour(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h_perimeter = 0
    h_contour = []
    for cnt in contours:
        cnt_perimeter = cv2.arcLength(cnt,True)
        if cnt_perimeter > h_perimeter:
            h_perimeter = cnt_perimeter
            h_contour = cnt
            
    return np.array(h_contour).astype(np.float32)


# In[7]:


def adjust_receipt_rotation(receipt_contour, img):   
    rect = cv2.minAreaRect(receipt_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    angle = rect[-1]

    if angle < -45:
        angle =  90 + angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# In[8]:


#def prepare_image(gray_scaled_image):
#    no_noise_image = denoise_filter2D(gray_scaled_image)
#
#    threshold_image = standard_thresholding(no_noise_image)
#
#    no_noise_image = denoise_filter2D(threshold_image)
    
#    edges = cv2.Canny(no_noise_image,100,200)
    
#    return no_noise_image, edges


# In[9]:


def get_cropped_image(image, contour):
    x,y,w,h = cv2.boundingRect(contour)
    cropped = image[y:y+h, x:x+w]
    return cropped


# In[10]:


#IMAGEPATH = "/home/simone/Desktop/receipt_test.jpg"

#image = cv2.imread(IMAGEPATH)
#gray_scaled_image = cv2.imread(IMAGEPATH, cv2.IMREAD_GRAYSCALE)

#resize_param = 500

#image = image_resize(image,resize_param)
#gray_scaled_image = image_resize(gray_scaled_image,resize_param)


# In[11]:


#useful_image = prepare_image(gray_scaled_image)


# In[ ]:





# In[12]:


#receipt_contour = get_receipt_contour(edges)
        
#rotated = adjust_receipt_rotation(receipt_contour, useful_image)
   
#_ = cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# In[13]:



#receipt_contour = get_receipt_contour(rotated)


# In[ ]:





# In[14]:



#cv2.imshow("gray image", gray_image_scaled)
#cv2.imshow("starting image", image)
#cv2.imshow("threshold image", threshold_image)
#cv2.imshow("no noise image", useful_image)
#cv2.imshow("edges", edges)

#cv2.imshow("rotated image", rotated)
#cv2.imshow("cropped image", cropped)







#cv2.waitKey()
#cv2.destroyAllWindows()

