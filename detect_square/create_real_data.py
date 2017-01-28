import numpy as np
import cv2
import random

for i in range(0,4000):
    img = np.zeros((28,28),np.uint8)
   
    #top_left = (11,11)
    #bottom_right = (20,20)
    top_left = (int(random.random()*26),int(random.random()*26))
    bottom_right = ( int((28 - top_left[0])*random.random()),int((28 - top_left[1])*random.random())) 
    #rotate_deg = random.random()*180
    cv2.rectangle(img,top_left,bottom_right,255)
 


#    rows,cols = img.shape

#    M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_deg,1) 
#    img = cv2.warpAffine(img,M,(cols,rows))

    cv2.imwrite('./data/{}rect{}.png'.format(int(random.random()*10000),i),img)
    #cv2.imshow('img',img)
    #cv2.waitKey(1)
for i in range(0,4000):
    img = (np.random.rand(28,28)*255).astype('uint8')
    cv2.imwrite('./data/{}rand{}.png'.format(int(random.random()*10000),i),img)
