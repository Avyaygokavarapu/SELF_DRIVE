#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:28:32 2020

@author: avyay
"""

import numpy as np
import cv2 as cv 
lines=[]

def colorconvert(frame):
    hls=cv.cvtColor(frame,cv.COLOR_RGB2HLS)
    lower=np.array([0,120,0])
    upper=np.array([255,255,255])
    yellower = np.array([0,0,0])
    yelupper = np.array([70,255,255])
    yellowmask = cv.inRange(hls, yellower, yelupper)
    whitemask = cv.inRange(hls, lower, upper)
    mask = cv.bitwise_or(yellowmask, whitemask)
    masked = cv.bitwise_and(frame, frame, mask = mask)
    return masked

def roi(img):
    x = int(img.shape[1])
    y = int(img.shape[0])
    shape = np.array([[int(0), int(y)], [int(x), int(y)], [int(0.55*x), int(0.60*y)], [int(0.45*x), int(0.60*y)]])
    #print(shape)
    #define a numpy array with the dimensions of img, but comprised of zeros
    
    mask = np.zeros_like(img)
    
    #print(mask)
    #Uses 3 channels or 1 channel for color depending on input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #creates a polygon with the mask color
    cv.fillPoly(mask, np.int32([shape]), ignore_mask_color)
    #returns the image only where the mask pixels are not zero
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

rightSlope, leftSlope, rightIntercept, leftIntercept = [],[],[],[]
def draw_lines(img, lines, thickness=5):
    global rightSlope, leftSlope, rightIntercept, leftIntercept
    rightColor=[0,255,0] 
    leftColor=[255,0,0]
    for x in lines:
        print(x)
        
    print("over")
    #this is used to filter out the outlying lines that can affect the average
    #We then use the slope we determined to find the y-intercept of the filtered lines by solving for b in y=mx+b
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y1-y2)/(x1-x2)
            if slope > 0.3:
                if x1 > 500 :
                    yintercept = y2 - (slope*x2)
                    rightSlope.append(slope)
                    rightIntercept.append(yintercept)
                else: None
            elif slope < -0.3:
                if x1 < 600:
                    yintercept = y2 - (slope*x2)
                    leftSlope.append(slope)
                    leftIntercept.append(yintercept)
    #We use slicing operators and np.mean() to find the averages of the 30 previous frames
    #This makes the lines more stable, and less likely to shift rapidly
    leftavgSlope = np.mean(leftSlope[-30:])
    leftavgIntercept = np.mean(leftIntercept[-30:])
    rightavgSlope = np.mean(rightSlope[-30:])
    rightavgIntercept = np.mean(rightIntercept[-30:])
    #Here we plot the lines and the shape of the lane using the average slope and intercepts
    try:
        left_line_x1 = int((0.65*img.shape[0] - leftavgIntercept)/leftavgSlope)
        left_line_x2 = int((img.shape[0] - leftavgIntercept)/leftavgSlope)
        right_line_x1 = int((0.65*img.shape[0] - rightavgIntercept)/rightavgSlope)
        right_line_x2 = int((img.shape[0] - rightavgIntercept)/rightavgSlope)
        pts = np.array([[left_line_x1, int(0.65*img.shape[0])],[left_line_x2, int(img.shape[0])],[right_line_x2, int(img.shape[0])],[right_line_x1, int(0.65*img.shape[0])]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv.fillPoly(img,[pts],(0,0,255))
        cv.line(img, (left_line_x1, int(0.65*img.shape[0])), (left_line_x2, int(img.shape[0])), leftColor, 10)
        cv.line(img, (right_line_x1, int(0.65*img.shape[0])), (right_line_x2, int(img.shape[0])), rightColor, 10)
    except ValueError:
    #I keep getting errors for some reason, so I put this here. Idk if the error still persists.
        pass
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    """
    lines = cv.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
def linedetect(img):
    return hough_lines(img, 1, np.pi/180, 10, 20, 100)
#hough_img = list(map(linedetect, canny_img))
#display_images(hough_img)

#def weightSum(input_set):
 #   img = list(input_set)
  #  return cv.addWeighted(img[0], 1, img[1], 0.8, 0)
#result_img = list(map(weightSum, zip(hough_img, imageList)))


    
vid=cv.VideoCapture("/home/avyay/test_video.mp4")
while(True):
     
     ret, frame = vid.read()
     cv.imshow("Original",frame)
     orig=frame
     frame=colorconvert(frame)
     
     #cv.imshow("After Color Conv",frame)
     
     frame=roi(frame)
     
     #cv.imshow("After ROI",frame)
     
     frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
     frame=cv.Canny(frame,50,120)
     
     #cv.imshow("Post Canny",frame)
     
     
     
     frame=linedetect(frame)
     #cv.imshow("Final Output",frame)
     
     final=cv.addWeighted(frame,1,orig,0.8,0)
     cv.imshow("final",final)
     
     
     
     if cv.waitKey(10) & 0xFF == ord('q'):
         break
     
vid.release()
cv.destroyAllWindows()
