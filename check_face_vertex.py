#coding:UTF-8
#This code is written by tongshiwen based on the paper
# Learning from Millions of 3D Scans for Large-scale 3D Face Recognition
# based on the 3D point to create depth, azimuth and elevation image.

import math
import numpy as np
import scipy
#import cv2 as cv
def check_face_vertex(vertex, face,):
    vertex = check_size(vertex)
    face = check_size(face)
    return vertex, face

def check_size(a):
    if np.all(a == 0):
        return
    (h, w) = a.shape
    #print ('hw',h,w)
    if h > w:
        a = a.T
    if h< 3 & w== 3:
        a = a.T
    if h <= 3 & w >= 3 & np.sum (np.abs(a[:,2]) ==0):
        a = a.T
    if  h != 3 & h!= 4:
        print ("face or vertex is not of correct size")
    return a