#coding:UTF-8
'''
#This code is written by tongshiwen based on the paper
#Learning from Millions of 3D Scans for Large-scale 3D Face Recognition
#based on the 3D point to create depth, azimuth and elevation image.
'''
import numpy as np
#from mayavi import mlab
import numpy as np
import cv2 as cv
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import check_face_vertex
np.set_printoptions(threshold=np.inf)
scale = 0.4
p = 0
i = 2
#read 3D pointcloud data
pointcloud = [ ]
def cart2sph(x, y, z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r
def rescale(x, a, b):
    m = np.nanmin(x)
    M = np.nanmax(x)
    y = (b - a) * (x - m) / (M - m) + a
    #y = 255 * (x - m) / (M - m
    #print ('M',M)
    return y

def crossproduct(x, y):
    z = x
    z[0, :] = x[1, :] * y[2, :] - x[2, :] * y[1, :]
    z[1, :] = x[2, :] * y[0, :] - x[0, :] * y[2, :]
    z[2, :] = x[0, :] * y[1, :] - x[1, :] * y[0, :]
    return z

def find_normals(im,tri):
    print ('im', im.shape)
    print ('tri',tri.shape)
    #print ('rei',tri)
    #exit()
    im, tri =  check_face_vertex.check_face_vertex(im, tri)
    nface = tri.shape[1]
    nvert = im.shape[1]
    normalf = crossproduct(im[:, tri[1,:]] - im[:,tri[0,:]], im[:,tri[2,:]] - im[:,tri[0,:] ])
    d = np.array(np.sqrt(np.sum(normalf** 2, axis=0)))
    d = np.where(d < np.finfo(np.float64).eps, 1, d)
    normalf = normalf / np.tile(d, (3, 1))
    normal = np.zeros((3, nvert))
    for i in range (0, nface):
        f = []
        f.append(tri[:, i].tolist())
        for j in range (0,3):
            normal[:, f[0][j]] = normal[:, f[0][j]] + normalf[:, i]
    d = np.array(np.sqrt(np.sum(normal** 2, axis=0)))
    d = np.where(d < np.finfo(np.float64).eps, 1, d)###
    normal = normal / np.tile(d, (3, 1))
    v = im - np.tile(im.mean(axis=0), (3, 1))
    s = np.sum(v* normal,axis=1)
    if np.sum(s > 0) < np.sum(s < 0):
        normal = -normal
        normalf = -normalf

    return normal, normalf

with open("test.wrl","r") as f:
    lines = f.readlines()
    for idx, elem in enumerate(lines):
        if  'point' in lines[idx] and '[' in lines[idx+1]:
            j= idx
            print (j)
            while True:
                data2 = lines[17 + i]
                if ']' in data2:
                    break
                listLine = data2[0:data2.rfind(',', 1)].split(' ')
                listLine = [float(e) for e in listLine]
                pointcloud.append(listLine)
                i=i+2
pointcloud = np.array(pointcloud)
def  ptc2dae(im, tri, scale):
    xmin = np.min(im[:,0])
    ymin = np.min(im[:,1])
    xmax = np.max(im[:,0])
    ymax = np.max(im[:,1])
    xscale = scale
    yscale = scale
    point1 = np.arange(xmin, xmax, xscale)
    point2 = np.arange(ymax, ymin, -yscale)
    [X1, Y1] = np.meshgrid(point1, point2)
    lon_lat = np.c_[im[:,0].ravel(), im[:,1].ravel()]
    Zd = griddata(lon_lat, im[:, 2].ravel(), (X1, Y1))
    Norm, normf = find_normals(im.T, tri.T)
    Norm = Norm.T
    print ('normal', Norm.shape)
    Theta, Phi, r = cart2sph(Norm[:,0], Norm[:,1], Norm[:,2])
    if np.sum(Phi) < 0:
        Phi = -Phi
    Phi = griddata(lon_lat, Phi.ravel(), (X1, Y1))
    Theta = griddata(lon_lat, np.abs(Theta).ravel(), (X1, Y1))
    Zd = np.array(rescale(Zd, 0, 255),dtype='uint8')
    Phi = np.array(rescale(Phi, 0, 255),dtype='uint8')
    Theta = np.array(rescale(Theta, 0, 255),dtype='uint8')
    zero_index = np.where(Zd == 0)
    Phi[zero_index] = 0
    Theta[zero_index] = 0
    cv.imshow('con_img1', Theta)
    cv.waitKey(0)
    I =  np.array(I, dtype = 'uint8')
    #resize the image
    [H, W, C] = np.shape(I)
    if H >= W:
        I = np.resize(I, [512, NaN])
    else:
        I = np.resize(I, [NaN, 512])
    [H, W, C] = np.shape(I)

    temp = min(H, W)
    up = np.floor((512 - temp) / 2)
    dowm = round((512 - temp) / 2)
    (h, w) = I.shape
    if H > W:
        I1 = np.zeros([h, w+up+down], dtype = float, order = 'C')
        Ia = np.zeros([h, down], dtype = float, order = 'C')
        Ib = np.zeros([h, up], dtype = float, order = 'C')
        I1[:,0:down-1] = Ia, I1[:,down:w-1] = I, I1[:,w:up-1] = Ib
    else:
        I1 = np.zeros([h+ up + down, w], dtype=float, order='C')
        Ic = np.zeros([down, w], dtype=float, order='C')
        Id = np.zeros([up, w], dtype=float, order='C')
        I1[0:down - 1:,] = Ic, I1[down:h-1:,] = I, I1[ h:up - 1:,] = Id,
    depth = I1[:,:, 1]
    return depth, I1

if __name__ == "__main__":
    scale = 0.4
    im = pointcloud
    points =  np.zeros((31844, 2)) #31844
    points[:, 1] = im[:,0]
    points[:, 0] = im[:,1]
    hull = ConvexHull(points)
    N = 2 # The dimensions of our points
    options = 'Qt Qbb Qc' if N <= 3 else 'Qt Qbb Qc Qx' # Set the QHull options
    tri = scipy.spatial.Delaunay(points,qhull_options = options).simplices
    keep = np.ones(len(tri),dtype = bool)
    for i,t in enumerate(tri):
        if abs(np.linalg.det(np.hstack((points[t],np.ones([1,N+1]).T)))) < 1E-15:
            keep[i] = False # Point is coplanar,we don't want to keep it
    tri = tri[keep]
    ptc2dae(im, indices, scale)

