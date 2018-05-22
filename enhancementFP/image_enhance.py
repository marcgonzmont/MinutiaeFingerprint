# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 22:50:30 2016

@author: utkarsh
"""
from enhancementFP.ridge_segment import ridge_segment
from enhancementFP.ridge_orient import ridge_orient
from enhancementFP.ridge_freq import ridge_freq
from enhancementFP.ridge_filter import ridge_filter
from myPackage import tools as tl
import cv2
from os.path import exists, altsep

def image_enhance(gray, name, plot= False, path= None):
    print("Enhancing ridges...")
    blksze = 16
    thresh = 0.1
    normim,mask = ridge_segment(gray,blksze,thresh)           # normalise the image and find a ROI


    gradientsigma = 1
    blocksigma = 7
    orientsmoothsigma = 7
    orientim = ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma)              # find orientation of every pixel


    blksze = 38
    windsze = 5
    minWaveLength = 5
    maxWaveLength = 15
    freq,medfreq = ridge_freq(normim, mask, orientim, blksze, windsze, minWaveLength,maxWaveLength)   #find the overall frequency of ridges
    
    
    freq = medfreq*mask
    kx = 0.65;ky = 0.65
    newim = ridge_filter(normim, orientim, freq, kx, ky)      # create gabor filter and do the actual filtering
    
    img_enhanced = (newim < -3).astype(float)

    if path is not None:
        new_path = altsep.join((path, "Enhanced"))
        if not exists(new_path):
            tl.makeDir(new_path)
        dst = altsep.join((new_path, (name + ".png")))
        img_color = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(dst, img_color)

    if plot:
        cv2.imshow("Enhanced '{}'".format(name), img_enhanced)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    return img_enhanced