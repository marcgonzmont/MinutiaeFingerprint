import cv2
import numpy as np
from myPackage import tools as tl

def process(skeleton, plot= False):
    # harris_corners = np.zeros_like(skeleton.shape, cv2.CV_32FC1)
    harris_corners = cv2.cornerHarris(skeleton, 2, 3, 0.04, cv2.BORDER_DEFAULT)
    harris_normalised = cv2.normalize(harris_corners, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1)
    threshold = 125
    keypoints = []
    rescaled = cv2.convertScaleAbs(harris_normalised)
    harris_c = np.zeros((rescaled.rows, rescaled.cols, cv2.CV_8UC3))
    # vect = []
    for i in range(2):
        vect = np.hstack(rescaled)
    from_to = [ 0.0, 1.1, 2.2 ]
    cv2.mixChannels(vect, harris_c, from_to)
    for x in range(harris_normalised.cols):
        for y in range(harris_normalised.rows):
            if int(harris_normalised[y, x]) > threshold:
                cv2.circle(harris_c, (x, y), 5, (0, 255, 0), 1)
                cv2.circle(harris_c, (x, y), 1, (0, 0, 255), 1)
                keypoints.append(cv2.KeyPoint(x, y, 1))
    if plot:
        titles = ["skeleton", "keypoints"]
        images = [skeleton, harris_c]
        title = "Key Points"
        tl.plotImages(titles, images, title, 1, 2)