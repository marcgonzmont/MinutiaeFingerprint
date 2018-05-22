import cv2
from cv2.ximgproc import thinning
import numpy as np
from myPackage import tools as tl
from os.path import join, basename, splitext, altsep, exists
from numba import jit

@jit
def cleanImage(bin_img, name, plot= False, path= None):
    print("Cleaning image...")
    n = 3
    n_2 = 3
    iter = 2 #3
    iter_2 = 2
    kernel = (n, n)
    kernel2 = (n_2, n_2)
    cleaned = bin_img.copy()

    # cleaned = cv2.dilate(cleaned, kernel= kernel2, iterations= iter_2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel= kernel, iterations= iter)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel= kernel, iterations= iter)
    cleaned = cv2.erode(cleaned, kernel= kernel2, iterations= iter_2)

    if path is not None:
        new_path = altsep.join((path, "Cleaned"))
        if not exists(new_path):
            tl.makeDir(new_path)
        dst = altsep.join((new_path, (name + ".png")))
        cv2.imwrite(dst, cleaned)

    if plot:
        cv2.imshow("Cleaned '{}'".format(name), cleaned)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    return cleaned

@jit
def blurrImage(nameImage, name, plot= False, path= None):
    print("Blurring image...")
    image = cv2.imread(nameImage)
    if image.shape[2] == 3:
        # print("COLOR")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    kernel = (7, 7)
    cleaned = cv2.GaussianBlur(gray, kernel, 0)

    if path is not None:
        new_path = altsep.join((path, "Blurred"))
        if not exists(new_path):
            tl.makeDir(new_path)
        dst = altsep.join((new_path, (name + ".png")))
        img_color = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(dst, img_color)

    if plot:
        cv2.imshow("Original '{}'".format(name), image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

        cv2.imshow("Cleaned '{}'".format(name), cleaned)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    return cleaned

@jit
def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

@jit
def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

@jit
def zhangSuen(image, name, plot= False, path= None):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1
                    P2 * P4 * P6 == 0  and    # Condition 3
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))
        for x, y in changing2:
            Image_Thinned[x][y] = 0

    if path is not None:
        new_path = altsep.join((path, "ZhangSuen"))
        if not exists(new_path):
            tl.makeDir(new_path)
        dst = altsep.join((new_path, (name + ".png")))
        cv2.imwrite(dst, Image_Thinned)

    if plot:
        cv2.imshow("Thinning '{}'".format(name), Image_Thinned)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    return Image_Thinned

@jit
def thinImage(image, name, plot= False, path= None):
    print("Thinning image...")
    image *= 255
    image8 = np.uint8(image)
    thinned = thinning(image8.astype(np.uint8))

    if path is not None:
        new_path = altsep.join((path, "Thinned"))
        if not exists(new_path):
            tl.makeDir(new_path)
        dst = altsep.join((new_path, (name + ".png")))
        cv2.imwrite(dst, thinned)

    if plot:
        cv2.imshow("Thinning '{}'".format(name), thinned)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    return thinned
