import cv2
import numpy as np
from myPackage import tools as tl
from skimage.morphology import skeletonize, skeletonize_3d, thin



def cleanImage(nameImage, plot= False):
    n = 3
    n_2 = 5
    iter = 2 #3
    iter_2 = 2
    kernel = (n, n)
    kernel2 = (n_2, n_2)
    image = cv2.imread(nameImage)
    if image.shape[2] == 3:
        # print("COLOR")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # bin_img = cv2.dilate(thresh, kernel= kernel2, iterations= iter_2)
    bin_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations= iter)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations= iter)
    # bin_img = cv2.erode(thresh, kernel=kernel2, iterations=iter_2)


    if plot:
        titles = ["original", "binary"]
        images = [image, bin_img]
        title = "Binaryzation"
        tl.plotImages(titles, images, title, 1, 2)

    return bin_img

def process_skeleton(bin_img_inv, plot= False):
    bin_img_inv = cv2.normalize(bin_img_inv, 0, 255, cv2.NORM_MINMAX)
    # aux = bin_img_inv.copy()
    # skeleton = skeletonize(bin_img_inv)
    # skeleton3d = skeletonize_3d(bin_img_inv)
    thinned = thin(bin_img_inv)
    thinned_partial = thin(bin_img_inv, max_iter= 1)

    if plot:
        # titles = ["bin_inv", "thinning", "thinning_par", "bin_inv", "skeletonize", "skeletonize_3D"]
        # images = [bin_img_inv, thinned, thinned_partial, bin_img_inv, skeleton, skeleton3d]
        titles = ["bin_inv", "thinning", "thinning_par"]
        images = [bin_img_inv, thinned, thinned_partial]
        # titles = ["bin_inv", "skeleton", "skeleton3D"]
        # images = [bin_img_inv, skeleton, skeleton3d]
        title = "Skeletonize"
        tl.plotImages(titles, images, title, 1, 3)
    thinned_partial = thinned_partial.astype(int)*255
    thinned_partial = cv2.cvtColor(thinned_partial, cv2.COLOR_GRAY2RGB)
    return thinned_partial

def thinning(bin_img_inv, plot):
    bin_cp = cv2.normalize(bin_img_inv, 0, 255, cv2.NORM_MINMAX)
    # bin_cp /= 255
    prev = np.zeros_like(bin_cp)
    diff = np.zeros_like(bin_cp)
    while True:
        bin_cp = thinningIteration(bin_cp, 0)
        bin_cp = thinningIteration(bin_cp, 1)
        diff = cv2.absdiff(bin_cp, prev)
        prev = bin_cp.copy()
        if cv2.countNonZero(diff) > 0:
            break
    # bin_cp *= 255

    if plot:
        titles = ["bin_inv", "Thinned"]
        images = [bin_img_inv, bin_cp]
        title = "Thinning image"
        tl.plotImages(titles, images, title, 1, 2)

    return bin_img_inv

def thinningIteration(bin_img_inv, iter):
    (h, w) = bin_img_inv.shape
    # print(h,w)
    marker = np.zeros_like(bin_img_inv)
    for i in range(h-1):
        for j in range(w-1):
            p2 = bin_img_inv[i - 1, j]
            p3 = bin_img_inv[i - 1, j + 1]
            p4 = bin_img_inv[i, j + 1]
            p5 = bin_img_inv[i + 1, j + 1]
            p6 = bin_img_inv[i + 1, j]
            p7 = bin_img_inv[i + 1, j - 1]
            p8 = bin_img_inv[i, j - 1]
            p9 = bin_img_inv[i - 1, j - 1]

            A = int((p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) + (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) + \
                (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) + (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1))

            B = int(p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)
            m1 = (p2 * p4 * p6) if iter == 0 else (p2 * p4 * p8)
            m2 = (p4 * p6 * p8) if iter == 0 else (p2 * p6 * p8)

            if (A == 1 and (B >= 2 and B <= 6) and m1 == 0 and m2 == 0):
                marker[i, j] = 1
    bin_img_inv &= ~marker
    return bin_img_inv