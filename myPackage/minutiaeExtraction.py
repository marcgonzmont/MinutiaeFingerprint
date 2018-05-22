import cv2
import numpy as np
from myPackage import tools as tl
from os.path import join, altsep, basename

def process(skeleton, name, plot= False, path= None):
    print("Minutiae extraction...")
    img = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    (h,w) = skeleton.shape[:2]
    if path is not None:
        filename = 'minutiae_'+name+'.txt'
        full_name = altsep.join((path, filename))
        file = open(full_name, 'w')
        file.write('# (x, y) position of minutiae and type as class (0: termination, 1: bifurcation)\n')
    for i in range(h):
        for j in range(w):
            if skeleton[i, j] == 255:
                # En caso de valer 255 se analizan sus vecinos,
                # para saber si se trata de una terminación o de una bifurcación

                window = skeleton[i - 1:i + 2, j - 1:j + 2]
                neighbours = sum(window.ravel()) // 255

                if neighbours == 2:
                    # En caso de que los vecinos sean igual a 2 (Contando el mismo píxel que se analiza)
                    # se trataría de una terminación y esta se almacenaría en el archivo fichero,
                    # también se dibuja en la imágenes un circulo de color verde.
                    if path is not None:
                        file.write(str(i) + ',' + str(j) + ',0\n')
                    cv2.circle(img, (j, i), 1, (0, 255, 0), 1)

                if neighbours > 3:
                    # En caso de que los vecinos sean mayores a 3 (Contando el mismo píxel que se analiza)
                    # se trataría de una bifurcación y esta se almacenaría en el archivo fichero,
                    # también se dibuja en la imágenes un circulo de color rojo.
                    if path is not None:
                        file.write(str(i) + ',' + str(j) + ',1\n')
                    cv2.circle(img, (j, i), 1, (255, 0, 0), 1)
    if plot:
        cv2.imshow("Minutiae '{}'".format(name), img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
