# coding=utf-8
# import thinning  # Hay que instalarla con pip install thinning
import cv2
from cv2 import ximgproc
import numpy as np
import glob
import matplotlib.pyplot as plt


def almacenamiento(imagen):
    # Función para crear el archivo que almacenará los datos sobre las minucias
    pos1 = imagen.rfind('/')
    pos2 = imagen.rfind('.')
    fichero = open('minucias' + imagen[pos1:pos2] + '.txt', 'w')
    fichero.write('Tipología y posición de las minucias\n')
    fichero.write(imagen[(pos1 + 1):pos2] + '\n \n')
    return fichero


def minucias(thin, img, fichero):
    # Función encargada de localizar las minucias, tanto las terminaciones como las bifurcaciones
    # la información de posición y tipología se almacena en el archivo 'minucias/nombre_imagen.txt'
    # entrada(fichero)

    # Se transforma al rango de color las imágenes, original y la adelgazada
    imgColor = cv2.cvtColor(thin, cv2.COLOR_GRAY2BGR)
    orImgColor = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for i in range(thin.shape[0]):
        for j in range(thin.shape[1]):

            if thin[i, j] == 255:
                # En caso de valer 255 se analizan sus vecinos,
                # para saber si se trata de una terminación o de una bifurcación

                recorte = thin[i - 1:i + 2, j - 1:j + 2]
                vecinos = sum(recorte.ravel()) / 255

                if vecinos == 2:
                    # En caso de que los vecinos sean igual a 2 (Contando el mismo píxel que se analiza)
                    # se trataría de una terminación y esta se almacenaría en el archivo fichero,
                    # también se dibuja en la imágenes un circulo de color verde.
                    fichero.write('Terminación' + '\n')
                    fichero.write('x= ' + str(i) + ' y= ' + str(j) + '\n')

                    cv2.circle(imgColor, (j, i), 1, (0, 255, 0), 1)
                    cv2.circle(orImgColor, (j, i), 1, (0, 255, 0), 1)

                if vecinos > 3:
                    # En caso de que los vecinos sean mayores a 3 (Contando el mismo píxel que se analiza)
                    # se trataría de una bifurcación y esta se almacenaría en el archivo fichero,
                    # también se dibuja en la imágenes un circulo de color rojo.
                    fichero.write('Bifurcación\n')
                    fichero.write('x= ' + str(i) + ' y= ' + str(j) + ' \n')

                    cv2.circle(imgColor, (j, i), 1, (255, 0, 0), 1)
                    cv2.circle(orImgColor, (j, i), 1, (255, 0, 0), 1)

    return imgColor, orImgColor


def resultadoFigura(imgColor, orImgColor, imagen):
    # Función para crear la figura con las imágenes resultantes de la función minucias
    # Esta se almacena en el archivo 'imagenes/nombre_imagen.jpg'
    # En caso de querer visualizar las imágenes seguir los pasos del final de la función
    pos1 = imagen.rfind('/')
    pos2 = imagen.rfind('.')

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(imgColor)
    plt.subplot(1, 2, 2)
    plt.imshow(orImgColor)

    plt.text(20, 450, 'Bifurcacion= Rojo')
    plt.text(20, 470, 'Terminacion= Verde')

    plt.rcParams["figure.figsize"] = [100, 100]

    plt.savefig('imagenes/' + imagen[pos1:pos2] + '.jpg')

    # plt.show() # Descomentar si se quiere visualizar
    plt.close()  # Comentar si se quiere visualizar


def main():
    # Se leen todas las imágenes de la base de datos FVC04
    imagenes = glob.glob('../Data/fingerprints/*.tif')

    for imagen in imagenes[:2]:
        print(imagen)  # Para saber que imagen se está tratando

        fichero = almacenamiento(imagen)  # Creación del archivo de almacenamiento fichero

        # Se lee la imagen y se muestra por pantalla
        img = cv2.imread(imagen, 0)
        cv2.imshow('Original', img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        # Se realiza un filtrado gaussiano, con un kernel 7x7, para eliminar el ruido existente
        kernel = (7, 7)
        blur = cv2.GaussianBlur(img, kernel, 0)
        cv2.imshow('Filtrado', blur)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        # Se umbraliza la imagen con un umbralizado adaptativo y Gaussiano,
        # dando como resultado una binarización de la huella invertida,
        # para tener en blanco la huella y el fondo negro.
        # Y se visualiza por pantalla
        print("**** THRESHOLD ****")
        thr = cv2.adaptiveThreshold(blur.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                    11, 2)
        cv2.imshow('Umbralizado', thr)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        # Utilizando la librería thinning se adelgaza la huella umbralizada y se muestra el resultado por pantalla
        # thin = thinning.guo_hall_thinning(thr)
        print("**** THINNING ****")
        thin = ximgproc.thinning(thr)
        cv2.imshow('Thinning', thin)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # Se buscan las minucias y se almacena la información en el fichero.
        imgColor, orImgColor = minucias(thin, img)
        cv2.imshow("ImgColor", imgColor)
        cv2.imshow("orImgColor", orImgColor)
        cv2.waitKey()
        cv2.destroyAllWindows()

main()