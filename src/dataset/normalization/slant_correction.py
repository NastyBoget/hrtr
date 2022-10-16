import numpy as np
import cv2


def slant_angle(img, treshold_up=100, treshold_down=100):
    """ Calculate slant angle. Cursive
        - Check the upper neighboords of pixels with left blank
        - Utilizar despeus de hacer una mejora de contraste.
        - Usar despues de habr corregido el slope de la linea
    Parametros:
        img: Imagen en escala de grises. Positiva no normalizada: fondo valor 0 y negro valor 255
        treshold_up: umbral de gris para decidir que algo es negro
        treshold_down: umbral de gris pra decidir que algo es blanco
    """
    angle = []
    # Contadores para pixeles Centrales, Left y Right
    C = 0
    L = 0
    R = 0
    for w in range(1, img.shape[1] - 1):
        for h in range(2, img.shape[0] - 1):
            if img[h, w] > treshold_up and img[h, w - 1] < treshold_down:  # si pixel negro y blanco a la izquierda..
                if img[h - 1, w - 1] > treshold_up:  # si arriba izquierda es negro
                    L += 1
                    angle += [-45 * 1.25]
                elif img[h - 1, w] > treshold_up:  # si arriba centro es negro
                    C += 1
                    angle += [0]
                elif img[h - 1, w + 1] > treshold_up:  # si arriba derecha es negro
                    R += 1
                    angle += [45 * 1.25]
    return np.arctan2((R - L), (L + C + R))


def correct_slant(img, treshold=100):
    """Corrige slant del texto. Cursiva

    Parametros:
        img: Imagen en escala de grises. Positiva no normalizada: fondo valor 0 y negro valor 255
        treshold:
    """
    # Estimate slant angle
    angle = slant_angle(img, treshold_up=treshold, treshold_down=treshold)

    # convert image to to negative
    img = 255 - img

    # Add blanks in laterals to compensate the shear transformation cut
    if angle > 0:
        img = np.concatenate([img, np.zeros([img.shape[0], int(img.shape[0] * angle)])], axis=1)
    else:
        img = np.concatenate([np.zeros([img.shape[0], int(img.shape[0] * (-angle))]), img], axis=1)

    # Numero de columnas añadidas a la imagen
    # positions//2 permiten ajusta las posiciones de cada palabra si se tiene segmentandas antes de esta transformación
    positions = int(abs(img.shape[0] * angle))

    # shear matrix and affine transformation
    M = np.float32([[1, -angle, 0], [0, 1, 0]])
    img2 = cv2.warpAffine(
        img,
        M,
        (img.shape[1], img.shape[0]),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )

    return img2, angle, positions // 2
