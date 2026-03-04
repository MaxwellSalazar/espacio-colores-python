# -*- coding: utf-8 -*-
"""
Demostración completa: todos los espacios de color en un solo panel
Autor: Maxwell
"""

import cv2
import numpy as np
import os

# -------------------
# CONFIGURACIÓN
# -------------------
ruta_imagenes = "dataset"
imagen_seleccionada = 1

espacios_color = {
    'RGB': lambda img: img,
    'HSV': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
    'YCbCr': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb),
    'LAB': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
}

rangos = {
    'HSV': ([10, 100, 100], [25, 255, 255]),
    'RGB': ([100, 50, 0], [255, 200, 150]),
    'YCbCr': ([0, 150, 100], [255, 200, 150]),
    'LAB': ([20, 140, 140], [255, 200, 200])
}

kernel = np.ones((5,5), np.uint8)

# -------------------
# FUNCIONES
# -------------------

def segmentar(img, espacio_nombre):
    convertir = espacios_color[espacio_nombre]
    img_convertida = convertir(img)
    lower = np.array(rangos[espacio_nombre][0])
    upper = np.array(rangos[espacio_nombre][1])
    mask = cv2.inRange(img_convertida, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return img_convertida, mask

def obtener_imagenes():
    return [os.path.join(ruta_imagenes, f) for f in os.listdir(ruta_imagenes) if f.endswith((".jpg",".png"))]

def preparar_panel(img_original, img_convertida, mask):
    # Convertir máscara a color
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Superposición
    overlay = cv2.addWeighted(img_original, 0.7, mask_color, 0.3, 0)
    # Redimensionar a altura común
    h = 200
    def resize(img):
        ratio = h / img.shape[0]
        w = int(img.shape[1] * ratio)
        return cv2.resize(img, (w, h))
    return np.hstack([resize(img_original), resize(img_convertida), resize(mask_color), resize(overlay)])

# -------------------
# EJECUCIÓN
# -------------------

imagenes = obtener_imagenes()
if len(imagenes) == 0:
    print("No hay imágenes en la carpeta.")
    exit()

img = cv2.imread(imagenes[imagen_seleccionada])

paneles = []

for nombre_espacio in espacios_color.keys():
    img_conv, mask = segmentar(img, nombre_espacio)

    # Ajuste de color para visualización
    if nombre_espacio == 'LAB':
        img_conv_show = cv2.cvtColor(img_conv, cv2.COLOR_Lab2BGR)
    elif nombre_espacio == 'YCbCr':
        img_conv_show = cv2.cvtColor(img_conv, cv2.COLOR_YCrCb2BGR)
    else:
        img_conv_show = img_conv

    panel = preparar_panel(img, img_conv_show, mask)
    # Agregar texto sobre cada panel
    panel = cv2.putText(panel, nombre_espacio, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    paneles.append(panel)

# Combinar todos los paneles verticalmente
final_panel = np.vstack(paneles)
cv2.imshow("Proceso completo de segmentación cromática", final_panel)
print("Presiona cualquier tecla para cerrar la ventana.")
cv2.waitKey(0)
cv2.destroyAllWindows()