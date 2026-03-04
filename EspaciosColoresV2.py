# -*- coding: utf-8 -*-
"""
Comparative Chromatic Segmentation Experimental Platform
Author: Maxwell
"""

import cv2
import numpy as np
import os
import time
import csv
import psutil
from datetime import datetime

# ===============================
# CONFIGURACIÃ“N GENERAL
# ===============================

modo_experimento = True
usar_camara = False
ruta_imagenes = "dataset"   # Carpeta con imÃ¡genes
guardar_resultados = True

espacios_color = {
    'RGB': lambda img: img,
    'HSV': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
    'YCbCr': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb),
    'LAB': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
}

# RANGOS REALES PARA TODOS LOS ESPACIOS
rangos = {
    'HSV': ([10, 100, 100], [25, 255, 255]),
    'RGB': ([100, 50, 0], [255, 200, 150]),
    'YCbCr': ([0, 150, 100], [255, 200, 150]),
    'LAB': ([20, 140, 140], [255, 200, 200])
}

kernel = np.ones((5,5), np.uint8)

# ===============================
# FUNCIONES
# ===============================

def segmentar(img, espacio_nombre):
    convertir = espacios_color[espacio_nombre]
    img_convertida = convertir(img)

    lower = np.array(rangos[espacio_nombre][0])
    upper = np.array(rangos[espacio_nombre][1])

    mask = cv2.inRange(img_convertida, lower, upper)

    # Operaciones morfolÃ³gicas
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    area = cv2.countNonZero(mask)

    return mask, area

def obtener_imagenes():
    imagenes = []
    for file in os.listdir(ruta_imagenes):
        if file.endswith(".jpg") or file.endswith(".png"):
            imagenes.append(os.path.join(ruta_imagenes, file))
    return imagenes

# ===============================
# MODO EXPERIMENTO
# ===============================

if modo_experimento:

    resultados = []

    if usar_camara:
        cap = cv2.VideoCapture(0)

    else:
        lista_imagenes = obtener_imagenes()

    for nombre_espacio in espacios_color.keys():

        print(f"\nProcesando espacio: {nombre_espacio}")

        if usar_camara:
            ret, frame = cap.read()
            imagenes = [frame]
        else:
            imagenes = [cv2.imread(img_path) for img_path in lista_imagenes]
            imagenes = [img for img in imagenes if img is not None]

        for idx, img in enumerate(imagenes):

            inicio = time.perf_counter()

            mask, area = segmentar(img, nombre_espacio)

            fin = time.perf_counter()

            delta = fin - inicio
            tiempo_procesamiento = delta * 1000  # ms
            fps = (1.0 / delta) if delta > 0 else 0.0
            cpu = psutil.cpu_percent()

            resultados.append([
                nombre_espacio,
                idx,
                area,
                tiempo_procesamiento,
                fps,
                cpu
            ])

            print(f"{nombre_espacio} | Img {idx} | Area: {area} | "
                  f"{tiempo_procesamiento:.2f} ms | FPS: {fps:.2f} | CPU: {cpu}%")

    if guardar_resultados:
        nombre_csv = f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(nombre_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Espacio", "Imagen_ID", "Area", "Tiempo_ms", "FPS", "CPU_%"])
            writer.writerows(resultados)

        print(f"\n[âœ”] Resultados guardados en {nombre_csv}")

