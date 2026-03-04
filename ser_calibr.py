# -*- coding: utf-8 -*-
"""
Segmentación y calibración automática de frutas
Colores: Rojo, Verde, Naranja, Amarillo
Autor: Maxwell
"""

import cv2
import numpy as np
import os
import time
import csv
import psutil
import json
from sklearn.cluster import KMeans
from datetime import datetime

# -------------------
# CONFIGURACIÓN
# -------------------
ruta_imagenes = "dataset"
guardar_resultados = True
guardar_paneles = True
espacios_color = {
    'RGB': lambda img: img,
    'HSV': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
    'YCbCr': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb),
    'LAB': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
}
kernel = np.ones((5,5), np.uint8)
colores_frutas = ["Rojo", "Verde", "Naranja", "Amarillo"]
k_clusters = 3  # K-means clusters

# -------------------
# FUNCIONES
# -------------------

def calibrar_rango_kmeans(img_bgr, espacio='HSV', k=3, cluster_obj=0):
    if espacio == 'RGB':
        img = img_bgr
    elif espacio == 'HSV':
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    elif espacio == 'YCbCr':
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    elif espacio == 'LAB':
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    else:
        raise ValueError("Espacio de color no soportado")
    pixels = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
    labels = kmeans.labels_
    cluster_pixels = pixels[labels == cluster_obj]
    lower = np.min(cluster_pixels, axis=0)
    upper = np.max(cluster_pixels, axis=0)
    return lower.astype(int), upper.astype(int)

def segmentar(img, rango):
    lower = np.array(rango[0])
    upper = np.array(rango[1])
    mask = cv2.inRange(img, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    area = cv2.countNonZero(mask)
    return mask, area

def preparar_panel(img_original, img_convertida, mask):
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img_original, 0.7, mask_color, 0.3, 0)
    h = 200
    def resize(img):
        ratio = h / img.shape[0]
        w = int(img.shape[1] * ratio)
        return cv2.resize(img, (w, h))
    return np.hstack([resize(img_original), resize(img_convertida), resize(mask_color), resize(overlay)])

def obtener_imagenes():
    return [os.path.join(ruta_imagenes, f) for f in os.listdir(ruta_imagenes)
            if f.lower().endswith((".jpg",".png"))]

# -------------------
# CALIBRACIÓN AUTOMÁTICA
# -------------------
# Para cada color de fruta y cada espacio de color
imagen_calibracion = cv2.imread(obtener_imagenes()[0])  # primera imagen para calibrar
rangos_calibrados = {}

print("Calibrando rangos automáticamente...")
for fruta_idx, fruta in enumerate(colores_frutas):
    rangos_calibrados[fruta] = {}
    for espacio in espacios_color.keys():
        lower, upper = calibrar_rango_kmeans(imagen_calibracion, espacio=espacio,
                                             k=k_clusters, cluster_obj=fruta_idx % k_clusters)
        rangos_calibrados[fruta][espacio] = [lower.tolist(), upper.tolist()]

# Guardar rangos calibrados
with open("rangos_calibrados_frutas.json", "w") as f:
    json.dump(rangos_calibrados, f, indent=4)
print("Rangos calibrados guardados en rangos_calibrados_frutas.json")

# -------------------
# PROCESAMIENTO DEL DATASET
# -------------------
imagenes = obtener_imagenes()
resultados = []

for img_idx, ruta_img in enumerate(imagenes):
    img_original = cv2.imread(ruta_img)
    print(f"\nProcesando imagen {img_idx+1}/{len(imagenes)}: {os.path.basename(ruta_img)}")
    
    for fruta in colores_frutas:
        for espacio in espacios_color.keys():
            convertir = espacios_color[espacio]
            img_conv = convertir(img_original)
            
            rango = rangos_calibrados[fruta][espacio]
            inicio = time.perf_counter()
            mask, area = segmentar(img_conv, rango)
            fin = time.perf_counter()
            
            delta = fin - inicio
            tiempo_ms = delta * 1000
            fps = 1.0 / delta if delta > 0 else 0
            cpu = psutil.cpu_percent()
            
            resultados.append([fruta, espacio, img_idx, area, tiempo_ms, fps, cpu])
            
            if guardar_paneles:
                panel = preparar_panel(img_original, img_conv, mask)
                cv2.imshow(f"{fruta}-{espacio}", panel)
                cv2.waitKey(200)  # mostrar 200 ms por panel
                cv2.destroyAllWindows()

# -------------------
# GUARDAR RESULTADOS CSV
# -------------------
if guardar_resultados:
    nombre_csv = f"resultados_frutas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(nombre_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fruta", "Espacio", "Imagen_ID", "Area", "Tiempo_ms", "FPS", "CPU_%"])
        writer.writerows(resultados)
    print(f"Resultados guardados en {nombre_csv}")