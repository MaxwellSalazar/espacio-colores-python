# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 08:53:08 2025

@author: maxwell
"""
import cv2
import numpy as np
import datetime

color_actual = 1  # 1: rojo, 2: verde, 3: amarillo, 4: naranja
espacio_actual = 1  # 0: RGB, 1: HSV, 2: YCbCr, 3: LAB
modo_calibracion = True  # Calibrar con trackbars

colores = {1: 'rojo', 2: 'verde', 3: 'amarillo', 4: 'naranja'}
espacios_color = {
    0: ('RGB', lambda img: img),
    1: ('HSV', lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HSV)),
    2: ('YCbCr', lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)),
    3: ('LAB', lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
}

# RANGOS por color y espacio, con lista de rangos (cada rango: (lower, upper))
rangos = {
    'HSV': {
        'rojo': [
            ([0, 120, 70], [10, 255, 255]),
            ([170, 120, 70], [180, 255, 255])
        ],
        'verde': [
            ([36, 50, 70], [89, 255, 255])
        ],
        'amarillo': [
            ([20, 100, 100], [30, 255, 255])
        ],
        'naranja': [
            ([10, 100, 100], [20, 255, 255])
        ]
    },
    'YCbCr': {
        'rojo': [
            ([0, 140, 100], [255, 180, 130])
        ],
        'verde': [
            ([0, 80, 90], [255, 120, 120])
        ],
        'amarillo': [
            ([0, 130, 140], [255, 170, 180])
        ],
        'naranja': [
            ([0, 150, 100], [255, 190, 140])
        ]
    },
    'LAB': {
        'rojo': [
            ([20, 150, 130], [255, 200, 170])
        ],
        'verde': [
            ([20, 80, 70], [255, 130, 120])
        ],
        'amarillo': [
            ([20, 120, 140], [255, 150, 170])
        ],
        'naranja': [
            ([20, 140, 140], [255, 180, 170])
        ]
    },
    'RGB': {  # No definido, rangos totales para ejemplo
        'rojo': [
            ([0, 0, 0], [255, 255, 255])
        ],
        'verde': [
            ([0, 0, 0], [255, 255, 255])
        ],
        'amarillo': [
            ([0, 0, 0], [255, 255, 255])
        ],
        'naranja': [
            ([0, 0, 0], [255, 255, 255])
        ]
    }
}

# Función para obtener rango principal para cargar sliders (tomamos primer rango)
def obtener_rango_principal(color, espacio):
    espacio_dict = rangos.get(espacio, {})
    lista_rangos = espacio_dict.get(color, [])
    if lista_rangos:
        return lista_rangos[0]
    else:
        return ([0,0,0],[255,255,255])

# Video
cap = cv2.VideoCapture(0)

if modo_calibracion:
    cv2.namedWindow("Calibración", cv2.WINDOW_NORMAL)
    for name in ["L1", "A1", "B1", "L2", "A2", "B2"]:
        cv2.createTrackbar(name, "Calibración", 0, 255, lambda x: None)

def actualizar_trackbars(color_nombre, espacio_nombre):
    lower, upper = obtener_rango_principal(color_nombre, espacio_nombre)
    cv2.setTrackbarPos("L1", "Calibración", lower[0])
    cv2.setTrackbarPos("A1", "Calibración", lower[1])
    cv2.setTrackbarPos("B1", "Calibración", lower[2])
    cv2.setTrackbarPos("L2", "Calibración", upper[0])
    cv2.setTrackbarPos("A2", "Calibración", upper[1])
    cv2.setTrackbarPos("B2", "Calibración", upper[2])

actualizar_trackbars(colores[color_actual], espacios_color[espacio_actual][0])

def preparar(img):
    img = cv2.resize(img, (480, 360))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

ultimo_color = color_actual
ultimo_espacio = espacio_actual

print("Teclas: 1–4 colores | q–r espacios | s guardar | p imprimir rangos | ESC salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    color_nombre = colores[color_actual]
    nombre_espacio, convertir = espacios_color[espacio_actual]
    espacio = convertir(frame.copy())

    if color_actual != ultimo_color or espacio_actual != ultimo_espacio:
        actualizar_trackbars(color_nombre, nombre_espacio)
        ultimo_color = color_actual
        ultimo_espacio = espacio_actual

    l1 = cv2.getTrackbarPos("L1", "Calibración")
    a1 = cv2.getTrackbarPos("A1", "Calibración")
    b1 = cv2.getTrackbarPos("B1", "Calibración")
    l2 = cv2.getTrackbarPos("L2", "Calibración")
    a2 = cv2.getTrackbarPos("A2", "Calibración")
    b2 = cv2.getTrackbarPos("B2", "Calibración")
    lower = np.array([l1, a1, b1])
    upper = np.array([l2, a2, b2])

    # Crear máscara combinada para todos los rangos
    mascara_total = np.zeros_like(espacio[:,:,0])

    # Se usan dos rangos si existen, sino uno solo
    espacio_dict = rangos.get(nombre_espacio, {})
    lista_rangos = espacio_dict.get(color_nombre, [])
    if len(lista_rangos) == 0:
        # Si no hay rangos definidos, usar el rango calibrado
        mascara_total = cv2.inRange(espacio, lower, upper)
    elif len(lista_rangos) == 1:
        # Solo un rango, usar calibración trackbar
        mascara_total = cv2.inRange(espacio, lower, upper)
    else:
        # Dos rangos -> combinar el rango calibrado con el segundo rango fijo
        lower1 = lower
        upper1 = upper
        lower2 = np.array(lista_rangos[1][0])
        upper2 = np.array(lista_rangos[1][1])
        mask1 = cv2.inRange(espacio, lower1, upper1)
        mask2 = cv2.inRange(espacio, lower2, upper2)
        mascara_total = cv2.bitwise_or(mask1, mask2)

    bordes = cv2.Canny(mascara_total, 50, 150)
    area = cv2.countNonZero(mascara_total)
    contornos, _ = cv2.findContours(mascara_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resultado = frame.copy()
    cv2.drawContours(resultado, contornos, -1, (255, 0, 0), 2)
    cv2.putText(resultado, f"{color_nombre} | {nombre_espacio}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(resultado, f"Area: {area}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    mask_color = cv2.cvtColor(mascara_total, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask_color, contornos, -1, (0, 0, 255), 2)

    p1 = preparar(frame)
    p2 = preparar(espacio)
    p3 = preparar(mascara_total)
    p4 = preparar(bordes)
    p5 = preparar(resultado)
    p6 = preparar(mask_color)

    fila1 = cv2.hconcat([p1, p2, p3])
    fila2 = cv2.hconcat([p4, p5, p6])
    mosaico = cv2.vconcat([fila1, fila2])

    cv2.imshow("Panel de procesamiento", mosaico)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('1'):
        color_actual = 1
    elif key == ord('2'):
        color_actual = 2
    elif key == ord('3'):
        color_actual = 3
    elif key == ord('4'):
        color_actual = 4
    elif key == ord('q'):
        espacio_actual = 0
    elif key == ord('w'):
        espacio_actual = 1
    elif key == ord('e'):
        espacio_actual = 2
    elif key == ord('r'):
        espacio_actual = 3
    elif key == ord('s'):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"panel_{now}.png"
        cv2.imwrite(filename, mosaico)
        print(f"[✔] Imagen guardada: {filename}")
    elif key == ord('p'):
        print(f"LOWER = [{l1}, {a1}, {b1}]")
        print(f"UPPER = [{l2}, {a2}, {b2}]")

cap.release()
cv2.destroyAllWindows()


