# -*- coding: utf-8 -*-
"""
Análisis completo de resultados de segmentación de frutas
Autor: Maxwell
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
archivo_csv = "resultados_frutas_20260304_124516.csv"
guardar_figuras = True
carpeta_figuras = "figuras_analisis"
imagen_alto = 640  # altura de las imágenes usadas (para cálculo %)
imagen_ancho = 480
img_size = imagen_alto * imagen_ancho  # total de pixeles por imagen

# Crear carpeta de figuras si no existe
if guardar_figuras:
    os.makedirs(carpeta_figuras, exist_ok=True)

# -----------------------------
# CARGAR CSV
# -----------------------------
df = pd.read_csv(archivo_csv)
print(f"Archivo cargado: {archivo_csv}")
print(df.head())

# -----------------------------
# RESUMEN ESTADÍSTICO
# -----------------------------
resumen = df.groupby(["Fruta","Espacio"]).agg({
    "Area":["mean","std"],
    "Tiempo_ms":["mean","std"],
    "FPS":["mean","std"],
    "CPU_%":["mean","std"]
}).reset_index()

# Aplanar MultiIndex
resumen.columns = ["Fruta","Espacio",
                   "Area_mean","Area_std",
                   "Tiempo_ms_mean","Tiempo_ms_std",
                   "FPS_mean","FPS_std",
                   "CPU_mean","CPU_std"]

# Calcular % de cobertura relativa
resumen["Area_pct"] = resumen["Area_mean"] / img_size * 100
resumen["Area_pct_std"] = resumen["Area_std"] / img_size * 100

print("\nResumen de métricas por fruta y espacio:")
print(resumen)

# -----------------------------
# FUNCIONES DE GRAFICA
# -----------------------------
def graficar_metricas(df_sub, fruta, metric, metric_std, ylabel, titulo, nombre_archivo=None):
    espacios = df_sub["Espacio"]
    valores = df_sub[metric]
    errores = df_sub[metric_std]
    
    plt.figure(figsize=(8,6))
    barras = plt.bar(espacios, valores, yerr=errores, capsize=5, color=['red','gold','green','blue'])
    plt.title(f"{titulo} - {fruta}")
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.3)
    
    # Escribir valores sobre cada barra
    for bar, val in zip(barras, valores):
        plt.text(bar.get_x() + bar.get_width()/2, val + max(valores)*0.02, f"{val:.1f}", ha='center', va='bottom')
    
    plt.tight_layout()
    if guardar_figuras and nombre_archivo:
        plt.savefig(os.path.join(carpeta_figuras, nombre_archivo), dpi=300)
    plt.show()

# -----------------------------
# GENERAR GRÁFICAS POR FRUTA
# -----------------------------
for fruta in resumen["Fruta"].unique():
    df_sub = resumen[resumen["Fruta"]==fruta]
    
    graficar_metricas(df_sub, fruta, "Tiempo_ms_mean", "Tiempo_ms_std", "ms", "Tiempo promedio", f"{fruta}_tiempo.png")
    graficar_metricas(df_sub, fruta, "FPS_mean", "FPS_std", "FPS", "FPS promedio", f"{fruta}_fps.png")
    graficar_metricas(df_sub, fruta, "CPU_mean", "CPU_std", "%", "Uso de CPU", f"{fruta}_cpu.png")
    graficar_metricas(df_sub, fruta, "Area_mean", "Area_std", "pixeles", "Área promedio (pixeles)", f"{fruta}_area.png")
    graficar_metricas(df_sub, fruta, "Area_pct", "Area_pct_std", "%", "Área relativa (%)", f"{fruta}_area_pct.png")

print("\n¡Análisis completo y figuras generadas!")