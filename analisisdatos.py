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
archivo_csv = "resultados_frutas_20260304_124516.csv"  # Cambia según tu CSV
guardar_figuras = True
carpeta_figuras = "figuras_analisis"
imagen_alto = 640  # altura de las imágenes usadas
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

# -----------------------------
# PANEL RESUMEN POR FRUTA
# -----------------------------
for fruta in resumen["Fruta"].unique():
    df_sub = resumen[resumen["Fruta"]==fruta]
    
    fig, axs = plt.subplots(2,2, figsize=(12,8))
    fig.suptitle(f"Resumen de métricas - {fruta}", fontsize=16)
    
    # Tiempo
    barras = axs[0,0].bar(df_sub["Espacio"], df_sub["Tiempo_ms_mean"], 
                          yerr=df_sub["Tiempo_ms_std"], capsize=5,
                          color=['red','gold','green','blue'])
    axs[0,0].set_title("Tiempo promedio (ms)")
    axs[0,0].grid(axis='y', alpha=0.3)
    for bar, val in zip(barras, df_sub["Tiempo_ms_mean"]):
        axs[0,0].text(bar.get_x()+bar.get_width()/2, val+max(df_sub["Tiempo_ms_mean"])*0.02,
                      f"{val:.1f}", ha='center', va='bottom')
    
    # FPS
    barras = axs[0,1].bar(df_sub["Espacio"], df_sub["FPS_mean"], 
                          yerr=df_sub["FPS_std"], capsize=5,
                          color=['red','gold','green','blue'])
    axs[0,1].set_title("FPS promedio")
    axs[0,1].grid(axis='y', alpha=0.3)
    for bar, val in zip(barras, df_sub["FPS_mean"]):
        axs[0,1].text(bar.get_x()+bar.get_width()/2, val+max(df_sub["FPS_mean"])*0.02,
                      f"{val:.1f}", ha='center', va='bottom')
    
    # CPU
    barras = axs[1,0].bar(df_sub["Espacio"], df_sub["CPU_mean"], 
                          yerr=df_sub["CPU_std"], capsize=5,
                          color=['red','gold','green','blue'])
    axs[1,0].set_title("Uso de CPU (%)")
    axs[1,0].grid(axis='y', alpha=0.3)
    for bar, val in zip(barras, df_sub["CPU_mean"]):
        axs[1,0].text(bar.get_x()+bar.get_width()/2, val+max(df_sub["CPU_mean"])*0.02,
                      f"{val:.1f}", ha='center', va='bottom')
    
    # Área relativa (%)
    barras = axs[1,1].bar(df_sub["Espacio"], df_sub["Area_pct"], 
                          yerr=df_sub["Area_pct_std"], capsize=5,
                          color=['red','gold','green','blue'])
    axs[1,1].set_title("Área relativa (%)")
    axs[1,1].grid(axis='y', alpha=0.3)
    for bar, val in zip(barras, df_sub["Area_pct"]):
        axs[1,1].text(bar.get_x()+bar.get_width()/2, val+max(df_sub["Area_pct"])*0.02,
                      f"{val:.1f}", ha='center', va='bottom')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if guardar_figuras:
        nombre_panel = f"{fruta}_resumen_completo.png"
        plt.savefig(os.path.join(carpeta_figuras, nombre_panel), dpi=300)
    
    plt.show()

# -----------------------------
# PANEL COMPARATIVO COMPLETO
# -----------------------------
frutas = resumen["Fruta"].unique()
metricas = [("Tiempo_ms_mean","Tiempo (ms)"),
            ("FPS_mean","FPS"),
            ("CPU_mean","CPU (%)"),
            ("Area_pct","Área relativa (%)")]

fig, axs = plt.subplots(len(frutas), len(metricas), figsize=(18, 12))
fig.suptitle("Comparación completa de métricas por fruta y espacio de color", fontsize=18)

for i, fruta in enumerate(frutas):
    df_sub = resumen[resumen["Fruta"]==fruta]
    
    for j, (metric, ylabel) in enumerate(metricas):
        ax = axs[i,j]
        error_col = metric.replace("_mean","_std") if metric != "Area_pct" else "Area_pct_std"
        barras = ax.bar(df_sub["Espacio"], df_sub[metric], yerr=df_sub[error_col],
                        capsize=5, color=['red','gold','green','blue'])
        ax.set_title(f"{fruta} - {ylabel}", fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(barras, df_sub[metric]):
            ax.text(bar.get_x()+bar.get_width()/2, val+max(df_sub[metric])*0.02,
                    f"{val:.1f}", ha='center', va='bottom', fontsize=8)
        
        if i == len(frutas)-1:
            ax.set_xlabel("Espacio de color")
        if j == 0:
            ax.set_ylabel(fruta)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Guardar figura completa
if guardar_figuras:
    nombre_panel_completo = os.path.join(carpeta_figuras, "comparacion_completa_frutas.png")
    plt.savefig(nombre_panel_completo, dpi=300)

plt.show()
print("\n¡Análisis completo y figuras generadas!")