#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Archivo local con las funciones del modelo 
import Linear_Regression
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt


# In[2]:


# Cargamos el archivo con los datos
file = "watchHDfinalDB.csv"
df_datos = pd.read_csv(file, sep=";")

# Renombramos las variables problemáticas con caracteres utilizados para describir las funciones de bambi (+, -, *, /, ...)
df_datos.rename(columns={'CHA2DS2-VASc': 'CHA2DS2VASc'}, inplace=True)

# Creamos la variable 'IMC' a partir de los datos de la tabla
df_datos['IMC'] = df_datos['Weight']/((df_datos['Height']/100)**2)


# # Definimos las variables del dataset entre categóricas y numéricas. 
# En el caso de variables categóricas con múltiples posibles valores se recomienda considerarlas como numéricas (Conteo de factores de riesgo) o agruparlas por grupos (leve, moderado, alto),
# ya que demasiadas categorías con pocos datos crean una mayor incertidumbre con intervalos más amplios
# 
# ## n.events ~ IMC + CoronRevascPRE + Age + C(Antiagregación) + C(HT) + C(TIA)

# # Modelos individuales

# In[5]:


val = ['Age', 'Weight', 'Height', 'IMC', 'HASBLED', 'CoronRevascPRE', 'IctusPre']
categ = ['BARCg1pre','ReBARC2','CAD','IAM','PeriphArteryDisPRE','TIA','prospect', 'Anticoag', 'Antiagregacion', 'CHA2DS2VASc', 'Male', 'HT', 'DM', 'ResidualLeak', 'AF', 'ReACV']
HDI = Linear_Regression.BayesRegInd(df_datos, 'n.events', 'TiempoFU', categ, val)


# In[6]:


# Ejecución
Linear_Regression.generar_grafico('Ind.csv')


# # Modelo conjunto de todas las variables

# In[8]:


val2 = ['IMC',  'CoronRevascPRE', 'Age'] # val2 = ['Age', 'Weight', 'Height', 'IMC', 'HASBLED', 'CoronRevascPRE', 'IctusPre']
categ2 = ['Antiagregacion','HT', 'TIA'] # categ2 = ['BARCg1pre','ReBARC2','CAD','IAM','PeriphArteryDisPRE','TIA','prospect', 'Anticoag', 'Antiagregacion', 'CHA2DS2VASc', 'Male', 'HT', 'DM', 'ResidualLeak', 'AF', 'ReACV']
HDIm, results = Linear_Regression.BayesRegMul(df_datos, 'n.events', 'TiempoFU', categ2, val2)


# In[9]:


Linear_Regression.generar_grafico('Mult_Antiagregacion_HT_TIA_IMC_CoronRevascPRE_Age.csv')


# # Pruebas de múltiples modelos 

# In[ ]:


val2 = ['IMC',  'CoronRevascPRE', 'Age']
categ2 = ['Antiagregacion','HT', 'TIA'] #, 'ResidualLeak''AF', 'Anticoag'

Test = Linear_Regression.probar_todas_las_combinaciones(df_datos, 'n.events', 'TiempoFU', categ2, val2)


# In[ ]:


print(HDI)


# In[ ]:


conteo = df_datos['CoronRevascPRE'].value_counts()

print(conteo)


# In[ ]:


# Cuenta cuántos eventos hay por cada grupo de HT
print(df_datos.groupby('CoronRevascPRE')['n.events'].sum())

# Mira cuánto tiempo de seguimiento tiene cada grupo
print(df_datos.groupby('CoronRevascPRE')['TiempoFU'].sum())


# In[25]:


# Cálculo de RMST comparativo
# Extraer las muestras de la distribución posterior (Posterior Samples)
posterior = results.posterior

# Calcular el Log-Lambda para el "Sujeto Promedio"
# El intercepto es la base de la tasa
log_lambda_samples = posterior["Intercept"].values.flatten()

# Sumamos el efecto de las covariables multiplicadas por su media
# Nota: Para las categóricas, esto usa la proporción de la categoría en la muestra
for var_name in results.posterior.data_vars:
    if var_name != "Intercept":
        # Extraer muestras de la variable y multiplicar por su valor medio en el dataset original
        # Para las numéricas ya hiciste (Media - Valor), así que su media en el dataset actual es 0
        # Para las categóricas, el modelo ajusta el efecto respecto a la referencia.
        muestras_var = posterior[var_name].values.flatten()
        # Si es numérica centrada, su aporte promedio es 0. 
        # Si no, podrías sumar muestras_var * dataset_final[var_name].mean()
        pass 

# Transformar Log-Lambda a Lambda (Tasa de incidencia)
lambda_samples = np.exp(log_lambda_samples)

# Calcular RMST para cada muestra de la posterior (Horizonte t=1 año, ya que normalizaste /365)
t_horizonte = 1.0 
rmst_samples = (1 - np.exp(-lambda_samples * t_horizonte)) / lambda_samples

# 5. RMST de Referencia (para tasa 0.14)
rmst_ref = (1 - np.exp(-0.14 * t_horizonte)) / 0.14

# 6. Probabilidad de que tu cohorte sea distinta a la referencia
prob_mayor_ref = (rmst_samples > rmst_ref).mean()

print(f"\n--- ANÁLISIS DE RMST (Horizonte {t_horizonte} año) ---")
print(f"RMST Medio Observado: {rmst_samples.mean():.4f}")
print(f"RMST Referencia (0.14): {rmst_ref:.4f}")
print(f"Probabilidad de que el RMST sea mayor al de referencia: {prob_mayor_ref:.2%}")


# In[29]:


import seaborn as sns
import matplotlib.pyplot as plt

def graficar_comparativa_rmst(rmst_samples, rmst_ref):
    plt.figure(figsize=(10, 6))
    
    # 1. Dibujar la distribución de las muestras del RMST de tu modelo
    sns.kdeplot(rmst_samples, fill=True, color="skyblue", label="Posterior RMST (Tu Cohorte)", lw=2)
    
    # 2. Línea vertical para el valor de referencia
    plt.axvline(rmst_ref, color="red", linestyle="--", lw=2, label=f"Referencia (Índice 0.14, RMST={rmst_ref:.4f})")
    
    # 3. Calcular el Intervalo de Credibilidad (HDI) del 94% para sombrearlo
    hdi_lower = np.percentile(rmst_samples, 3)
    hdi_upper = np.percentile(rmst_samples, 97)
    plt.hlines(y=0, xmin=hdi_lower, xmax=hdi_upper, color='navy', lw=5, label='Intervalo Credibilidad 94%')

    # Anotaciones
    plt.title("Distribución Posterior del RMST Ajustado vs. Referencia", fontsize=14)
    plt.xlabel("Años libres de eventos (RMST a 1 año)", fontsize=12)
    plt.ylabel("Densidad de Probabilidad", fontsize=12)
    
    # Añadir texto explicativo en la gráfica
    prob_menor = (rmst_samples < rmst_ref).mean()
    plt.text(rmst_ref, plt.gca().get_ylim()[1]*0.8, f'  Prob(RMST < Ref) = {prob_menor:.1%}', 
             color='red', fontweight='bold')

    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Llamada a la función (usa las variables calculadas en el paso anterior)
graficar_comparativa_rmst(rmst_samples, rmst_ref)


# In[37]:


import numpy as np
import matplotlib.pyplot as plt

def graficar_ajuste_bayesiano_vs_ref(results, dataset, categ, val, tasa_ref=0.14, t_max_anios=1.0):
    # 1. Obtener el Log-Lambda del Intercepto (Paciente promedio por tu centrado de variables)
    # En tu función BayesRegMul, las variables 'val' están centradas (Media - Valor), 
    # por lo que el Intercepto es el log(tasa) del promedio.
    log_lambda_post = results.posterior["Intercept"].values.flatten()
    lambda_medio_obs = np.exp(log_lambda_post.mean())
    
    # 2. Definir el eje de tiempo (años)
    t_eje = np.linspace(0, t_max_anios, 100)
    
    # 3. Calcular Curva de Supervivencia de la Referencia (0.14)
    s_referencia = np.exp(-tasa_ref * t_eje)
    
    # 4. Calcular Curva de Supervivencia Ajustada (Promedio Bayesiano)
    s_ajustada_media = np.exp(-lambda_medio_obs * t_eje)
    
    # 5. Calcular Intervalo de Credibilidad para la curva (opcional pero muy pro)
    # Tomamos 100 muestras de la posterior para ver la incertidumbre
    plt.figure(figsize=(10, 6))
    
    indices_muestra = np.random.choice(len(log_lambda_post), size=100, replace=False)
    for idx in indices_muestra:
        lmb = np.exp(log_lambda_post[idx])
        plt.plot(t_eje, np.exp(-lmb * t_eje), color='skyblue', alpha=0.1, lw=1)

    # 6. Dibujar las líneas principales
    plt.plot(t_eje, s_ajustada_media, color='blue', lw=2.5, 
             label=f'Cohorte Ajustada (Bayes $\lambda$={lambda_medio_obs:.3f})')
    
    plt.plot(t_eje, s_referencia, color='red', linestyle='--', lw=2.5, 
             label=f'Referencia Teórica ($\lambda$={tasa_ref})')

    # Estética
    plt.title("Comparación de Supervivencia Ajustada (Modelo Multivariable)", fontsize=14)
    plt.xlabel("Tiempo (Años)")
    plt.ylabel("Probabilidad libre de eventos")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.2)
    
    # Añadir sombreado de la diferencia de RMST
    plt.fill_between(t_eje, s_ajustada_media, s_referencia, color='gray', alpha=0.2, label='Diferencia RMST')
    
    plt.show()

# Llamada a la función después de ejecutar tu BayesRegMul
graficar_ajuste_bayesiano_vs_ref(results, df_datos, categ2, val2)


# In[ ]:




