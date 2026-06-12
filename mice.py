#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
import missingno as msno
import pandas as pd
from scipy.stats import ks_2samp
import seaborn as sns
import Linear_Regression


# In[2]:


def mice_f(df, cats, nums):

    df_clean = df.copy()
    ax = msno.matrix(df_clean[cats + nums], labels=True)
        # 2. Add the Title
    # You can adjust the fontsize and pad (distance from the graph) as needed
    ax.set_title("Missing Data before imputation", fontsize=22, pad=20)

    # 3. Add the Grid
    # alpha controls transparency so the grid doesn't overpower the data bars
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    # Guardar y mostrar
    plt.savefig(
        "comparativa_imputacion_before.png", 
        dpi=300, 
        bbox_inches="tight", 
        facecolor=ax.get_facecolor()
    )
    plt.show()
    # 1. Asegurar que las columnas existan y sean numéricas para MICE
    # (Incluso las de 0 y 1 deben ser float/int temporalmente)
    cols_a_imputar = cats + nums
    for col in cols_a_imputar:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 2. Configurar MICE con Random Forest (Estimador muy preciso para medicina)
    print("Iniciando imputación MICE (esto puede tardar unos segundos)...")
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=20, random_state=42),
        #estimator = BayesianRidge(),
        sample_posterior=False,
        max_iter=10,
    )
    
    # Solo imputamos las columnas clínicas para evitar ruido de IDs o tiempos
    df_imputado_vals = imputer.fit_transform(df_clean[cols_a_imputar])
    
    # Creamos el DataFrame con los resultados
    df_imputado = pd.DataFrame(df_imputado_vals, columns=cols_a_imputar, index=df.index)
    
    # 3. Post-procesamiento: Redondeo de binarias (evita que un 0 sea 0.001)
    for col in cats:
        df_imputado[col] = df_imputado[col].round().astype(int)
        
    # 4. Mantener las columnas que no se imputaron (como el tiempo de seguimiento o el evento)
    cols_extra = [c for c in df.columns if c not in cols_a_imputar]
    df_final = pd.concat([df_imputado, df[cols_extra]], axis=1)
    #msno.matrix(df_final[cats + nums], labels=True)
    print("¡Preprocesado listo!")
    return df_final


# 
# 

# In[3]:


def mice_multiple_datasets(df, cats, nums, n_imputaciones=10):
    """
    Genera una lista de N datasets imputados usando diferentes semillas.
    """
    lista_datasets = []
    
    for i in range(n_imputaciones):
        df_clean = df.copy()
        cols_a_imputar = cats + nums
        
        # 1. Asegurar tipos numéricos
        for col in cols_a_imputar:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # 2. MICE con semilla variable (i)
        # Usamos sample_posterior=True para que cada imputación sea una extracción aleatoria
        # 2. MICE con BayesianRidge (Compatible con sample_posterior=True)
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=20, random_state=i),
            max_iter=10,
            random_state=i,
            sample_posterior=False
        )
        
        df_imputado_vals = imputer.fit_transform(df_clean[cols_a_imputar])
        df_imputado = pd.DataFrame(df_imputado_vals, columns=cols_a_imputar, index=df.index)
        
        # 3. Redondeo de categóricas
        for col in cats:
            df_imputado[col] = df_imputado[col].round().astype(int)
            
        # 4. Recuperar columnas no imputadas
        cols_extra = [c for c in df.columns if c not in cols_a_imputar]
        df_final = pd.concat([df_imputado, df[cols_extra]], axis=1)
        
        lista_datasets.append(df_final)
        print(f"Dataset {i+1}/{n_imputaciones} imputado con éxito.")
        
    return lista_datasets

