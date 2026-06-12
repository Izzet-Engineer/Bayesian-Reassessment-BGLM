#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Librerías para
import bambi as bmb
import pymc as pm

# Alternative libraries for modellong
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# from firthlogist import FirthLogisticRegression

# Manejo de diccionarios y operaciones básicas
import pandas as pd, math
import numpy as np

# Kaplan Meier y cálculo de RMST
from lifelines import CoxPHFitter, KaplanMeierFitter

# Análisis de rendimiento e intervalos de los modelos. Encargado de determinar si los parámetros son 
# significativos y en contrar la combinación de valores reportan los modelos más robustos.
import arviz as az

# Distribuciones de las funciones
from scipy.stats import gamma, chi2, poisson

#Representación de gráficos
import matplotlib.pyplot as plt
import seaborn as sns


#Herramientas para el cálculo de los modelos usando la GPU en vez de la CPU
#Es necesario instalar NVIDIA CUDA Toolkit y cuDNN, además de jax con numpyro para entrenar múltiples modelos en paralelo
import pytensor
import itertools
import jax
jax.config.update("jax_enable_x64", True)
import numpyro
import os


# In[ ]:


# Forzar a PyTensor a usar un modo eficiente
pytensor.config.mode = "NUMBA"

# Configuración opcional para JAX y la memoria de la GPU
# Esto evita que JAX reserve el 90% de la VRAM de golpe
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# In[4]:


# Función individual del modelo de poisson con prior de Jeffrey's
def BayesRegInd(dataset, term, tiempo, categ, val):
    # 1. Preprocesado inicial
    dataset = dataset[dataset[tiempo] > 0].copy()
    columnas_necesarias = list(set(categ + val + [term, tiempo]))
    dataset = dataset[columnas_necesarias].dropna()
    
    for col in dataset.columns:
        if dataset[col].dtype == 'int32' or dataset[col].dtype == 'bool':
            dataset[col] = dataset[col].astype(np.int64)
            
    # B) La Regresión de Poisson EXIGE que la variable objetivo (term) sea un entero estricto.
    # Como en tu dataset n.events es float64, esto lo arregla automáticamente:
    dataset[term] = dataset[term].astype(np.int64)
    
    # 2. Preparación de variables de tiempo
    dataset[tiempo] = pd.to_numeric(dataset[tiempo]) / 365
    dataset['log_time'] = np.log(dataset[tiempo])
    
    # Lista para almacenar los resultados parciales
    tablas_resultados = []
    inferencia = []
    # Priors Jeffreys (No informativos)
    priors_jeffreys = {
        "Intercept": bmb.Prior("Flat"),
        "common": bmb.Prior("Flat")
    }

    # Función auxiliar para procesar y extraer métricas
    def extraer_metricas(model_results, var_name):
        summary = az.summary(model_results, hdi_prob=0.94)
        # Filtramos para no incluir el Intercept en la tabla de comparación de variables
        summary = summary[summary.index != "Intercept"]
        
        summary['Variable'] = var_name
        summary['IRR'] = np.exp(summary['mean'])
        summary['IRR_lower'] = np.exp(summary['hdi_3%'])
        summary['IRR_upper'] = np.exp(summary['hdi_97%'])
        
        return summary[['Variable', 'IRR', 'IRR_lower', 'IRR_upper', 'r_hat']]

    # BUCLE CATEGÓRICAS
    for i in list(set(categ)): # set() evita duplicados
        try:
            print(f"Calculando modelo para: {i}...")
            formula = f"{term} ~ C({i}) + offset(log_time)"
            model = bmb.Model(formula, dataset, family="poisson")
            results = model.fit(priors=priors_jeffreys, draws=5000, tune=2000, target_accept=0.99, progressbar=False, idata_kwargs={"log_likelihood": True}, nuts_sampler="numpyro")
            
            tablas_resultados.append(extraer_metricas(results, i))
            inferencia.append(results) # Guardamos el objeto para el gráfico
        except Exception as e:
            print(f"Error en {i}: {e}")

    # BUCLE NUMÉRICAS
    for j in list(set(val)):
        try:
            print(f"Calculando modelo para: {j}...")
            # Centramos la variable numérica para estabilidad
            dataset_temp = dataset.copy()
            dataset_temp[j] = dataset_temp[j] - dataset_temp[j].mean()
            
            formula = f"{term} ~ {j} + offset(log_time)"
            model = bmb.Model(formula, dataset_temp, family="poisson")
            results = model.fit(priors=priors_jeffreys, draws=5000, tune=2000, target_accept=0.99, progressbar=False, idata_kwargs={"log_likelihood": True}, nuts_sampler="numpyro")
            
            tablas_resultados.append(extraer_metricas(results, j))
            inferencia.append(results)
        except Exception as e:
            print(f"Error en {j}: {e}")

    # 3. UNIFICACIÓN DE TABLA FINAL
    if tablas_resultados:
        tabla_final = pd.concat(tablas_resultados)
        
        # Interpretación
        def interpretar(row):
            if row['IRR_lower'] > 1: return "Incrementa Riesgo"
            if row['IRR_upper'] < 1: return "Protector"
            return "No Significativo"
        
        tabla_final['Efecto'] = tabla_final.apply(interpretar, axis=1)
        
        # Limpiar nombres de índices
        tabla_final.index = tabla_final.index.str.replace("C\(", "", regex=True).str.replace("\)", "", regex=True)
        
        # Guardar a CSV (formato Excel amigable)
        tabla_final.to_csv('Ind.csv', sep=';', decimal='.', encoding='utf-8-sig')
        print(f"\n>>> PROCESO COMPLETADO. Archivo guardado como: Ind.csv")
        
        return inferencia
    else:
        return "No se pudieron generar resultados."


# In[6]:


# Función combinada del modelo de poisson con prior de Jeffrey's. Dependiente de 'probar_todas_las_combinaciones' para el preprocesado de datos y eliminación de valores vacíos.
def BayesRegMul(dataset, term, tiempo, categ, val):
    
    todas_vars = categ + val
    columnas_totales = list(set(todas_vars + [term, tiempo]))
    dataset = dataset[columnas_totales].dropna().copy()
    
    # 2. Preparación de variables de tiempo
    dataset[tiempo] = pd.to_numeric(dataset[tiempo]) / 365
    dataset['log_time'] = np.log(dataset[tiempo])
    
    # Lista para almacenar los resultados parciales
    tablas_resultados = []
    
    # Priors Jeffreys (No informativos)
    priors_jeffreys = {
        "Intercept": bmb.Prior("Flat"),
        "common": bmb.Prior("Flat")
    }
    
        # Función auxiliar para procesar y extraer métricas
    def extraer_metricas(model_results, var_name_list):
        summary = az.summary(model_results, hdi_prob=0.94)
        summary = summary[summary.index != "Intercept"]

        # 1. Convertimos el índice (los 14 nombres) en una columna
        summary = summary.reset_index().rename(columns={'index': 'Variable_Tecnica'})

        # 2. Creamos una columna 'Variable' limpia para tu tabla final
        # Esta lógica busca cuál de tus 8 variables originales está contenida en el nombre técnico
        def encontrar_original(nombre_tecnico):
            for v in var_name_list:
                if v in nombre_tecnico:
                    return v
            return nombre_tecnico # Por si no encuentra coincidencia

        summary['Variable'] = summary['Variable_Tecnica'].apply(encontrar_original)

        # 3. Cálculos de IRR (Tasas de Incidencia)
        summary['IRR'] = np.exp(summary['mean'])
        summary['IRR_lower'] = np.exp(summary['hdi_3%'])
        summary['IRR_upper'] = np.exp(summary['hdi_97%'])

        # Devolvemos la tabla con la columna 'Variable' ya rellena correctamente
        return summary[['Variable', 'IRR', 'IRR_lower', 'IRR_upper', 'r_hat']]

    for v in val:
        # Aplicamos tu fórmula: Valor = Media - Valor Actual
        dataset[v] = dataset[v].mean() - dataset[v]
    
    # Definir la función del modelo
    cat_f = [f"C({col})" for col in categ]
    pred = cat_f + val
    formula_mult = f"{term} ~ {' + '.join(pred)} + offset(log_time)"
    dataset.info()
    # Cálculo y entreno del modelo
    print(f"Calculando modelo múltiple: ...")
    model = bmb.Model(formula_mult, dataset, family="poisson")
    results = model.fit(priors=priors_jeffreys, draws=5000, tune=2000, target_accept=0.99, progressbar=False, idata_kwargs={"log_likelihood": True}, nuts_sampler="numpyro")
    print(categ+val)
    print(results)
    # Validación (PPC) ---
    print("Generando Chequeo Predictivo Posterior (PPC)...")
    # Generamos muestras de la distribución predictiva
    model.predict(results, kind="pps")
    
    # Gráfico de densidad PPC
    az.plot_ppc(results, num_pp_samples=100)
    plt.title(f"PPC - {term}")
    plt.show()
    tablas_resultados.append(extraer_metricas(results, categ + val))
            
 # Unificación de todos los valores recogidos en una misma tabla
    if len(tablas_resultados) > 0:
        tabla_final = pd.concat(tablas_resultados)
        
        # Interpretación
        def interpretar(row):
            if row['IRR_lower'] > 1: return "Incrementa Riesgo"
            if row['IRR_upper'] < 1: return "Protector"
            return "No Significativo"
        
        tabla_final['Efecto'] = tabla_final.apply(interpretar, axis=1)
        
        tabla_final['Variable'] = tabla_final['Variable'].astype(str)
    
        # Limpiamos los caracteres C( y )
        tabla_final['Variable'] = (tabla_final['Variable']
                                   .str.replace(r"C\(", "", regex=True)
                                   .str.replace(r"\)", "", regex=True))
        
        # Solución a 'Index Error': Crear un nombre basado en las variables reales
        nombre_combinacion = "_".join(categ + val)
        # Limitamos el nombre por si la combinación es muy larga
        nombre_archivo = (nombre_combinacion[:50] + '..') if len(nombre_combinacion) > 50 else nombre_combinacion
        # Eliminar cualquier fila que sea un parámetro de observación (como mu o interceptos individuales)
        tabla_final = tabla_final[~tabla_final['Variable'].str.contains('mu\[|Intercept', na=False)]
        # Guardamos los valores en un fichero '.csv'
        print(results)
        return tabla_final, results
    else:
        return "Los resultados no fueron guardados"


# In[ ]:


# Realizamos múltiples modelos con todas las combinaciones posibles de los parámetros proporcionados. Es recomendable utilizar CUDA para agilizar el proceso o utilizar un número reducido de variables.
def probar_todas_las_combinaciones(dataset, term, tiempo, categ2, val2):
    todas_vars = categ2 + val2
    # Eliminamos todos los NANs presentes en las variables proporcionadas. (Es posible que los resultados cambien en base si se utiliza algún método para reemplazar los valores vacíos)
    columnas_totales = list(set(todas_vars + [term, tiempo]))
    df_limpio = dataset[columnas_totales].dropna().copy()
    
    # Mostramos la cantidad de filas eliminadas, si superan un umbral es necesario considerar métodos de reemplazo
    print(f"Filas originales: {len(dataset)} | Filas tras limpiar nulos: {len(df_limpio)}")
    
    # Lista de modelos 
    lista_maestra = [] 
    # Diccionario para az.compare
    modelos_inf = {} 
    
    # Iteración de BayesRegMul para cada combinación de los parámetros
    for r in range(1, len(todas_vars) + 1):
        for combo in itertools.combinations(todas_vars, r):
            c_actual = [v for v in combo if v in categ2]
            v_actual = [v for v in combo if v in val2]
            
            try:
                print(f"Ejecutando: {combo}")
                # Recibimos los dos valores retornados por BayesRegMul
                df_res, inference = BayesRegMul(df_limpio, term, tiempo, c_actual, v_actual)
                
                if isinstance(df_res, pd.DataFrame):
                    lista_maestra.append(df_res)
                    # Guardamos el objeto de inferencia usando el nombre del combo como clave
                    nombre_modelo = "+".join(combo)
                    modelos_inf[nombre_modelo] = inference
                    
            except Exception as e:
                print(f"Error en modelo {combo}: {e}")

    # Procesado de resultados
    if lista_maestra:
        # Guardar tabla de coeficientes global
        resultado_final = pd.concat(lista_maestra, ignore_index=True)
        
        # Interpretación del IRR
        def interpretar(row):
            if row['IRR_lower'] > 1: return "Incrementa Riesgo"
            if row['IRR_upper'] < 1: return "Protector"
            return "No Significativo"
        # Guardar resultados en .csv
        resultado_final['Efecto'] = resultado_final.apply(interpretar, axis=1)
        resultado_final.to_csv('Resultados_Todos_Los_Modelos.csv', sep=';', decimal='.', index=False, encoding='utf-8-sig')

            # --- COMPARATIVA AUTOMÁTICA ---
        if len(modelos_inf) > 1:
            print("\nGenerando comparativa LOO y WAIC...")
            # Comparamos usando LOO (por defecto) y WAIC
            comp_loo = az.compare(modelos_inf, ic="loo")
            comp_waic = az.compare(modelos_inf, ic="waic")

            comp_loo.to_csv('Comparativa_LOO.csv', sep=';')
            comp_waic.to_csv('Comparativa_WAIC.csv', sep=';')

            # Gráfico de comparación
            az.plot_compare(comp_loo)
            plt.title("Comparación de Modelos (LOO)")
            plt.show()
        
        print("¡Proceso finalizado con éxito!")
    else:
        print("No se generaron resultados.")


# In[ ]:


def generar_grafico(csv_path, name, max_irr=6):
    # 1. Carga de datos oficiales
    df = pd.read_csv(csv_path, sep=';', decimal='.')
    df.columns = [c.strip() for c in df.columns]
    
    # 2. Limpieza: quitar Intercepto y ordenar por magnitud del efecto
    df = df[df['Variable'] != 'Intercept']
    df = df.sort_values(by='IRR', ascending=True)

    # 3. Estilo del gráfico
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 14))
    y_pos = np.arange(len(df))

    # 4. DIBUJO DE PRECISIÓN
    # Línea fina: representa exactamente el intervalo IRR_lower a IRR_upper del CSV
    ax.hlines(y_pos, df['IRR_lower'], df['IRR_upper'], color='steelblue', alpha=0.7, linewidth=2)
    
    # Punto central: representa el IRR (Media)
    ax.scatter(df['IRR'], y_pos, color='navy', s=50, zorder=3, label='IRR (Media)')

    # 5. Referencias Visuales
    ax.axvline(1, color='#c0392b', linestyle='--', linewidth=2.5, zorder=1) # Línea Nula
    ax.axvspan(0, 1, color='green', alpha=0.04, label='Protector') # Zona Verde
    ax.axvspan(1, max_irr, color='red', alpha=0.04, label='Riesgo') # Zona Roja

    # 6. Etiquetas y Ejes
    # Usamos la primera columna del CSV para los nombres de las categorías
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.iloc[:, 0], fontsize=10)
    
    ax.set_xlim(0, max_irr)
    ax.set_xticks(np.arange(0, max_irr + 0.5, 0.5))

    ax.set_title("Factores de Riesgo (IRR) - Análisis Basado en Intervalos Reales", fontsize=15, pad=20)
    ax.set_xlabel("Incidence Rate Ratio (IRR)", fontsize=12)
    
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(f'{name} Incidence Rate Ratio.png', dpi=300, bbox_inches="tight")
    plt.show()


# In[ ]:


def comparar_distribuciones(df_raw, df_cat, df_mice, variable):
    plt.figure(figsize=(8, 5))
    
    # Curva de los datos originales (ignorando nulos)
    sns.kdeplot(df_raw[variable].dropna(), color='blue', label='Original (Sin Nulos)', linewidth=2)
    
    # Curva de los datos originales (ignorando nulos)
    sns.kdeplot(df_cat[variable].dropna(), color='red', linestyle='-.', label='Computados (Anticoag./Antiagr.)', linewidth=2)
    
    # Curva de los datos tras MICE
    sns.kdeplot(df_mice[variable], color='green', linestyle='--', label='Imputado (MICE)', linewidth=2)
    
    plt.title(f'Validación MICE: Distribución de {variable}')
    plt.xlabel(variable)
    plt.ylabel('Densidad')
    plt.legend()
    plt.savefig(f'{variable}.png', dpi=300, bbox_inches='tight')
    plt.show()

