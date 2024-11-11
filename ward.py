# Implementación del método de Ward - Minería de Datos Otoño 2024

'''
## Importación de librerías
'''
from openpyxl import Workbook
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster
import seaborn as sns


'''
## Carga de datos
'''
higher_education_students_performance_evaluation = fetch_ucirepo(id=856) 
features = higher_education_students_performance_evaluation.data.features 
targets = higher_education_students_performance_evaluation.data.targets 
original = higher_education_students_performance_evaluation.data.original

cluster_history = Workbook()
cluster_history_sheet = cluster_history.active
cluster_history_sheet.title = 'Cluster History'

'''
-Limpiamos los datos eliminando las filas con valores nulos y las columnas con valores no numéricos
-Eliminamos la columna "COURSE ID" ya que no es relevante para el análisis
'''
features = features.dropna()
features = features.select_dtypes(include=['number'])
features = features.drop(columns=['Course ID'])

'''
-Estandarizamos los datos.
Esto lo hacemos para que todas las variables tengan la misma escala y no haya variables que dominen el análisis.
Además que se realizó una ejecución previa sin estandarizar y las distancias no eran las adecuadas / congruentes.
'''
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = pd.DataFrame(features_scaled, columns=features.columns)

'''
- Transponemos los datos para que las filas sean los estudiantes y las columnas las variables
'''
features_transposed = features_scaled.T
print("Datos cargados y preprocesados correctamente.")
print("Atributos: ", features_transposed.shape[0])

'''
## Implementamos el método de Ward
'''

'''
-Se inicializa un diccionario inicial que asigna cada fila del conjunto de datos a un cluster individual.
-Se crea una lista / arreglo para almacenar el "historial" de los pasos de enlace usados para construir el dendrograma.
'''
current_clusters = {i: features_transposed.iloc[[i]] for i in range(len(features_transposed))}
linkage_matrix = []

def calculate_delta_sse(ci, cj):
    '''
    ### Función que calcula el cambio en el SSE al combinar dos clusters.
    Esta funcion toma como parámetros dos clusters y calcula el cambio en el SSE al combinarlos.
    '''
    combined = pd.concat([ci, cj])
    sse_combined = ((combined - combined.mean())**2).sum(axis=0).sum()
    sse_i = ((ci - ci.mean())**2).sum(axis=0).sum()
    sse_j = ((cj - cj.mean())**2).sum(axis=0).sum()
    delta_sse = sse_combined - sse_i - sse_j
    return delta_sse

'''
-Iteramos sobre la lista de clusters actuales y calculamos el cambio en el SSE al combinar cada par de clusters.
'''
print('Iniciando proceso de clustering, esto puede tardar unos minutos...')
while len(current_clusters) > 1:
    '''
    -Inicializamos la variable min_delta_sse con un valor infinito para encontrar el par de clusters con el menor cambio en el SSE.
    -Inicializamos la variable best_pair con un par de clusters vacío.
    '''
    min_delta_sse = float('inf')
    best_pair = (None, None)

    cluster_keys = list(current_clusters.keys())
    for i in range(len(cluster_keys)):
        for j in range(i + 1, len(cluster_keys)):
            delta_sse = calculate_delta_sse(current_clusters[cluster_keys[i]], current_clusters[cluster_keys[j]])
            if delta_sse < min_delta_sse:
                min_delta_sse = delta_sse
                best_pair = (cluster_keys[i], cluster_keys[j])

    i, j = best_pair
    new_cluster_index = max(current_clusters.keys()) + 1
    current_clusters[new_cluster_index] = pd.concat([current_clusters[i], current_clusters[j]])
    linkage_matrix.append([i, j, min_delta_sse, len(current_clusters[new_cluster_index])])
    del current_clusters[i], current_clusters[j]

'''
-Guardamos el historial de los pasos de enlace en un archivo de Excel, con las etiquetas correspondientes a los atributos.
'''
cluster_history_sheet.append(['Paso','Cluster 1', 'Cluster 2', 'ΔSSE', 'Tamaño del Cluster Resultante'])
for i, (c1, c2, delta_sse, size) in enumerate(linkage_matrix):
    cluster_history_sheet.append([i + 1, c1, c2, delta_sse, size])
cluster_history.save('CLUSTERS.xlsx')

'''
## Visualización del dendrograma
'''
plt.figure(figsize=(50, 30))
plt.title('Dendrograma con Implementación del Método de Ward')

# Generamos el dendrograma y le pasamos las etiquetas de los atributos
dendrogram(linkage_matrix, labels=features.columns)
plt.xlabel('Atributos')
plt.ylabel('Distancia de Ward')
plt.xticks(rotation=90)

'''
-Generamos las lineas de corte en base a la distancia de Ward y la cantidad de grupos que queremos.
'''
num_clusters = [10,15,20]
colors = ['r', 'g', 'b', 'c', 'm', 'y']
for i, num in enumerate(num_clusters):
    plt.axhline(y=linkage_matrix[-num][2], color=colors[i], linestyle='--', label=f'{num} Clusters')
plt.legend()
plt.savefig('Dendrograma.png', dpi=300)

'''
## Asignación de Clusters
Utilizamos la función fcluster de scipy para asignar los clusters a los estudiantes.
'''
df = higher_education_students_performance_evaluation.data.original
    
#Limiamos el dataframe
df.drop(columns=['Student ID'], inplace=True)
aux_df = df.copy()
aux_df.drop(columns=['OUTPUT Grade'], inplace=True)
aux_df.drop(columns=['Course ID'], inplace=True)


feature_df = aux_df
target_df = df['OUTPUT Grade']

for num in num_clusters:
    cluster_assignments = fcluster(linkage_matrix, num, criterion='maxclust')
    
    



    # Aseguramos que cluster_assignments tenga la misma longitud que la cantidad de estudiantes (filas)
    if len(cluster_assignments) != len(features_transposed):
        raise ValueError(f"Los cluster_assignments deben tener el mismo número de elementos que las filas de features_transposed. Esperado: {len(features_transposed)}, pero obtenido: {len(cluster_assignments)}")

    features_transposed[f'{num} Clusters'] = cluster_assignments

    '''
    En este punto, se guardan los datos con los clusters asignados en un archivo de Excel.
    '''
    features_transposed.to_excel(f'Clusters_{num}.xlsx')
    print(f'Clusters asignados para {num} clusters. Archivo guardado como Clusters_{num}.xlsx')

    '''
    ## Correlación de los atributos
    '''
    corr =feature_df.corrwith(target_df, method='pearson')
    print(corr)
    corr.to_csv('correlations.csv')


    '''
    - Juntamos el atributo, el cluster asignado, la correlacion, la correlacion absoluta
    -Se guarda en un archivo de Excel

    '''
    cluster_corr = pd.DataFrame()
    cluster_corr['Atributo'] = corr.index
    cluster_corr['Cluster'] = cluster_assignments
    cluster_corr['Correlacion'] = corr.values
    cluster_corr['Correlacion Absoluta'] = np.abs(corr.values)

    cluster_corr = cluster_corr.sort_values(by=['Cluster', 'Correlacion Absoluta'], ascending=False)
    cluster_corr = cluster_corr.reset_index(drop=True)

    cluster_corr.to_excel(f'Cluster_Correlaciones_{num}.xlsx')

    '''
    ## Visualización de la correlación de los atributos en los clusters
    Se grafica la correlación de los atributos.
    '''
    feature_list = cluster_corr['Atributo'].tolist()
    print(feature_list)
    aux_corr = original[feature_list].corr()

    
    print(aux_corr)
    plt.figure(figsize=(20, 20))
    sns.heatmap(aux_corr, 
                annot=True, 
                cmap='coolwarm', 
                linewidths=0.5, 
                fmt=".1f",
                annot_kws={"size": 7},
                vmin=-1, vmax=1)
    plt.title(f'Correlación de los atributos con la variable objetivo para {num} clusters')
    plt.xticks(size=7)
    plt.yticks(size=7)

    # Save figure
    plt.savefig(f'Correlacion_entre_atributos{num}.png', dpi=300)

print('Proceso de clustering finalizado.')
    
    




    




