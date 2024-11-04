from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# fetch dataset
higher_education_students_performance_evaluation = fetch_ucirepo(id=856)

# data (as pandas dataframes)
X = higher_education_students_performance_evaluation.data.features
y = higher_education_students_performance_evaluation.data.targets
df = higher_education_students_performance_evaluation.data.original
df.head()



#Limiamos el dataframe
df.drop(columns=['Student ID'], inplace=True)
feature_df = df.drop(columns=['OUTPUT Grade'])
feature_df.drop(columns=['Course ID'])

target_df = df['OUTPUT Grade']
corr =feature_df.corrwith(target_df, method='pearson')
corr.to_csv('correlations.csv')

x_df = df['Cumulative grade point average in the last semester (/4.00)']
y_df = df['OUTPUT Grade']

#Juntamos los dos DF
location_df = pd.concat([x_df, y_df], axis=1)

#Procedemos a hacer el mapa de dispersión
plt.scatter(df["Cumulative grade point average in the last semester (/4.00)"], df['OUTPUT Grade'])
plt.xlabel("Cumulative grade point average in the last semester (/4.00)")
plt.ylabel('OUTPUT Grade')
plt.title(f'Diagrama de dispersión de: Cumulative grade point average in the last semester (/4.00), vs target')
plt.show()

# 1. Cargar y preparar los datos
# Supongamos que tienes un DataFrame `df`
# Estandariza los datos (esto es opcional, pero recomendado para la mayoría de los casos)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 2. Aplicar el método de Ward usando linkage
Z = linkage(X_scaled, method='ward')

# 3. Visualizar el dendrograma
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Dendograma utilizando el Método de Ward")
plt.xlabel("Atributos")
plt.ylabel("Distancia")
plt.show()

# Definir los cortes y almacenar la cantidad de grupos en cada corte
cuts = [2, 5, 8]  # Número de grupos para cada corte
excel_files = []

for num_clusters in cuts:
    # Generar etiquetas de clusters para el corte actual
    cluster_labels = fcluster(Z, num_clusters, criterion='maxclust')

    # Crear un nuevo DataFrame para los datos etiquetados
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels

    # Crear un archivo de Excel para este corte
    filename = f"clusters_{num_clusters}_grupos.xlsx"
    with pd.ExcelWriter(filename) as writer:
        # Dividir los datos por grupo y escribir cada grupo en una hoja separada
        for cluster_id in range(1, num_clusters + 1):
            # Filtrar los objetos del grupo actual
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            # Escribir en una hoja del archivo de Excel
            cluster_data.to_excel(writer, sheet_name=f'Grupo_{cluster_id}', index=False)

    # Guardar el nombre del archivo para referencia
    excel_files.append(filename)

print("Archivos de Excel generados:", excel_files)

# Visualizar el dendrograma
plt.figure(figsize=(12, 8))
dendrogram(Z)
plt.title("Dendograma con cortes")
plt.xlabel("Atributos")
plt.ylabel("Distancia")

# Añadir líneas de corte para 5, 8 y 12 grupos
cut_levels = [2, 5, 8]  # número de grupos deseado en cada corte
for num_clusters in cut_levels:
    # Calcular la altura a la que se debe cortar para obtener num_clusters
    max_d = Z[-(num_clusters-1), 2]
    plt.axhline(y=max_d, color='r', linestyle='--', label=f'{num_clusters} grupos')

plt.legend()
plt.show()

hierarchical_cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hierarchical_cluster.fit_predict(location_df)

plt.scatter(x_df, y_df, c=labels)
plt.xlabel("Cumulative grade point average in the last semester (/4.00)")
plt.ylabel('OUTPUT Grade')
plt.show()

# Probar varios números de grupos
num_clusters_options = range(2, 15)
best_score = -1
best_num_clusters = None

for num_clusters in num_clusters_options:
    # Crear clusters para el número actual de grupos
    cluster_labels = fcluster(Z, num_clusters, criterion='maxclust')

    # Calcular el coeficiente de silueta
    score = silhouette_score(X_scaled, cluster_labels)

    # Verificar si es el mejor número de grupos
    if score > best_score:
        best_score = score
        best_num_clusters = num_clusters

print(f"El mejor número de grupos es: {best_num_clusters} con un coeficiente de silueta de: {best_score:.2f}")



