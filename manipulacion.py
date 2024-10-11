import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import csv


# 1. Cargar el dataset
rows = []
max_rows = 1000  # Limitar a 1000 registros
row_count = 0

try:
    with open('YT_Videos_Comments.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row_count >= max_rows:
                break  # Detener después de leer 1000 registros
            rows.append(row)
            row_count += 1
except Exception as e:
    print(f"Error leyendo el archivo CSV: {e}")

df = pd.DataFrame(rows)
print(f"CSV cargado correctamente con {len(df)} registros")


# 2. Preprocesamiento: Limpiar y transformar datos

#Limpieza de datos: eliminar duplicados y nulos
df = df.drop_duplicates()
df = df.dropna()
print("Limpieza completada")

# Combinamos los campos de texto que pueden ser relevantes para el análisis, como el título del video, descripción, y comentarios
df['text_data'] = df['Video Title'] + ' ' + df['Video Description'] + ' ' + df['Comment (Displayed)']

# Usamos TF-IDF para transformar los datos de texto en vectores numéricos
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)  # Reducimos a 1000 features para mayor eficiencia
X_tfidf = vectorizer.fit_transform(df['text_data'])
print("Datos transformados")

# 3. Algoritmos de Agrupamiento

## A. K-Means Clustering (Agrupamiento particional)
kmeans = KMeans(n_clusters=5, random_state=42)  # Ajusta el número de clústeres según tus datos
df['kmeans_labels'] = kmeans.fit_predict(X_tfidf)

## B. DBSCAN Clustering (Agrupamiento basado en densidad)
dbscan = DBSCAN(eps=0.5, min_samples=10, metric='cosine')
df['dbscan_labels'] = dbscan.fit_predict(X_tfidf)
print("Algoritmos aplicados")

# 4. Detección de Anomalías con Isolation Forest

iso_forest = IsolationForest(contamination=0.01, random_state=42)  # Ajusta la tasa de contaminación (proporción de anomalías)
df['anomaly'] = iso_forest.fit_predict(X_tfidf)

# Marcar las anomalías identificadas (-1 indica anomalías)
df['anomaly'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
print("Anomalias detectadas")

# 5. Visualización de Clústeres y Anomalías

# Para visualizar los clústeres, usaremos PCA para reducir a 2 dimensiones
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_tfidf.toarray())

# Añadir las columnas de las componentes principales
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

# A. Visualización de los Clústeres con K-Means
plt.figure(figsize=(10,6))
sns.scatterplot(x='pca1', y='pca2', hue='kmeans_labels', data=df, palette='Set1', legend='full', alpha=0.6)
plt.title('Clusters formados por K-Means')
plt.show()

# B. Visualización de los Clústeres con DBSCAN
plt.figure(figsize=(10,6))
sns.scatterplot(x='pca1', y='pca2', hue='dbscan_labels', data=df, palette='Set1', legend='full', alpha=0.6)
plt.title('Clusters formados por DBSCAN')
plt.show()

# C. Visualización de las Anomalías detectadas
plt.figure(figsize=(10,6))
sns.scatterplot(x='pca1', y='pca2', hue='anomaly', data=df, palette={'Normal': 'blue', 'Anomaly': 'red'}, alpha=0.6)
plt.title('Anomalias detectadas por Isolation Forest')
plt.show()