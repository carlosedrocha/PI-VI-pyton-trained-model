import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

class MusicRecommendation:
    def __init__(self, data_path='dataset_pivi_spotify_data.csv'):
        self.df = pd.read_csv(data_path)
        self.playlist_track_ids = [item for item in self.df['id']]
        self.feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence', 'speechiness', 'loudness']

    def fetch_audio_features(self, track_id):
        row = self.df[self.df['id'] == track_id]
        
        if not row.empty:
            features = row.iloc[:, 3:9].values.flatten()  # Excluir a coluna 'audio_features'
            return features
        else:
            return None

    def normalize_data(self, data):
        valid_data = [features for features in data if features is not None]
        
        if not valid_data:
            print("Nenhuma característica válida disponível para normalização.")
            return None
        
        min_length = min(len(features) for features in valid_data)
        normalized_data = [features[:min_length] for features in valid_data]
        
        if not all(isinstance(value, (int, float)) for features in normalized_data for value in features):
            print("Existem valores não numéricos nos dados. Verifique suas características:")
            for i, features in enumerate(normalized_data):
                print(f"Características da música {i + 1}: {features}")
            return None

        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(normalized_data)
        return normalized_data

    def standardize_and_handle_outliers(self, data):
        # Remover valores None antes de padronizar e tratar outliers
        data = [features for features in data if features is not None]

        if not data:
            print("Nenhuma característica válida disponível para normalização.")
            return None

        # Convertendo para um DataFrame antes de padronizar
        df = pd.DataFrame(data)

        # Tratar NaNs antes da padronização
        df = df.dropna()

        if df.empty:
            print("Todos os dados são NaN após o tratamento. Verifique suas características.")
            return None

        # Padronizar os dados
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(df)

        return standardized_data

    def find_optimal_num_clusters(self, data):
        inertias = []
        silhouettes = []
        max_clusters = min(10, len(data))  # Define um limite superior para o número de clusters

        for num_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10, max_iter=300, init='k-means++')
            cluster_labels = kmeans.fit_predict(data)
            
            # Verificar se há mais de um cluster identificado
            unique_labels = np.unique(cluster_labels)
            if len(unique_labels) > 1:
                inertia = kmeans.inertia_
                silhouette = silhouette_score(data, cluster_labels)
                inertias.append(inertia)
                silhouettes.append(silhouette)
            else:
                inertias.append(np.nan)
                silhouettes.append(np.nan)

        self.plot_elbow_method(range(2, max_clusters + 1), inertias, silhouettes)

    def plot_elbow_method(self, ks, inertias, silhouettes):
        fig = go.Figure()

        # Convertendo range para lista
        ks = list(ks)

        # Plot da Inércia
        fig.add_trace(go.Scatter(x=ks, y=inertias, mode='lines+markers', name='Inércia'))
        fig.update_layout(title='Método do Cotovelo para Inércia', xaxis_title='Número de Clusters', yaxis_title='Inércia')

        # Plot da Silhueta
        fig.add_trace(go.Scatter(x=ks, y=silhouettes, mode='lines+markers', name='Silhueta'))
        fig.update_layout(title='Método do Cotovelo para Silhueta', xaxis_title='Número de Clusters', yaxis_title='Silhueta')

        fig.show()

    def visualize_cluster_characteristics(self, data, labels, num_components):
        df = pd.DataFrame(data, columns=self.feature_names[:num_components])  # Usando nomes reais das features
        df['Cluster'] = labels

        cluster_means = df.groupby('Cluster').mean()

        for feature in self.feature_names[:num_components]:
            fig = px.bar(cluster_means, x=cluster_means.index, y=feature, title=f'Média de {feature} por Cluster')
            fig.update_layout(xaxis_title='Cluster', yaxis_title=f'Média de {feature}')
            fig.show()

    def feature_importance(self, data, labels):
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(data, labels)

        feature_importances = classifier.feature_importances_

        fig = px.bar(x=self.feature_names, y=feature_importances, title='Importância das Características')
        fig.update_layout(xaxis_title='Característica', yaxis_title='Importância')
        fig.show()

    def recommend_songs(self, query_track_id='7r4GcILxwSpjADa4KFbob3'):
        playlist_features = [self.fetch_audio_features(track_id) for track_id in self.playlist_track_ids]
        query_features = self.fetch_audio_features(query_track_id)

        # Remover valores None antes de padronizar e tratar outliers
        playlist_features = [features for features in playlist_features if features is not None]
        query_features = [query_features] if query_features is not None else []

        # Padronizar e tratar outliers
        standardized_data = self.standardize_and_handle_outliers(playlist_features + query_features)

        if standardized_data is None:
            print("Não é possível prosseguir com características de áudio ausentes ou não numéricas.")
            return None, None

        # Encontrar o número ótimo de clusters usando o método do cotovelo
        num_components = 2  # Ajuste conforme necessário
        self.find_optimal_num_clusters(standardized_data)

        # Redução de Dimensionalidade usando PCA
        pca = PCA(n_components=num_components)
        reduced_data = pca.fit_transform(standardized_data)

        # Calcular a matriz de similaridade usando cosseno
        similarity_matrix = cosine_similarity(reduced_data)

        best_kmeans = None
        best_cluster_labels = None
        best_silhouette_score = -1

        for num_clusters in range(2, 11):
            if len(reduced_data) >= num_clusters:
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=20, max_iter=500, init='random')
                cluster_labels = kmeans.fit_predict(reduced_data)

                # Verificar se há mais de um cluster identificado
                unique_labels = np.unique(cluster_labels)
                if len(unique_labels) > 1:
                    silhouette = silhouette_score(reduced_data, cluster_labels)

                    print(f'Número de clusters: {num_clusters}, Silhueta: {silhouette}')

                    if silhouette > best_silhouette_score:
                        best_silhouette_score = silhouette
                        best_kmeans = kmeans
                        best_cluster_labels = cluster_labels
                else:
                    print(f'Número de clusters: {num_clusters}, Não é possível calcular a silhueta com apenas um cluster.')
            else:
                print(f'Número de clusters: {num_clusters}, Não há amostras suficientes para formar {num_clusters} clusters.')

        if best_kmeans is not None:
            self.visualize_cluster_characteristics(reduced_data, best_cluster_labels, num_components)

            inertia = best_kmeans.inertia_
            print(f'Inércia do Modelo: {inertia}')

            query_cluster = best_kmeans.predict([reduced_data[-1]])[0]
            cluster_indices = np.where(best_cluster_labels == query_cluster)[0]

            if len(cluster_indices) == 0:
                print("Nenhuma música semelhante encontrada.")
                return None, inertia

            recommended_songs = [self.playlist_track_ids[i] for i in cluster_indices if i < 90]

            return recommended_songs, inertia
        else:
            print("Não foi possível identificar clusters com mais de um rótulo.")
            return None, None

# Uso da classe MusicRecommendation
music_rec = MusicRecommendation()
recommended_songs, inertia = music_rec.recommend_songs()

if recommended_songs is not None:
    for i in range(len(recommended_songs)):
        print(f'Músicas Recomendadas: {recommended_songs[i]}')
else:
    print("Nenhuma música recomendada.")

print(f'Inércia do Modelo: {inertia}')
