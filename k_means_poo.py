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
        self.feature_names = list(self.df.select_dtypes(include=['number']).columns)  # Selecionar todas as colunas numéricas

    def fetch_audio_features(self, track_id):
        row = self.df[self.df['id'] == track_id]
        
        if not row.empty:
            features = row[self.feature_names].values.flatten()
            return features
        else:
            return None

    def preprocess_data(self, data):
        # Remover valores None antes de padronizar e tratar outliers
        data = [features for features in data if features is not None]

        if not data:
            print("Nenhuma característica válida disponível para normalização.")
            return None

        # Convertendo para um DataFrame antes de tratar outliers
        df = pd.DataFrame(data, columns=self.feature_names)

        # Tratar NaNs antes da padronização
        df = df.dropna()

        if df.empty:
            print("Todos os dados são NaN após o tratamento. Verifique suas características.")
            return None

        # Tratar outliers
        df = self.handle_outliers(df)

        # Padronizar os dados
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(df)

        return standardized_data

    def handle_outliers(self, df):
        # Implemente aqui a lógica para lidar com outliers, por exemplo, usando IQR ou Z-score.
        # Exemplo com IQR:
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df_no_outliers = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        return df_no_outliers

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

    def select_best_features(self, data, num_features=5):
        # Utilizando RandomForestClassifier para encontrar a importância das features
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        # Obtendo rótulos (labels) equivalentes a self.feature_names
        labels = np.arange(len(data))

        # Adicionando esta verificação para garantir que os tamanhos correspondam
        if len(labels) != len(data):
            print("Erro: O número de rótulos não coincide com o número de amostras.")
            return None, None

        classifier.fit(data, labels)

        # Obtendo a importância das características
        feature_importances = classifier.feature_importances_

        # Selecionando os índices das melhores características
        selected_feature_indices = np.argsort(feature_importances)[::-1][:num_features]

        # Selecionando as melhores features
        selected_data = data[:, selected_feature_indices]
        selected_feature_names = np.array(self.feature_names)[selected_feature_indices]

        return selected_data, selected_feature_names

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

    def select_best_features(self, data, num_features=5):
        # Utilizando RandomForestClassifier para encontrar a importância das features
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        # Obtendo rótulos (labels) equivalentes a self.feature_names
        labels = np.arange(len(data))

        # Adicionando esta verificação para garantir que os tamanhos correspondam
        if len(labels) != len(data):
            print("Erro: O número de rótulos não coincide com o número de amostras.")
            return None, None

        classifier.fit(data, labels)

        # Obtendo a importância das características
        feature_importances = classifier.feature_importances_

        # Selecionando os índices das melhores características
        selected_feature_indices = np.argsort(feature_importances)[::-1][:num_features]

        # Selecionando as melhores features
        selected_data = data[:, selected_feature_indices]
        selected_feature_names = np.array(self.feature_names)[selected_feature_indices]

        return selected_data, selected_feature_names

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

        fig = px.scatter_3d(df, x=df.columns[0], y=df.columns[1], z=df.columns[2], color='Cluster',
                            title=f'Clusters com Detalhes das Características',
                            labels={df.columns[0]: df.columns[0], df.columns[1]: df.columns[1], df.columns[2]: df.columns[2], 'Cluster': 'Cluster'})

        # Adicionar informações detalhadas sobre cada item no cluster
        for index, row in df.iterrows():
            item_info = f"Track ID: {self.playlist_track_ids[index]}<br>"
            for feat in self.feature_names[:num_components]:
                item_info += f"{feat}: {row[feat]:.4f}<br>"
            fig.update_layout(scene=dict(annotations=[dict(x=row[df.columns[0]], y=row[df.columns[1]], z=row[df.columns[2]], text=item_info,
                                                            showarrow=False, xshift=10)]))

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
        standardized_data = self.preprocess_data(playlist_features + query_features)

        if standardized_data is None:
            print("Não é possível prosseguir com características de áudio ausentes ou não numéricas.")
            return None, None

        # Selecionar as melhores características usando RandomForestClassifier
        selected_data, selected_feature_names = self.select_best_features(standardized_data)

        # Exibir as features escolhidas
        print(f"Features Escolhidas: {', '.join(selected_feature_names)}")

        # Encontrar o número ótimo de clusters usando o método do cotovelo
        num_components = 2  # Ajuste conforme necessário
        self.find_optimal_num_clusters(selected_data)

        # Redução de Dimensionalidade usando PCA após a seleção de características
        pca = PCA(n_components=num_components)
        reduced_data = pca.fit_transform(selected_data)

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
            # self.plot_cluster_characteristics(reduced_data, best_cluster_labels, num_components)

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
