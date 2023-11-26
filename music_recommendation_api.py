from flask import Flask, request, jsonify
from sklearn.metrics import silhouette_score
from flask_restful import Resource, Api
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
api = Api(app)

class MusicRecommendationAPI(Resource):
    def __init__(self):
        # Carregar o modelo salvo
        self.model = self.load_model()

    def load_model(self, filename='music_recommendation_model.pkl'):
        try:
            with open(filename, 'rb') as file:
                model = pickle.load(file)
            print(f"Modelo carregado de {filename}.")
            return model
        except FileNotFoundError:
            print(f"Arquivo {filename} não encontrado. Certifique-se de ter treinado um modelo antes de iniciar a API.")
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

    def recommend_songs(self, query_track_id):
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
        selected_data, _ = self.select_best_features(standardized_data)

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
    def post(self):
        data = request.get_json()

        if 'query_track_id' not in data:
            return {'error': 'Missing query_track_id in the request data'}, 400

        query_track_id = data['query_track_id']

        if self.model is not None:
            # Coloque aqui a lógica para processar os dados da mesma forma que o modelo processou
            # Use self.model para fazer predições

            # Exemplo: preprocessamento
            playlist_features = [self.fetch_audio_features(track_id) for track_id in self.playlist_track_ids]
            query_features = self.fetch_audio_features(query_track_id)

            # Exemplo: selecionar melhores características
            standardized_data = self.preprocess_data(playlist_features + [query_features])
            selected_data, _ = self.select_best_features(standardized_data)

            # Exemplo: redução de dimensionalidade
            num_components = 2
            pca = PCA(n_components=num_components)
            reduced_data = pca.fit_transform(selected_data)

            # Exemplo: calcular matriz de similaridade usando cosseno
            similarity_matrix = cosine_similarity(reduced_data)

            # Exemplo: fazer predição usando o modelo treinado
            predicted_labels = self.model.predict(reduced_data)

            # Exemplo: processar os resultados e retornar a resposta
            result = {
                'recommended_songs': [],
                'inertia': -1  # Substitua pelo valor real
            }

            return result
        else:
            return {'error': 'Model not loaded'}, 500

api.add_resource(MusicRecommendationAPI, '/recommendation')

if __name__ == '__main__':
    app.run(debug=True)
