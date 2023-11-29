from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

class MusicClusterAPI:
    def __init__(self):
        # Carregue o modelo e os objetos de pré-processamento aqui
        self.model = None
        self.scaler = None
        self.pca = None
        self.selected_feature_names = None

    def load_model(self, model_path='modelo_kmeans.pkl', preprocessing_path='preprocessing_objects.pkl'):
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)

        with open(preprocessing_path, 'rb') as preprocessing_file:
            preprocessing_objects = pickle.load(preprocessing_file)
            self.scaler = preprocessing_objects['scaler']
            self.pca = preprocessing_objects['pca']
            self.selected_feature_names = preprocessing_objects['selected_feature_names']

    def recommend_songs(self, input_data, model_save_path='modelo_kmeans.pkl', preprocessing_save_path='preprocessing_objects.pkl'):
        # Padronizar e reduzir dimensionalidade dos dados de entrada usando preprocess_input
        processed_input = self.preprocess_input(input_data)

        if processed_input is None or processed_input.size == 0 or np.any(processed_input):
            print("Não é possível prosseguir com características de áudio ausentes ou não numéricas.")
            return None, None

        
        # Carregar objetos de pré-processamento com pickle
        with open(preprocessing_save_path, 'rb') as preprocessing_file:
            preprocessing_objects = pickle.load(preprocessing_file)

        pca = preprocessing_objects['pca']

        # Carregar o modelo com pickle
        with open(model_save_path, 'rb') as model_file:
            best_kmeans = pickle.load(model_file)

        query_cluster = best_kmeans.predict(processed_input)[0]
        cluster_indices = np.where(best_kmeans.labels_ == query_cluster)[0]

        if len(cluster_indices) == 0:
            print("Nenhuma música semelhante encontrada.")
            return None, None

        recommended_songs = [self.playlist_track_ids[i] for i in cluster_indices if i < 90]

        return recommended_songs

    def preprocess_input(self, input_data):
        if not self.scaler or not self.pca or not self.selected_feature_names:
            raise RuntimeError("Model and preprocessing objects not loaded.")

        # Extrair as características relevantes na ordem correta
        input_features = [input_data[feature] for feature in self.selected_feature_names]

        # Criar um array numpy 2D com uma única amostra
        input_array = np.array(input_features).reshape(1, -1)

        # Padronizar as características usando o scaler carregado
        scaled_input = self.scaler.transform(input_array)

        # Reduzir a dimensionalidade usando o PCA carregado
        reduced_input = self.pca.transform(scaled_input)

        return reduced_input

# Instancie a classe MusicClusterAPI
music_api = MusicClusterAPI()

# Carregue o modelo e objetos de pré-processamento
music_api.load_model()

# Rota para a API
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Obtenha os dados do JSON da solicitação
        input_data = request.get_json()

        # Pré-processamento dos dados
        processed_input = music_api.preprocess_input(input_data)

        # Obtenha recomendações do modelo
        recommendations, inertia = music_api.recommend_songs(processed_input)

        if recommendations is not None:
            response = {
                'status': 'success',
                'recommendations': recommendations,
                'inertia': inertia
            }
        else:
            response = {'status': 'error', 'message': 'Não foi possível fazer recomendações.'}

    except Exception as e:
        response = {'status': 'error', 'message': str(e)}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
