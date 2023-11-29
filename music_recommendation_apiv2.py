from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Carregar objetos necessários
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('pca.pkl', 'rb') as pca_file:
    pca = pickle.load(pca_file)

with open('kmeans_model.pkl', 'rb') as kmeans_model_file:
    kmeans_model = pickle.load(kmeans_model_file)

with open('selected_feature_names.pkl', 'rb') as selected_feature_names_file:
    selected_feature_names = pickle.load(selected_feature_names_file)

def preprocess_input(data):
    # Converter o objeto JSON para uma lista de características
    input_data = [data[feature] for feature in selected_feature_names]

    # Padronizar e aplicar a redução de dimensionalidade
    standardized_data = scaler.transform([input_data])
    reduced_data = pca.transform(standardized_data)

    return reduced_data

def predict_cluster(input_data):
    # Prever o cluster usando o modelo KMeans
    cluster = kmeans_model.predict(input_data)[0]
    return cluster

def get_recommended_songs(input_data, cluster, num_recommendations=5):
    # Obter índices das músicas no mesmo cluster
    cluster_indices = np.where(kmeans_model.labels_ == cluster)[0]

    # Obter as posições dos pontos no espaço reduzido para as músicas do mesmo cluster
    cluster_positions = pca.transform(scaler.transform([input_data.fetch_audio_features(track_id) for track_id in music_rec.playlist_track_ids if track_id is not None]))

    # Calcular a distância euclidiana entre a entrada e cada ponto do cluster
    distances = np.linalg.norm(cluster_positions - input_data, axis=1)

    # Obter os índices das músicas mais próximas
    closest_song_indices = np.argsort(distances)[:num_recommendations]

    # Retornar as URIs das músicas recomendadas
    recommended_songs = [input_data.playlist_track_ids[cluster_indices[i]] for i in closest_song_indices]

    return recommended_songs

@app.route('/recommend', methods=['POST'])
def recommend_songs():
    try:
        # Receber dados do corpo da requisição
        data = request.get_json()

        # Pré-processar os dados
        input_data = preprocess_input(data)

        # Prever o cluster
        cluster = predict_cluster(input_data)

        # Obter músicas recomendadas do mesmo cluster
        recommended_songs = get_recommended_songs(cluster)

        return jsonify({"cluster": cluster, "recommended_songs": recommended_songs})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
