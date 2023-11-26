import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time

# Adicione suas credenciais da API do Spotify
SPOTIPY_CLIENT_ID = '36d905a4ba3b40949117a4de067f023e'
SPOTIPY_CLIENT_SECRET = 'd79e4e747de1433bb6ffe5365754ede5'

username = 'carlosedrocha'
playlist_id = '6gWVEuUQPDvn8TjKaLTcGZ'  # Use o ID da sua playlist

# Autenticação
client_credentials_manager = SpotifyClientCredentials(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Obter todas as músicas da playlist diretamente
source_playlist = sp.playlist_tracks(playlist_id)
tracks = source_playlist['items']

# Extrair IDs das músicas
track_ids = [song['track']['id'] for song in tracks if song['track']['id'] is not None]

# Inicializar a lista de recursos
features = []

# Definir o tamanho do lote
batch_size = 100

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time

# Adicione suas credenciais da API do Spotify
SPOTIPY_CLIENT_ID = '36d905a4ba3b40949117a4de067f023e'
SPOTIPY_CLIENT_SECRET = 'd79e4e747de1433bb6ffe5365754ede5'

username = 'carlosedrocha'
playlist_id = '6gWVEuUQPDvn8TjKaLTcGZ'  # Use o ID da sua playlist

# Autenticação
client_credentials_manager = SpotifyClientCredentials(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Obter todas as músicas da playlist diretamente
source_playlist = sp.playlist_tracks(playlist_id)
tracks = source_playlist['items']

# Extrair IDs das músicas
track_ids = [song['track']['id'] for song in tracks if song['track']['id'] is not None]

# Inicializar a lista de recursos
features = []

# Definir o tamanho do lote
batch_size = 100

# Iterar sobre as páginas de resultados
while source_playlist['next']:
    source_playlist = sp.next(source_playlist)
    tracks.extend(source_playlist['items'])
    track_ids.extend([song['track']['id'] for song in source_playlist['items'] if song['track']['id'] is not None])

# Iterar sobre os lotes de IDs das músicas
for i in range(0, len(track_ids), batch_size):
    # Obter os recursos de áudio para o lote atual
    batch_ids = track_ids[i:i + batch_size]
    audio_features_list = sp.audio_features(batch_ids)

    # Atribuir rótulos únicos para cada música e adicionar à lista de recursos
    for j, audio_features in enumerate(audio_features_list):
        if audio_features is not None:
            audio_features['label'] = len(features) + j
            features.append(audio_features)

    # Aguardar antes da próxima solicitação
    time.sleep(2)  # Pausa por 2 segundos entre as solicitações

# Converter os recursos em um DataFrame
df = pd.DataFrame(features)

# Salvar o DataFrame em um arquivo CSV, substituindo o arquivo existente
df.to_csv('/Users/carloseduardo/Dev/PI-VI/python/dataset_pivi_spotify_data.csv', mode='w', header=True, index=False)
