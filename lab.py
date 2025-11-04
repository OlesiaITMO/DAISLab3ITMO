# ==============================================
# Лабораторная работа 2: Рекомендательные системы
# Датасет: Spotify Million Playlist Dataset
# Папка: C:\Users\Olesia\Downloads\DAIS\data
# ==============================================

# ---------------------------
# 1. Импорт библиотек
# ---------------------------
import os
import glob
import json
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import ElasticNet
from tqdm import tqdm

# ---------------------------
# 2. Загрузка первых 100 JSON-файлов
# ---------------------------
data_folder = r"C:\Users\Olesia\Downloads\DAIS\data"
json_files = glob.glob(os.path.join(data_folder, "*.json"))

print("Найдено JSON-файлов:", len(json_files))

tracks_list = []

for file_path in tqdm(json_files[:100], desc="Обработка JSON-файлов"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for pl in data.get('playlists', []):
                pid = pl['pid']
                for track in pl.get('tracks', []):
                    tracks_list.append({
                        'playlist_id': pid,
                        'track_name': track['track_name'],
                        'track_uri': track['track_uri'],
                        'artist_name': track['artist_name'],
                        'album_name': track['album_name']
                    })
    except MemoryError:
        print("MemoryError при чтении файла:", file_path)
    except Exception as e:
        print("Ошибка при обработке файла:", file_path, e)

df = pd.DataFrame(tracks_list)
print("Общее количество треков:", len(df))
print("Пример данных:")
display(df.head())

# ---------------------------
# 3. EDA
# ---------------------------
print("Количество уникальных треков:", df['track_uri'].nunique())
print("Количество уникальных артистов:", df['artist_name'].nunique())
print("Количество плейлистов:", df['playlist_id'].nunique())

# Пропуски и дубликаты
print("\nПропуски по колонкам:")
print(df.isna().sum())
print("\nКоличество дубликатов:", df.duplicated().sum())

# Распределение числа треков на плейлист
playlist_sizes = df.groupby('playlist_id').size()
plt.figure(figsize=(10, 5))
sns.histplot(playlist_sizes, bins=50)
plt.title("Распределение числа треков в плейлистах")
plt.xlabel("Число треков")
plt.ylabel("Количество плейлистов")
plt.show()

# Топ-20 популярных треков
top_tracks = df['track_uri'].value_counts().head(20)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_tracks.values, y=df[df['track_uri'].isin(top_tracks.index)]['track_name'].unique())
plt.title("Топ-20 популярных треков")
plt.show()

# ---------------------------
# 4. Train/Test Split
# ---------------------------
playlist_ids = df['playlist_id'].unique()
train_ids, test_ids = train_test_split(playlist_ids, test_size=0.2, random_state=42)
train_df = df[df['playlist_id'].isin(train_ids)]
test_df = df[df['playlist_id'].isin(test_ids)]

# ---------------------------
# 5. MostPop Recommender
# ---------------------------
most_pop_tracks = train_df['track_uri'].value_counts()
top_10_most_pop = most_pop_tracks.head(10).index.tolist()

# ---------------------------
# 6. UserKNN и ItemKNN
# ---------------------------
playlist_track_matrix = train_df.pivot_table(index='playlist_id', columns='track_uri', aggfunc='size', fill_value=0)

user_sim = cosine_similarity(playlist_track_matrix)
item_sim = cosine_similarity(playlist_track_matrix.T)
user_sim_df = pd.DataFrame(user_sim, index=playlist_track_matrix.index, columns=playlist_track_matrix.index)
item_sim_df = pd.DataFrame(item_sim, index=playlist_track_matrix.columns, columns=playlist_track_matrix.columns)

def userknn_recommend(pid, top_n=10):
    if pid not in user_sim_df.index:
        return []
    sims = user_sim_df[pid].sort_values(ascending=False)
    neighbors = sims.index[1:6]
    recs = playlist_track_matrix.loc[neighbors].sum().sort_values(ascending=False)
    recs = [t for t in recs.index if playlist_track_matrix.loc[pid, t]==0]
    return recs[:top_n]

def itemknn_recommend(pid, top_n=10):
    tracks = playlist_track_matrix.loc[pid]
    recs = Counter()
    for t in tracks[tracks>0].index:
        sim_scores = item_sim_df[t].sort_values(ascending=False)
        for s_t, score in sim_scores[1:6].items():
            recs[s_t] += score
    recs = [t for t,_ in recs.most_common(top_n)]
    return recs

# ---------------------------
# 7. SLIM Recommender
# ---------------------------
X = playlist_track_matrix.values.T
slim_model = ElasticNet(alpha=1.0, l1_ratio=0.5, positive=True, max_iter=200)
slim_model.fit(X, X)
track_sim_matrix = cosine_similarity(X)
track_sim_matrix = pd.DataFrame(track_sim_matrix, index=playlist_track_matrix.columns, columns=playlist_track_matrix.columns)

def slim_recommend(track_uri, top_n=10):
    if track_uri not in track_sim_matrix.columns:
        return []
    sims = track_sim_matrix[track_uri].sort_values(ascending=False)
    return sims.index[1:top_n+1].tolist()

# ---------------------------
# 8. EASE (simplified)
# ---------------------------
X_bin = playlist_track_matrix.copy()
X_bin[X_bin>0] = 1
lambda_reg = 0.5
G = X_bin.T.dot(X_bin) + lambda_reg * np.eye(X_bin.shape[1])
B = np.linalg.inv(G)
B = B / -np.diag(B)
np.fill_diagonal(B, 0)
ease_sim_df = pd.DataFrame(B, index=X_bin.columns, columns=X_bin.columns)

def ease_recommend(pid, top_n=10):
    tracks = playlist_track_matrix.loc[pid]
    recs = Counter()
    for t in tracks[tracks>0].index:
        sim_scores = ease_sim_df[t].sort_values(ascending=False)
        for s_t, score in sim_scores[:10].items():
            recs[s_t] += score
    recs = [t for t,_ in recs.most_common(top_n)]
    return recs

# ---------------------------
# 9. Вывод персональных рекомендаций
# ---------------------------
example_pid = np.random.choice(test_df['playlist_id'].unique(), 1)[0]
print(f"\n=== Персональные рекомендации для плейлиста {example_pid} ===")
actual_tracks = test_df[test_df['playlist_id']==example_pid]['track_uri'].tolist()
display(df[df['track_uri'].isin(actual_tracks)][['track_name','artist_name']].drop_duplicates().head(5))

methods = {
    "MostPop": top_10_most_pop,
    "SLIM": slim_recommend,
    "EASE": ease_recommend,
    "UserKNN": userknn_recommend,
    "ItemKNN": itemknn_recommend
}

for name, rec in methods.items():
    print(f"\n{name} рекомендации:")
    if callable(rec):
        rec_tracks = rec(example_pid, top_n=10)
    else:
        rec_tracks = rec[:10]
    display(df[df['track_uri'].isin(rec_tracks)][['track_name','artist_name']].drop_duplicates())
