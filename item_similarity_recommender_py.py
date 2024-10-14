import pandas as pd

class item_similarity_recommender_py:
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
    
    # Створюємо матрицю взаємодій користувачів і пісень
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        # Унікальні пісні та користувачі
        all_songs = train_data[item_id].unique()
        all_users = train_data[user_id].unique()

        # Створюємо словники для пісень
        self.songs_dict = {song: idx for idx, song in enumerate(all_songs)}
        self.rev_songs_dict = {idx: song for song, idx in self.songs_dict.items()}

        # Створюємо матрицю пісень і користувачів
        self.cooccurence_matrix = np.zeros((len(all_songs), len(all_songs)))

        for song in all_songs:
            users_listen_song = train_data[train_data[item_id] == song][user_id].unique()

            for user in users_listen_song:
                other_songs = train_data[train_data[user_id] == user][item_id].unique()
                for other_song in other_songs:
                    if song != other_song:
                        self.cooccurence_matrix[self.songs_dict[song], self.songs_dict[other_song]] += 1

    # Генерація рекомендацій для користувача на основі схожості
    def recommend(self, user):
        user_songs = self.train_data[self.train_data[self.user_id] == user][self.item_id].unique()
        user_songs_indices = [self.songs_dict[song] for song in user_songs]
        
        # Рахуємо схожість користувацьких пісень з іншими
        similarity_scores = np.zeros(len(self.songs_dict))
        for user_song_idx in user_songs_indices:
            similarity_scores += self.cooccurence_matrix[user_song_idx]

        # Знаходимо пісні з найвищими балами схожості
        song_indices = np.argsort(similarity_scores)[::-1]
        
        # Фільтруємо пісні, які користувач уже слухав
        recommended_songs = []
        for idx in song_indices:
            song = self.rev_songs_dict[idx]
            if song not in user_songs:
                recommended_songs.append(song)
            if len(recommended_songs) >= 10:  # Обмежимо рекомендації топ-10
                break

        return recommended_songs


    # Знаходимо пісні, схожі на дану
    def get_similar_items(self, item_list):
        # Індекси для даних пісень
        item_indices = [self.songs_dict[item] for item in item_list]
        
        # Ініціалізуємо суму схожостей
        similarity_scores = np.zeros(len(self.songs_dict))

        # Підсумовуємо всі рядки, що відповідають пісням з item_list
        for idx in item_indices:
            similarity_scores += self.cooccurence_matrix[idx]

        # Сортуємо індекси пісень за їхньою схожістю
        similar_item_indices = np.argsort(similarity_scores)[::-1]

        # Повертаємо таблицю з топ-10 схожими піснями, їх Score і Rank
        similar_items = []
        rank = 1
        for idx in similar_item_indices:
            song = self.rev_songs_dict[idx]
            score = similarity_scores[idx]
            if song not in item_list and score > 0:
                similar_items.append({
                    'Song': song,
                    'Score': score,
                    'Rank': rank
                })
                rank += 1
            if len(similar_items) >= 10:
                break
        
        # Створюємо DataFrame для результату
        df_similar_items = pd.DataFrame(similar_items)
        return df_similar_items