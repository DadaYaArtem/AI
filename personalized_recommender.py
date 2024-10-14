# Наслідування від базового класу item_similarity_recommender_py
class personalized_recommender(item_similarity_recommender_py):
    
    # Метод для рекомендації однієї пісні
    def recommend_one_song(self, user_id):
        # Отримуємо всі пісні користувача
        user_songs = self.get_user_items(user_id)
        
        # Отримуємо всі пісні з тренувальних даних
        all_songs = self.get_all_items_train_data()
        
        # Створюємо матрицю співпадінь
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        # Підраховуємо схожість для всіх пісень (сума рядків в матриці співпадінь)
        similarity_scores = cooccurence_matrix.sum(axis=0)
        
        # Знаходимо індекс пісні з найвищою оцінкою
        top_song_index = similarity_scores.argmax()
        
        # Повертаємо пісню з найвищим значенням схожості
        return all_songs[top_song_index]

    # Метод для побудови матриці співпадінь (для прослуханих пісень користувача)
    def construct_cooccurence_matrix(self, user_songs, all_songs):
        # Створюємо порожню матрицю співпадінь
        cooccurence_matrix = np.zeros((len(user_songs), len(all_songs)))
        
        # Створюємо матрицю співпадінь для кожної пари "пісня користувача" - "інша пісня"
        for i, user_song in enumerate(user_songs):
            for j, song in enumerate(all_songs):
                # Якщо інші користувачі слухали обидві пісні (користувацьку і поточну), збільшуємо значення
                if len(self.train_data[(self.train_data[self.item_id] == user_song) & 
                                       (self.train_data[self.item_id] == song)]) > 0:
                    cooccurence_matrix[i, j] = 1
        return cooccurence_matrix

    # Метод для отримання всіх пісень з тренувальних даних
    def get_all_items_train_data(self):
        return self.train_data[self.item_id].unique()

    # Метод для отримання пісень, які слухав користувач
    def get_user_items(self, user_id):
        return self.train_data[self.train_data[self.user_id] == user_id][self.item_id].unique()
