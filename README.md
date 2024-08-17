# AIpowered
# ai_model.py
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class SchoolRecommender:
    def __init__(self, data):
        self.data = pd.DataFrame(data)
        self.model = self._train_model()

    def _train_model(self):
        features = self.data[['location', 'academic_performance', 'facilities']]
        model = NearestNeighbors(n_neighbors=3)
        model.fit(features)
        return model

    def recommend(self, user_preferences):
        user_data = pd.DataFrame([user_preferences])
        distances, indices = self.model.kneighbors(user_data)
        recommended_schools = self.data.iloc[indices[0]]
        return recommended_schools.to_dict(orient='records')

# Example data and usage
data = [
    {'school_name': 'School A', 'location': 1, 'academic_performance': 85, 'facilities': 8},
    {'school_name': 'School B', 'location': 2, 'academic_performance': 90, 'facilities': 7},
    {'school_name': 'School C', 'location': 1, 'academic_performance': 80, 'facilities': 6},
]

recommender = SchoolRecommender(data)
user_preferences = {'location': 1, 'academic_performance': 85, 'facilities': 7}
recommendations = recommender.recommend(user_preferences)
print(recommendations)
