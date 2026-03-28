import numpy as np
import pandas as pd


class MusicRecommender:
    def __init__(self, songs_df: pd.DataFrame):
        required_columns = ["name", "tempo", "energy", "valence"]
        missing = [col for col in required_columns if col not in songs_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.songs_df = songs_df[required_columns].reset_index(drop=True).copy()
        self.feature_columns = ["tempo", "energy", "valence"]
        self.feature_matrix = self._build_feature_matrix()

    def _build_feature_matrix(self) -> np.ndarray:
        features = self.songs_df[self.feature_columns].astype(float).to_numpy()
        means = features.mean(axis=0)
        stds = features.std(axis=0)
        stds[stds == 0] = 1.0
        return (features - means) / stds

    @staticmethod
    def _cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query_vector)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        matrix_norms[matrix_norms == 0] = 1e-8
        if query_norm == 0:
            query_norm = 1e-8
        return (matrix @ query_vector) / (matrix_norms * query_norm)

    def get_similar_songs(self, song_name: str, top_n: int = 5) -> pd.DataFrame:
        matches = self.songs_df.index[self.songs_df["name"].str.lower() == song_name.lower()].tolist()
        if not matches:
            raise ValueError(f"Song '{song_name}' not found in dataset.")

        source_idx = matches[0]
        source_vector = self.feature_matrix[source_idx]
        similarity = self._cosine_similarity(source_vector, self.feature_matrix)
        
        # Exclude the source song and any songs with the exact same name
        similarity[source_idx] = -np.inf
        same_name_mask = self.songs_df["name"].str.lower() == song_name.lower()
        similarity[same_name_mask] = -np.inf

        # Sort all indices by highest similarity
        sorted_indices = np.argsort(similarity)[::-1]
        
        # Create full recommendations dataframe
        recommendations = self.songs_df.loc[sorted_indices, ["name"]].copy()
        recommendations["similarity"] = similarity[sorted_indices]
        
        # Drop any duplicate song names, keeping the one with the highest similarity
        recommendations = recommendations.drop_duplicates(subset=["name"], keep="first")
        
        # Take only the top N unique results
        return recommendations.head(top_n).reset_index(drop=True)
