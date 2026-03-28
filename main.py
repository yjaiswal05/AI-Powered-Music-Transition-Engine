import pandas as pd

from database import DatabaseManager
from experiment import run_ab_test
from recommender import MusicRecommender


def build_sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1, "Midnight Echo", "Luna Park", "Pop", 118, 0.78, 0.82],
            [2, "Neon Streets", "Velocity", "Pop", 121, 0.81, 0.79],
            [3, "Cloud Dancer", "Aero", "EDM", 128, 0.90, 0.88],
            [4, "Low Tide", "Harbor", "Indie", 98, 0.52, 0.60],
            [5, "City Pulse", "Drift", "Hip-Hop", 92, 0.64, 0.75],
            [6, "Afterglow", "Mira", "EDM", 126, 0.87, 0.86],
            [7, "Paper Planes", "Northline", "Indie", 102, 0.58, 0.62],
            [8, "Golden Hour", "Mosaic", "Pop", 115, 0.74, 0.80],
        ],
        columns=["song_id", "title", "artist", "genre", "tempo", "energy", "danceability"],
    )


def main() -> None:
    db = DatabaseManager()
    db.init_db()

    songs = build_sample_dataset()
    db.upsert_songs(songs)

    loaded_songs = db.load_songs()
    recommender = MusicRecommender(loaded_songs)

    source_song_id = 1
    recommendations = recommender.recommend(song_id=source_song_id, top_k=3)
    db.log_recommendations(source_song_id, recommendations)

    print("Top recommendations:")
    print(recommendations.to_string(index=False))

    ab_results = run_ab_test(db)
    print("\nA/B test simulation:")
    print(ab_results.to_string(index=False))


if __name__ == "__main__":
    main()
