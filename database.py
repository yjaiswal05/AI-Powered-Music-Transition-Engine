import sqlite3

import pandas as pd


class DatabaseManager:
    def __init__(self, db_path: str = "data/music_reco.db"):
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS songs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    tempo REAL NOT NULL,
                    energy REAL NOT NULL,
                    valence REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    variant TEXT NOT NULL,
                    users INTEGER NOT NULL,
                    avg_session_length REAL NOT NULL,
                    skip_rate REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def store_songs(self, songs_df: pd.DataFrame) -> None:
        required_numeric_columns = ["tempo", "energy", "valence"]
        missing = [col for col in required_numeric_columns if col not in songs_df.columns]
        if missing:
            raise ValueError(f"songs_df is missing required columns: {missing}")

        cleaned = songs_df.copy()
        if "name" not in cleaned.columns:
            cleaned["name"] = [f"Song {i + 1}" for i in range(len(cleaned))]

        cleaned = cleaned[["name", "tempo", "energy", "valence"]].copy()
        cleaned["name"] = cleaned["name"].fillna("Unknown Song").astype(str)
        for col in required_numeric_columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        cleaned = cleaned.dropna(subset=required_numeric_columns)

        with self.connect() as conn:
            conn.execute("DELETE FROM songs;")
            cleaned.to_sql("songs", conn, if_exists="append", index=False)

    def upsert_songs(self, songs_df: pd.DataFrame) -> None:
        self.store_songs(songs_df)

    def store_songs_from_csv(self, csv_path: str) -> None:
        songs_df = pd.read_csv(csv_path)
        self.store_songs(songs_df)

    def load_songs(self) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                "SELECT id, name, tempo, energy, valence FROM songs ORDER BY id;",
                conn,
            )

    def get_average_tempo(self) -> float:
        with self.connect() as conn:
            cursor = conn.execute("SELECT AVG(tempo) AS avg_tempo FROM songs;")
            row = cursor.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def get_highest_energy_songs(self) -> pd.DataFrame:
        query = """
            SELECT id, name, tempo, energy, valence
            FROM songs
            WHERE energy = (SELECT MAX(energy) FROM songs)
            ORDER BY name;
        """
        with self.connect() as conn:
            return pd.read_sql_query(query, conn)

    def get_mood_groups(self) -> pd.DataFrame:
        query = """
            SELECT
                CASE
                    WHEN valence < 0.33 THEN 'Low Mood'
                    WHEN valence < 0.66 THEN 'Neutral Mood'
                    ELSE 'High Mood'
                END AS mood_group,
                COUNT(*) AS song_count,
                ROUND(AVG(valence), 3) AS avg_valence,
                ROUND(AVG(tempo), 2) AS avg_tempo
            FROM songs
            GROUP BY mood_group
            ORDER BY avg_valence;
        """
        with self.connect() as conn:
            return pd.read_sql_query(query, conn)

    def save_experiment_result(
        self, variant: str, users: int, avg_session_length: float, skip_rate: float
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO experiment_results (variant, users, avg_session_length, skip_rate)
                VALUES (?, ?, ?, ?);
                """,
                (variant, users, avg_session_length, skip_rate),
            )
