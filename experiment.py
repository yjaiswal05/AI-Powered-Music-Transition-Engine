from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExperimentConfig:
    num_users: int = 1000
    max_steps_per_user: int = 20
    base_skip_probability: float = 0.15
    dissimilarity_skip_weight: float = 0.70
    continue_after_skip_probability: float = 0.35
    continue_after_listen_probability: float = 0.85
    random_seed: int = 42


def _sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Midnight Echo", 118, 0.78, 0.66],
            ["Neon Streets", 121, 0.81, 0.72],
            ["Cloud Dancer", 128, 0.90, 0.84],
            ["Low Tide", 98, 0.52, 0.41],
            ["City Pulse", 92, 0.64, 0.55],
            ["Afterglow", 126, 0.87, 0.81],
            ["Paper Planes", 102, 0.58, 0.48],
            ["Golden Hour", 115, 0.74, 0.69],
            ["Night Drive", 124, 0.83, 0.77],
            ["Sunset Loop", 108, 0.61, 0.57],
        ],
        columns=["name", "tempo", "energy", "valence"],
    )


def _feature_matrix(songs_df: pd.DataFrame) -> np.ndarray:
    features = songs_df[["tempo", "energy", "valence"]].astype(float).to_numpy()
    mins = features.min(axis=0)
    maxs = features.max(axis=0)
    denom = maxs - mins
    denom[denom == 0] = 1.0
    return (features - mins) / denom


def _cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized = matrix / norms
    return normalized @ normalized.T


def _skip_probability(similarity: float, config: ExperimentConfig) -> float:
    similarity_0_1 = (similarity + 1.0) / 2.0
    dissimilarity = 1.0 - similarity_0_1
    skip_prob = config.base_skip_probability + config.dissimilarity_skip_weight * dissimilarity
    return float(np.clip(skip_prob, 0.01, 0.99))


def _pick_next_random(current_idx: int, n_songs: int, rng: np.random.Generator) -> int:
    choices = np.arange(n_songs)
    choices = choices[choices != current_idx]
    return int(rng.choice(choices))


def _pick_next_similar(current_idx: int, similarity_matrix: np.ndarray, rng: np.random.Generator) -> int:
    sims = similarity_matrix[current_idx].copy()
    sims[current_idx] = -np.inf
    ranked = np.argsort(sims)[::-1]
    top_k = ranked[:3]
    weights = np.array([0.6, 0.3, 0.1], dtype=float)[: len(top_k)]
    weights = weights / weights.sum()
    return int(rng.choice(top_k, p=weights))


def _simulate_variant(
    variant_name: str,
    similarity_matrix: np.ndarray,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> dict:
    n_songs = similarity_matrix.shape[0]
    total_session_length = 0
    total_skips = 0
    total_recommendations = 0

    for _ in range(config.num_users):
        current_idx = int(rng.integers(0, n_songs))
        session_length = 1

        for _ in range(config.max_steps_per_user):
            if variant_name == "A_random":
                next_idx = _pick_next_random(current_idx, n_songs, rng)
            else:
                next_idx = _pick_next_similar(current_idx, similarity_matrix, rng)

            similarity = float(similarity_matrix[current_idx, next_idx])
            skip_prob = _skip_probability(similarity, config)
            skipped = bool(rng.random() < skip_prob)

            total_recommendations += 1
            if skipped:
                total_skips += 1
                if rng.random() > config.continue_after_skip_probability:
                    break
            else:
                session_length += 1
                current_idx = next_idx
                if rng.random() > config.continue_after_listen_probability:
                    break

        total_session_length += session_length

    avg_session_length = total_session_length / config.num_users
    skip_rate = total_skips / total_recommendations if total_recommendations else 0.0

    return {
        "variant": variant_name,
        "users": config.num_users,
        "avg_session_length": round(avg_session_length, 3),
        "skip_rate": round(skip_rate, 3),
    }


def run_ab_test(
    songs_df: pd.DataFrame | object | None = None,
    config: ExperimentConfig = ExperimentConfig(),
) -> pd.DataFrame:
    """
    Simulate user sessions for:
    - Version A: random next-song recommendation
    - Version B: similarity-based recommendation

    Returns a dataframe with average session length and skip rate.
    """
    # Backward-compatible: if caller passes a non-DataFrame object (for example a DB manager),
    # ignore it and use sample songs unless a DataFrame is explicitly provided.
    if songs_df is None or not isinstance(songs_df, pd.DataFrame):
        songs_df = _sample_dataset()

    required_columns = ["name", "tempo", "energy", "valence"]
    missing = [c for c in required_columns if c not in songs_df.columns]
    if missing:
        raise ValueError(f"songs_df is missing required columns: {missing}")

    matrix = _feature_matrix(songs_df)
    similarity_matrix = _cosine_similarity_matrix(matrix)

    rng = np.random.default_rng(config.random_seed)
    results = [
        _simulate_variant("A_random", similarity_matrix, config, rng),
        _simulate_variant("B_similarity", similarity_matrix, config, rng),
    ]
    result_df = pd.DataFrame(results)

    print("=== A/B Recommendation Experiment (1000 Users) ===")
    print("A_random      : randomly pick next song")
    print("B_similarity  : pick from most similar songs (cosine similarity)")
    print("")
    print(result_df.to_string(index=False))

    return result_df


if __name__ == "__main__":
    run_ab_test()
