# AI-Powered Music Transition Engine

## Problem
Music streaming platforms often create abrupt transitions between songs (e.g., slow → high energy), leading to poor listening experience and higher skip rates.

## Solution
Designed a recommendation system that selects the next song based on similarity in tempo, energy, and mood (valence), ensuring smoother transitions and better session continuity.

## How It Works

1. Represent each song as a vector:
   - Tempo
   - Energy
   - Valence (mood)

2. Calculate similarity using cosine similarity

3. Recommend top N closest songs

## Data Layer
- Stored and queried song data using SQLite
- Analyzed:
  - average tempo trends
  - energy distribution
  - mood-based grouping

## Experimentation (A/B Testing)

Simulated 1000 users:

- **Version A:** Random recommendations  
- **Version B:** Similarity-based recommendations  

### Results:
- Reduced skip rate (simulated)
- Increased session continuity
- Improved listening experience

## Key Insight
Small improvements in transition smoothness can significantly enhance user engagement.

## Tech Stack
- Python (pandas, numpy)
- Cosine Similarity
- SQLite
- Streamlit (optional UI)

## Future Improvements
- Use real user data instead of simulation
- Integrate ML models (collaborative filtering)
- Personalization based on listening history
